"""Assess a frozen DINO patch probe fitted on lesion boxes and a fixed whole-film fusion."""
import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import torch
from torch.nn.functional import normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "training/scripts"))
from coco2yolo import rotated_bounds
from evaluate import fit_probe, probe_scores
from hf_features import FrozenEncoder, SIZE, transform
from nested import midpoint_threshold
from study_paths import DATA, PUBLISHED

STUDY = PUBLISHED / "cv84"
COCO = HERE.parents[1] / "training/source/coco/instances_default.json"
VIEWS = {"black_center": (0, True), "gray32_center": (32, True), "black_start": (0, False)}
GRID = 24
PATCH = SIZE // GRID


@dataclass
class Film:
    content: Image.Image
    size: tuple


@dataclass
class View:
    tokens: np.ndarray
    centers: np.ndarray
    whole: np.ndarray


def load_films(ids):
    paths = {p.stem:p for p in (DATA/"images").glob("*/*") if p.is_file()}
    assert set(paths) == set(ids)
    films = []
    for name in ids:
        with Image.open(paths[name]) as im:
            if im.getexif().get(274, 1) != 1:
                raise ValueError(f"COCO orientation needs explicit mapping: {name}")
            w, h = im.size
            scale = SIZE / max(w, h)
            content = im.convert("L").resize((max(1, round(w*scale)), max(1, round(h*scale))),
                                               Image.Resampling.BICUBIC)
        films.append(Film(content, (w,h)))
    return films


def load_annotations(ids, films, y):
    coco = json.loads(COCO.read_bytes())
    by_id = {r["id"]:r for r in coco["images"]}
    boxes = {name:[] for name in ids}
    dimensions = {name:f.size for name,f in zip(ids,films,strict=True)}
    lesion, = [c["id"] for c in coco["categories"] if c["name"] == "病变"]
    for ann in coco["annotations"]:
        if ann["category_id"] != lesion:
            continue
        image = by_id[ann["image_id"]]
        name = Path(image["file_name"]).stem
        assert dimensions[name] == (image["width"],image["height"])
        boxes[name].append((ann["bbox"], float(ann.get("attributes",{}).get("rotation",0))))
    aligned = [boxes[k] for k in ids]
    assert sum(map(len,aligned)) == 81 and np.array_equal(y, [bool(b) for b in aligned])
    return aligned


def prepare(film, view):
    color, center = VIEWS[view]
    w,h = film.content.size
    left,top = ((SIZE-w)//2,(SIZE-h)//2) if center else (0,0)
    frame = Image.new("L",(SIZE,SIZE),color)
    frame.paste(film.content,(left,top))
    rows,cols = np.indices((GRID,GRID))
    canvas_xy = (np.column_stack([cols.ravel(),rows.ravel()]) + .5) * PATCH
    original = (canvas_xy - [left,top]) * (np.asarray(film.size)/[w,h])
    valid = np.all((original >= 0) & (original < film.size),axis=1)
    return transform(frame), valid, original[valid]


@torch.inference_mode()
def encode(encoder, films, view):
    blocks = []
    for i in range(0,len(films),6):
        batch = [prepare(f,view) for f in films[i:i+6]]
        pixels = torch.stack([b[0] for b in batch]).to(encoder.device)
        hidden = encoder.model(pixel_values=pixels).last_hidden_state
        tokens = hidden[:,1:]
        assert tokens.shape[1] == GRID**2
        whole = torch.cat([normalize(hidden[:,0],dim=-1),normalize(tokens.mean(1),dim=-1)],dim=1).cpu().numpy()
        local = (normalize(tokens,dim=-1) * np.sqrt(tokens.shape[-1])).cpu().numpy()
        for j,(_,valid,xy) in enumerate(batch):
            assert len(xy) >= 3 and np.isfinite(local[j]).all() and np.isfinite(whole[j]).all()
            blocks.append(View(local[j,valid],xy,whole[j]))
    return blocks


def masks(centers, bbox, rotation):
    x,y,w,h = bbox
    left,top,right,bottom = rotated_bounds(bbox,rotation)
    in_envelope = np.all((centers >= [left,top]) & (centers <= [right,bottom]),axis=1)
    dx,dy = (centers-[x+w/2,y+h/2]).T
    angle = np.deg2rad(rotation)
    local_x = dx*np.cos(angle) + dy*np.sin(angle)
    local_y = -dx*np.sin(angle) + dy*np.cos(angle)
    in_rotated = (abs(local_x) <= w/2) & (abs(local_y) <= h/2)
    return in_envelope,in_rotated


def train_local(fit_views, fit_boxes):
    data,targets,weights = [],[],[]
    missing = 0
    for view, boxes in zip(fit_views,fit_boxes,strict=True):
        target = np.zeros(len(view.centers),dtype=bool)
        for bbox,angle in boxes:
            covered,_ = masks(view.centers,bbox,angle)
            if not covered.any():
                x,y,w,h = bbox
                covered[np.argmin(np.sum((view.centers-[x+w/2,y+h/2])**2,axis=1))] = True
                missing += 1
            target |= covered
        # One unit per available class per film before balancing total class weights.
        weight = np.where(target,1/max(1,sum(target)),1/max(1,sum(~target)))
        data.append(view.tokens)
        targets.append(target)
        weights.append(weight)
    y = np.concatenate(targets)
    w = np.concatenate(weights)
    for cls in (False,True):
        w[y==cls] *= len(fit_views)/(2*w[y==cls].sum())
    probe = LogisticRegression(C=1,solver="lbfgs",max_iter=1000).fit(np.vstack(data),y,sample_weight=w)
    assert probe.n_iter_[0] < 1000
    return probe.coef_[0],probe.intercept_[0],missing


def apply_heads(view, global_head, coef, intercept):
    probs = expit(view.tokens @ coef + intercept)
    point = view.centers[np.argmax(probs)]
    local = float(np.partition(probs,-3)[-3:].mean())
    whole = float(probe_scores(global_head,view.whole[None])[0])
    return {"global":whole,"local":local,"fusion":(whole+local)/2},point


def locate(view, point, size, boxes):
    xy = np.vstack([point, np.asarray(size)/2, view.centers])
    env = np.zeros(len(xy),dtype=bool)
    rot = env.copy()
    for bbox,angle in boxes:
        a,b = masks(xy,bbox,angle)
        env |= a
        rot |= b
    return dict(point_aabb=int(env[0]),center_aabb=int(env[1]),random_aabb=float(env[2:].mean()),
                point_rotated=int(rot[0]),center_rotated=int(rot[1]),random_rotated=float(rot[2:].mean()))


def report(rows):
    result = {}
    for view in VIEWS:
        selected = [r for r in rows if r["view"]==view]
        assert len({r["image"] for r in selected}) == len(selected) == 84
        y = np.array([r["truth"] for r in selected],dtype=bool)
        base = np.array([r["global_score"] >= r["global_threshold"] for r in selected])
        group = {}
        for model in ("global","local","fusion"):
            pred = np.array([r[model+"_score"] >= r[model+"_threshold"] for r in selected])
            group[model] = dict(correct=int(sum(pred==y)),tp=int(sum(y & pred)),tn=int(sum(~y & ~pred)),
                fp=int(sum(~y & pred)),fn=int(sum(y & ~pred)),fixed_vs_global=int(sum((pred==y)&(base!=y))),
                lost_vs_global=int(sum((pred!=y)&(base==y))))
        hits = {}
        for metric in ("point_aabb","center_aabb","random_aabb","point_rotated","center_rotated","random_rotated"):
            counts = [r[metric] for r in selected if r["truth"]]
            assert len(counts)==68
            hits[metric] = sum(counts)
        result[view] = dict(models=group,localization_positive_films=hits)
    return result


def run(output, weights):
    foldmap = json.loads((STUDY/"folds.json").read_bytes())
    ids,y = foldmap["ids"],np.array(foldmap["labels"])
    partitions = json.loads((STUDY/"detector-report.json").read_bytes())["folds"]
    with (STUDY/"three-model-oof.csv").open(encoding="utf-8-sig") as stream:
        prior = {r["image"]:r for r in csv.DictReader(stream)}
    films = load_films(ids)
    annotations = load_annotations(ids,films,y)
    encoder = FrozenEncoder("dinov2",device="cuda")
    views = {name:encode(encoder,films,name) for name in VIEWS}
    original = views["black_center"]
    rows,params,fold_info = [],{},[]
    global_x = np.stack([v.whole for v in original])
    for f in partitions:
        n = f["fold"]+1
        fit,cal,held = (f[k] for k in ("fit_indices","calibration_indices","test_indices"))
        assert sorted(fit+cal+held)==list(range(84))
        assert all(int(prior[ids[i]]["fold"])==n for i in held)
        ghead = fit_probe(global_x[fit],y[fit])
        coef,intercept,missing = train_local([original[i] for i in fit],[annotations[i] for i in fit])
        scores = [apply_heads(original[i],ghead,coef,intercept)[0] for i in cal]
        saved_global, = {float(prior[ids[i]]["dino_threshold"]) for i in held}
        cut = {"global":saved_global}
        for key in ("local","fusion"):
            cut[key] = midpoint_threshold(y[cal],np.array([s[key] for s in scores]))
        for key,val in ghead.items():
            params[f"fold{n}_global_{key}"] = val
        params.update({f"fold{n}_patch_coef":coef,f"fold{n}_patch_intercept":intercept})
        params.update({f"fold{n}_{k}_threshold":v for k,v in cut.items()})
        fold_info.append(dict(fold=n,fit=len(fit),calibration=len(cal),held=len(held),nearest_fit_fallback_boxes=missing))
        for view_name,features in views.items():
            for i in held:
                view = features[i]
                scores,point = apply_heads(view,ghead,coef,intercept)
                if view_name=="black_center":
                    assert abs(scores["global"]-float(prior[ids[i]]["dino_score"]))<.002
                    assert int(scores["global"]>=cut["global"])==int(prior[ids[i]]["dino"])
                # Scoring and point selection above have no access to this film's annotations.
                metrics = locate(view,point,films[i].size,annotations[i]) if y[i] else {}
                rows.append(dict(image=ids[i],fold=n,view=view_name,truth=int(y[i]),width=films[i].size[0],
                    height=films[i].size[1],point_x=float(point[0]),point_y=float(point[1]),
                    **{k+"_score":v for k,v in scores.items()},**{k+"_threshold":v for k,v in cut.items()},**metrics))
    output.mkdir(parents=True,exist_ok=True)
    weights.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(weights,**params)
    path=output/"predictions.csv"
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",encoding="utf-8-sig",newline="") as out:
        w=csv.DictWriter(out,fieldnames=fields); w.writeheader(); w.writerows(rows)
    results = report(rows)
    summary = dict(views=results,folds=fold_info,n=84,positives=68,boxes=81,
        plan_sha256=sha256((HERE/"box_token_plan.md").read_bytes()).hexdigest(),
        annotations_sha256=sha256(COCO.read_bytes()).hexdigest(),
        predictions_sha256=sha256(path.read_bytes()).hexdigest(),weights_sha256=sha256(weights.read_bytes()).hexdigest(),
        encoder="facebook/dinov2-small",device=torch.cuda.get_device_name())
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8",newline="\n")
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--weights",type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output,args.weights)
