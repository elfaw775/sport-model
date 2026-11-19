# Whole-film lesion classification on 84 equine radiographs

The image classifier answers “does this film contain an annotated lesion?” A positive film has at least one lesion annotation. The public archive contains 68 positive films, 16 negative films and 81 lesion boxes. The downloadable research head returns one score per JPG or PNG. A separate YOLO detector produces locations.

## What the experiments measured

| Model or control | Correct / 84 | False positives | Missed positive films |
| --- | ---: | ---: | ---: |
| Nested choice among seven frozen encoders | 83 | 0 | 1 |
| Frozen COCO YOLO11s backbone with the same probe | 80 | 2 | 2 |
| Pixel height alone | 81 | 1 | 2 |
| YOLO11s lesion detector retrained per fold | 62 | 2 | 20 |

These are internal out-of-fold predictions, one per film for seed 42. Each of five outer training sets supplied four inner folds for encoder selection and the decision threshold. The classifier's repeat totals were 83, 82 and 84 correct for seeds 42, 73 and 101, using the same 84 films each time. The downloadable research head uses DINOv2 Small with a threshold of 0.1788344393. A separate four-fold pass chose this encoder and cutoff, then fitted the head on all 84. Its fitted score on these images cannot estimate performance on future cases. [Plan](cv84/plan.md) · [Results for every model](published/cv84/summary.json) · [Fold memberships](published/cv84/folds.json) · [Primary decisions per film](published/cv84/primary-predictions.csv)

The retrained detector used the same five outer holdouts. Within each fold 50 or 51 films fitted detector weights and 17 selected a confidence threshold. The classifier probes could refit on all 67 or 68 outer-training films after inner selection. At box confidence 0.15 and intersection-over-union 0.50 the detector matched 16 of 81 annotated lesions, produced 26 false boxes and missed 65 lesions. In a post hoc check using the detector's exact fitting and calibration IDs, fixed DINOv2 Small classified 82/84 correctly. DINO alone was correct on 21 films and YOLO alone on one. The classifier supplies no coordinates. [Detector summary](published/cv84/detector-report.json) · [Detector decisions per film](published/cv84/detector-oof.csv)

The height-only rule was correct on 81 films. On 58 positives with an available disjoint, same-size comparison region, blurring the annotated region or the comparison region left every binary decision unchanged. Ten positives lacked a matching region. Image acquisition and export differences are plausible shortcuts. The dataset has no verified horse-level ID or independent external cohort. Grouping equal leading filename numbers placed 84 images in 70 candidate groups, including 14 pairs each containing one positive and one negative. This check cannot confirm horse identity. [Per-film audits and model provenance](published/evidence.zip)

An earlier experiment fitted on 59 films, chose thresholds on 13 and scored 12 historical holdout films. Its DINOv2 Small head made 0 false positives and 0 false negatives on those 12. Earlier project work had already examined this split. The later 84-film study reused all three partitions. Each published head has its own threshold. [Historical test report](published/holdout/test.json) · [Earlier experiment plan](evaluation-plan.md)

The repository also records Qwen3-VL-8B on the historical 12-film holdout. Direct reading classified 4 of 12 films correctly, and tuning on the 59 training films brought that to 6 of 12. Both variants located 2 of the 12 marked lesions. Qwen has a historical result only in this comparison. [Archived Qwen evaluation](../training/results/vision-model-scale-2026-11-18/README.md)

## Score a single image

From the repository root, use Python 3.13. The recorded NVIDIA environment used PyTorch 2.9.1, torchvision 0.24.1 and CUDA 12.8.

```powershell
python -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r image-study/requirements.txt
$film = (Get-ChildItem training/data/sporthorse_v2/images/test -File | Select-Object -First 1).FullName
& $py image-study/cv84/predict84.py $film --device cuda
```

Replace `$film` with your JPG or PNG path. Single-film inference supports `--device cpu` with a compatible PyTorch installation. The first run fetches the encoder at its pinned Hugging Face revision and verifies downloaded files and the bundled head. The score crosses the included cutoff when `predicted_positive` is true. It is an experimental model score whose calibration on new horses is unknown. To run the earlier 59-film head, use `image-study/predict_image.py` with the same arguments.

## Reproduce the 84-film analysis

These commands use PowerShell from the repository root. Point each new run at a fresh directory. Published artifacts are read from `image-study/published` and fresh outputs go into the chosen work directory. Model caches can be shared across runs.

```powershell
$env:PYTHONUTF8 = '1'
$env:SPORT_XRAY_RUN = Join-Path $PWD ('work/image-study-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$env:SPORT_XRAY_CACHE = Join-Path $env:LOCALAPPDATA 'sports-horse-xray'
$env:SPORT_XRAY_YOLO = Join-Path $env:SPORT_XRAY_CACHE 'weights/yolo11s.pt'
New-Item -ItemType Directory -Force (Split-Path $env:SPORT_XRAY_YOLO) | Out-Null
if (!(Test-Path $env:SPORT_XRAY_YOLO)) {
    curl.exe --fail --location https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt --output $env:SPORT_XRAY_YOLO
}
& $py image-study/cv84/folds.py
& $py image-study/hf_features.py --models dinov2 resnet18 --splits train val test --batch-size 6
& $py image-study/hf_features.py --models rad-dino --splits train val test --batch-size 3
& $py image-study/cv84/medical_features.py --batch-size 6
& $py image-study/cv84/vision_backbones.py --batch-size 2
& $py image-study/cv84/stability.py embed
& $py image-study/cv84/nested.py
& $py image-study/cv84/verify.py
& $py image-study/cv84/predict84.py $film --results (Join-Path $env:SPORT_XRAY_RUN 'cv84/results')
```

The frozen encoder pass uses image pixels and one unit-normalized embedding per film, including separately normalized CLS and mean patch components for the two general-purpose DINO models. Every head fits its scaler and class-balanced logistic regression on the fitting fold only. The inner loop ranks encoders by ROC-AUC, average precision and a fixed tie order, then selects an operating threshold by balanced accuracy. The frozen YOLO comparison uses a pooled COCO backbone feature. GPU batch size and software versions can affect near-tie scores. `verify.py` checks 2,772 prediction rows, 15 outer folds, thresholds and the five seed-42 saved heads.

### Localization, perturbations and grouping

```powershell
$env:SPORT_XRAY_DETECTOR_RUN = Join-Path $env:SPORT_XRAY_RUN 'detector-baseline'
& $py training/scripts/prepare_comparison.py --work $env:SPORT_XRAY_DETECTOR_RUN
& $py image-study/cv84/detection_folds.py
& $py image-study/cv84/detector_pairing.py
& $py image-study/cv84/matched_budget.py
& $py image-study/cv84/selected_stability.py
& $py image-study/cv84/oof_occlusion.py
& $py image-study/cv84/prefix_groups.py
& $py image-study/cv84/historical59.py
& $py image-study/cv84/stability.py dino
```

The preparation command regenerates rotation-corrected boxes from the repository's original COCO annotations. The detector can also prepare these automatically or accept `--labels PATH_TO_PREPARED_DATASET`. Its five 32-epoch GPU fits use a separate directory for each fold. The detector and frozen-embedding routes check the SHA-256 of the official COCO weights. If you already have the historical 59-film detector benchmark output, set `SPORT_XRAY_DETECTOR_RUN` to that directory to run the extra fitted detector perturbation check with `stability.py detector`. Rebuild that benchmark in a new directory with the linked training instructions and additional requirements. [Historical detector recipe](../training/docs/rfdetr-cuda.md)

The dataset defaults to `training/data/sporthorse_v2`. Set `SPORT_XRAY_DATA` to point classification scripts to a byte-equivalent local copy with its original split manifest and image folders. The repository's annotation converter reads source images and COCO annotations from its standard training layout. Cache paths default to the current user's local app data on Windows or home cache on Linux. All Hugging Face encoders are identified by revision and file digests in the three asset loaders.

The evidence ZIP contains the full outer prediction table, inner scores and thresholds, five seed-42 heads, detector boxes, paired comparisons, perturbation and occlusion audits, historical per-film predictions, feature provenance and original hash-bearing source files. The runnable copies adapt file locations for a checkout. Source hashes in the frozen experiment reports refer to the original files. Public encoder weights are fetched on demand. 
