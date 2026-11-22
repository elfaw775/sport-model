"""Printable, self-contained reports from the evidence saved at signature time."""

from html import escape


def signed_evidence(case):
    fields = ("id", "horse_name", "comment", "created_by", "created_at", "images", "report",
              "screening", "detections", "model", "threshold", "analyzed_at", "reviewer",
              "reviewed_at", "empty_review_confirmed")
    return {field: case[field] for field in fields}


def text(value):
    return escape(str(value)) if value is not None and value != "" else "未记录"


def table(headers, rows):
    titles = "".join(f'<th scope="col">{text(label)}</th>' for label in headers)
    body = "".join("<tr>" + "".join(f"<td>{text(value)}</td>" for value in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{titles}</tr></thead><tbody>{body}</tbody></table>"


def point_list(points):
    return "、".join(f"({p['x']:g}, {p['y']:g})" for p in points)


def image_record(image, position):
    role = "主图" if position == 0 else f"附图 {position}"
    parts = [f"<h3>{role} · {text(image['filename'])}</h3>",
             f"<p>{text(image['projection'])} · {image['width']} × {image['height']} px · "
             f"质量核对 {text(image['quality'])}</p>"]
    marks, measures = image["annotations"], image["measurements"]
    if marks:
        parts.append(table(("人工区域", "形状", "原图顶点（px）", "最后编辑人"),
                           ((m["label"], "矩形" if m["kind"] == "rectangle" else
                             ("四边形" if len(m["points"]) == 4 else "多边形"),
                             point_list(m["points"]), m["updated_by"]) for m in marks)))
    if measures:
        parts.append(table(("人工测量", "读数", "原图取点（px）", "记录人"),
                           (("距离" if m["kind"] == "distance" else "角度", f"{m['value']:.2f} {m['unit']}",
                             point_list(m["points"]), m["created_by"]) for m in measures)))
    if not (marks or measures):
        parts.append("<p>无人工区域或测量记录。</p>")
    return "".join(parts)


def report_html(case, *, historical):
    report = case["report"]
    provenance = "历史报告 · 按现存病例资料整理" if historical else "签署时保存的病例资料"
    parts = [f"<header><p>EQUIVISION · 运动马影像工作台</p><h1>影像复核报告</h1>"
             f"<p>{text(case['id'])} · {provenance}</p></header>",
             f"<h2>{text(case['horse_name'])}</h2><p>登记时间　{text(case['created_at'])}</p>",
             f"<p>就诊备注　{text(case['comment'])}</p>",
             "<h2>人工复核</h2>",
             f"<h3>影像所见</h3><p>{text(report['findings'])}</p>",
             f"<h3>复核结论</h3><p>{text(report['conclusion'])}</p>"]
    if case["empty_review_confirmed"]:
        parts.append("<p>复核人已二次确认两项留空。</p>")
    parts.append(f"<p class='signature'>复核人　{text(case['reviewer'])}<br>签署时间　{text(case['reviewed_at'])}</p>"
                 "<h2>模型辅助分析（主图）</h2>")
    film = case["screening"]
    if film:
        result = "达到提示阈值" if film["predicted_positive"] else "低于提示阈值"
        parts.append(table(("整片筛查模型", "筛查分数", "提示阈值", "结果"),
                           [(film["model"], film["score"], film["threshold"], result)]))
    else:
        parts.append("<p>此记录未保存整片筛查结果。</p>")
    boxes = case["detections"]
    parts.append(f"<p>定位模型　{text(case['model'])}<br>候选阈值　{text(case['threshold'])}"
                 f"<br>候选区域　{len(boxes)} 个</p>")
    if boxes:
        parts.append(table(("候选框", "分数", "左 X", "上 Y", "右 X", "下 Y"),
                           ((i, b.get("score"), *(round(b[k], 2) if k in b else None for k in ("x1", "y1", "x2", "y2")))
                            for i, b in enumerate(boxes, 1))))
    parts.append(f"<p>分析时间　{text(case['analyzed_at'])}。模型输出仅供人工阅片参考。</p>"
                 "<h2>检查影像与阅片记录</h2>")
    parts.extend(image_record(image, i) for i, image in enumerate(case["images"]))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>{text(case['id'])} 影像复核报告</title><style>
*{{box-sizing:border-box}}body{{max-width:800px;margin:0 auto;padding:32px;color:#193b30;
font:16px/1.6 system-ui,sans-serif;overflow-wrap:anywhere}}
header{{border-bottom:3px solid #245843}}h1,h2,h3,p{{margin:.55em 0}}h2{{margin-top:1.3em}}
h3{{font-size:1.02em}}p,td{{white-space:pre-wrap}}table{{width:100%;border-collapse:collapse;margin:.8em 0}}
th,td{{border:1px solid #c8d5cc;text-align:left;padding:.35em .6em}}th{{background:#eff4ed}}
.signature{{border-left:3px solid #245843;padding-left:1em}}
@page{{size:A4;margin:15mm}}@media print{{body{{max-width:none;margin:0;padding:0;font-size:10.5pt}}
h2,h3{{break-after:avoid}}tr,.signature{{break-inside:avoid}}}}
</style></head><body><main>{''.join(parts)}</main></body></html>"""
