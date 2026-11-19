function interpretation(screening, count) {
  const locations = count ? `定位到 ${count} 处候选区域` : "未定位到候选区域";
  if (!screening) {
    return {
      title: `历史记录：${locations}`,
      next: count ? "本次仅保存了定位结果。请核对主图上的 AI 候选框，并复核整张片。" : "本次仅保存了定位结果。请查看整张原片。"
    };
  }
  const finding = screening.predicted_positive ? "整片提示可疑" : "整片未触发提示";
  let next;
  if (count) {
    next = screening.predicted_positive
      ? "先核对主图上编号的 AI 候选框，再查看其余区域。"
      : "定位模型单独给出了提示。请逐一核对主图上编号的 AI 候选框，再查看其余区域。";
  } else {
    next = screening.predicted_positive
      ? "筛查模型提示了整张片，具体位置需要人工查找。请通览主图，可用矩形或多边形标出可疑处。"
      : "请按常规复核整张主图，并记录人工所见。";
  }
  return {title: `${finding}，${locations}`, next};
}

/** The same stored evidence is shown in the viewer and in the review report. */
export function readingGuide(record, escapeHTML) {
  const boxes = record.detections || [];
  const screening = record.screening;
  const {title, next} = interpretation(screening, boxes.length);
  const scores = boxes.map((box, i) => `AI 候选 ${String(i+1).padStart(2,"0")}：${Number(box.score).toFixed(3)}`);
  return `<div class="reading-guide">
    <h3 class="reading-title">${title}</h3>
    <p class="reading-next">${record.status === "reviewed" ? (record.empty_review_confirmed ? "报告已签署，复核人已确认本次人工所见与结论留空。" : "报告已签署，请查阅复核人填写的内容。") : next}</p>
    <details class="model-evidence"><summary>查看模型分数与判断依据</summary>
      <dl>
        <dt>整片筛查</dt><dd>${screening ? `分数 ${Number(screening.score).toPrecision(5)}，提示门槛 ${Number(screening.threshold).toPrecision(5)}。<small>${escapeHTML(screening.model)}</small>` : "本次历史分析未记录整片筛查。"}</dd>
        <dt>图上定位</dt><dd>${Number.isFinite(record.threshold) ? `候选框门槛 ${record.threshold.toFixed(2)}。` : "历史门槛未记录。"}${scores.length ? `<p>${scores.join("、")}</p>` : ""}<small>${record.model ? escapeHTML(record.model) : "本次历史分析未记录模型版本。"}</small></dd>
      </dl>
      <p>达到各自门槛时才显示提示。模型分数用于比较门槛，不能换算成患病概率。</p>
    </details></div>`;
}
