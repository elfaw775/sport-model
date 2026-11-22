import {readingGuide} from "./reading-guide.js";

const book = document.querySelector("#example-book");
const escape = value => String(value).replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
const imageURL = record => "../samples/" + encodeURIComponent(record.public_filename);
const title = record => record.id + " 号 · 球节" + record.projection + "片";
const findings = record => {
  const count = record.detections.length;
  return (record.screening.predicted_positive ? "整片提示可疑" : "整片未触发提示") + "，" +
    (count ? "定位到 " + count + " 处候选区域" : "未定位到候选区域");
};

function index(records) {
  document.title = "只读示例病例 · EQUIVISION";
  book.innerHTML = `<header class="examples-heading"><h1>只读示例病例</h1>
    <p>选择一张运动马球节 X 线片，直接查看已生成的模型报告，或将模型候选框与研究数据集的原有标注对照。</p></header>
    <div class="example-grid">${records.map(record => `<a class="example-card" href="#${encodeURIComponent(record.id)}">
      <img src="${imageURL(record)}" alt="${escape(title(record))}">
      <div><h2>${escape(title(record))}</h2><p>${findings(record)}</p><span>查看示例报告 →</span></div></a>`).join("")}</div>
    <div class="example-actions"><a class="primary" href="../#new">用体验片新建检查 →</a></div>`;
}

function outlines(boxes, prefix, width) {
  const fontSize = Math.max(10, Math.round(width * 0.023));
  return boxes.map(({x1,y1,x2,y2}, index) => `<rect x="${x1}" y="${y1}" width="${x2-x1}" height="${y2-y1}"/>
    <text x="${x1}" y="${Math.max(fontSize,y1-fontSize*0.3)}" font-size="${fontSize}">${prefix} ${index+1}</text>`).join("");
}

function report(record) {
  document.title = title(record) + " · EQUIVISION 示例报告";
  const aiCount = record.detections.length, referenceCount = record.reference_boxes.length;
  book.innerHTML = `<header class="examples-heading"><a href="#">← 全部示例病例</a>
    <h1>${escape(title(record))}</h1><p>只读示例报告 · 研究样片 ${escape(record.public_filename)} · ${record.width} × ${record.height} px</p></header>
    <div class="example-report"><section class="example-film"><figure class="example-figure">
      <svg class="radiograph" viewBox="0 0 ${record.width} ${record.height}" role="img" aria-label="${escape(title(record))}及所选区域">
        <image href="${imageURL(record)}" width="${record.width}" height="${record.height}"/>
        <g id="example-ai">${outlines(record.detections,"AI",record.width)}</g>
        <g id="example-reference" visibility="hidden">${outlines(record.reference_boxes,"参考",record.width)}</g></svg></figure>
      <div class="example-actions layer-switches"><label><input type="checkbox" data-toggle="example-ai" ${aiCount ? "checked" : "disabled"}> AI 候选框（${aiCount}）</label>
        <label><input type="checkbox" data-toggle="example-reference" ${referenceCount ? "" : "disabled"}> 数据集参考框（${referenceCount}）</label></div></section>
    <section class="card example-findings"><h2>模型分析结果</h2>${readingGuide(record, escape, false)}
      <h3>对照解读</h3><p>${referenceCount
        ? "研究数据集在这张片上留有 " + referenceCount + " 处病灶标注。" + (aiCount ? "开启参考框可比较两套框的位置与范围。" : "定位模型遗漏了数据集标出的区域。开启参考框可查看该处。")
        : "研究数据集在这张片上没有区域标注。"} 模型结果仅供研究参考。</p>
      <div class="example-actions"><a class="primary" href="../#new">选择体验片，亲自分析 →</a></div></section></div>`;
  for (const toggle of book.querySelectorAll("[data-toggle]")) {
    toggle.onchange = () => document.getElementById(toggle.dataset.toggle).setAttribute("visibility", toggle.checked ? "visible" : "hidden");
  }
}

try {
  const response = await fetch("./example-reports.json");
  if (!response.ok) throw new Error();
  const records = await response.json();
  const render = () => {
    const record = records.find(item => item.id === location.hash.slice(1));
    if (record) report(record); else index(records);
  };
  window.addEventListener("hashchange", () => { render(); book.focus({preventScroll:true}); window.scrollTo(0,0); });
  render();
} catch {
  book.innerHTML = '<h1>示例报告暂时无法载入</h1><p>请检查网络后重试。</p><button class="secondary">重新载入</button>';
  book.querySelector("button").onclick = () => location.reload();
}
