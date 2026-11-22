// Render per-film exports using the same resources as the download links.
export function mountArchiveImages(host, record, escape, openDialog) {
  if (!host || (!record.archived_at && record.status !== "reviewed")) return;
  const evidence = record.signed_evidence || record;
  if (!evidence.images?.length) return;

  function film(image, index) {
    const ai = image.id === "primary" ? evidence.detections?.length || 0 : 0;
    const manual = image.annotations?.length || 0;
    const layers = [];
    if (ai && manual) layers.push(["combined", "AI 与人工叠加"]);
    if (ai) layers.push(["ai", "仅 AI 定位框"]);
    if (manual) layers.push(["manual", "仅人工标注"]);
    const original = escape(image.url);
    const title = escape((index ? "附图 " + index : "主图") + " · " + image.projection + " · " + image.filename);
    const download = '<a class="archive-original" href="' + original + '" download="' + escape(image.filename) + '">下载原片</a>';
    const source = image.id === "primary" ? '<span class="ai-key">AI 候选 ' + ai + ' 处</span>' : "对照影像";
    const header = '<b>' + title + '</b><span class="layer-counts">' + source + ' · <span class="manual-key">人工 ' + manual + ' 处</span></span>';
    const cards = layers.map(([key, name]) => {
      const base = original + "/layers/" + key;
      return `<figure class="layer-card" data-layer="${key}"><button type="button" class="layer-preview" data-preview aria-label="放大查看：${title}，${name}"><img loading="lazy" src="${base}.png" alt="${title} · ${name}"></button>
        <figcaption><h4>${name}</h4><div class="layer-downloads"><a href="${base}.png" download aria-label="下载${name} PNG">标注图 PNG</a>
        <a href="${base}.zip" download aria-label="下载${name} COCO 训练包">训练包 ZIP</a></div></figcaption></figure>`;
    }).join("");
    const content = (cards ? '<div class="layer-grid">' + cards + '</div>' :
      `<div class="unmarked-film"><button type="button" class="layer-preview" data-preview aria-label="放大查看${title}原片"><img loading="lazy" src="${original}" alt="${title}原片"></button><span>本片没有已保存的区域标注。</span></div>`) + download;
    return index
      ? `<details class="archive-film" data-film="${escape(image.id)}"><summary>${header}</summary>${content}</details>`
      : `<section class="archive-film" data-film="${escape(image.id)}"><h3>${header}</h3>${content}</section>`;
  }

  host.innerHTML = `<section class="card archive-gallery"><div class="panel-title"><h2>影像分层与导出</h2></div>
    <p class="gallery-guide">${record.signed_evidence ? "采用签署时的区域记录。" : "采用当前留存的区域记录。"}训练包包含原片和 CVAT COCO 1.0 标注。</p>
    ${evidence.images.map(film).join("")}</section>`;
  host.onclick = event => {
    const preview = event.target.closest("[data-preview]");
    if (!preview) return;
    const image = preview.querySelector("img");
    openDialog(`<div class="spread"><h2 id="dialog-title">${escape(image.alt)}</h2><button id="close-layer" class="secondary" type="button" autofocus>关闭</button></div>
      <img class="expanded-layer" src="${escape(image.src)}" alt="${escape(image.alt)}">`, "close-layer");
  };
}
