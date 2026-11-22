import {analyzeFilm} from "./model-client.js";
// GitHub Pages serves the workbench files. Its case API stays on this device.
document.documentElement.dataset.storage = "browser";
const page = document.getElementById("page");
async function boot() {
  if (!navigator.serviceWorker || !window.indexedDB) throw new Error("请使用支持本地存储的新版 Chrome 或 Edge 打开工作台。");
  navigator.serviceWorker.addEventListener("message", event => {
    if (event.source !== navigator.serviceWorker.controller || event.data?.type !== "analyze-film") return;
    const reply = event.ports[0];
    analyzeFilm(event.data.blob).then(result => reply.postMessage({result}),
      error => reply.postMessage({error:error.message})).finally(() => reply.close());
  });
  const script = new URL("./service-worker.js", import.meta.url);
  const controlledHere = () => navigator.serviceWorker.controller?.scriptURL === script.href;
  await navigator.serviceWorker.register(script, {type:"module", updateViaCache:"none"});
  if (!controlledHere()) {
    await new Promise((resolve, reject) => {
      const controlled = () => {
        if (!controlledHere()) return;
        clearTimeout(timer);
        navigator.serviceWorker.removeEventListener("controllerchange", controlled);
        resolve();
      };
      const timer = setTimeout(() => {
        navigator.serviceWorker.removeEventListener("controllerchange", controlled);
        reject(new Error("工作台未完成初始化，请刷新重试。"));
      }, 30000);
      navigator.serviceWorker.addEventListener("controllerchange", controlled);
      controlled();
    });
  }
  document.querySelector(".research").lastChild.textContent = " 浏览器演示 · 病例保存在本机";
  await import("./static/app.js");
}
boot().catch(error => {
  page.textContent = error.message + " ";
  const retry = document.createElement("button");
  retry.textContent = "重新载入";
  retry.className = "secondary";
  retry.onclick = () => location.reload();
  page.append(retry);
  page.removeAttribute("aria-busy");
});
