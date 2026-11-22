// One pair of model sessions per open page. Site startup itself does not download weights.
let worker;
let preparation;
let stage = "";
const jobs = new Map();
const listeners = new Set();

function announce(text) {
  stage = text;
  for (const update of listeners) update(text);
}

function reset(reason) {
  worker?.terminate();
  worker = undefined;
  preparation = undefined;
  for (const job of jobs.values()) job.reject(reason);
  jobs.clear();
  announce(reason.message);
}

function request(type, blob) {
  return new Promise((resolve, reject) => {
    if (!worker) {
      worker = new Worker(new URL("./inference.js", import.meta.url), {type:"module"});
      worker.onerror = () => reset(new Error("浏览器模型程序载入失败，请检查网络后重试。"));
      worker.onmessage = ({data}) => {
        if (data.stage) { announce(data.stage); return; }
        const job = jobs.get(data.id);
        if (!job) return;
        jobs.delete(data.id);
        if (data.error) job.reject(new Error(data.error));
        else job.resolve(data.result);
      };
    }
    const id = crypto.randomUUID();
    jobs.set(id, {resolve,reject});
    try { worker.postMessage({id,type,blob}); }
    catch (error) { jobs.delete(id); reject(error); }
  });
}

export async function prepareModels(update) {
  if (update) {
    listeners.add(update);
    update(stage || "正在准备模型…");
  }
  try {
    preparation ??= request("prepare").then(() => announce("两个模型已就绪，可以分析。")).catch(error => {
      // ORT retains a failed WebAssembly initialization within a worker. Retry with a fresh one.
      reset(error);
      throw error;
    });
    await preparation;
  } finally {
    if (update) listeners.delete(update);
  }
}

export async function analyzeFilm(blob) {
  await prepareModels();
  return request("analyze", blob);
}
