import {handleCases} from "./cases.js";

const site = new URL(self.registration.scope);
const sessionKey = new URL("session", site);
const team = {wangyiyi:"王一伊", zhushijia:"朱诗嘉", alimire:"阿里米热", liqihang:"李其杭"};
let sampleLibrary;

self.addEventListener("install", event => event.waitUntil(self.skipWaiting()));
self.addEventListener("activate", event => event.waitUntil(self.clients.claim()));

function error(message, status) {
  return Response.json({detail:message}, {status});
}

async function samples(route) {
  sampleLibrary ??= fetch(new URL("samples/catalog.json", site)).then(async response => {
    if (!response.ok) throw new Error("无法读取样片目录。");
    return response.json();
  }).catch(reason => { sampleLibrary = undefined; throw reason; });
  const {catalog, aliases} = await sampleLibrary;
  const urlFor = file => new URL("api/demo-images/" + file, site).pathname;
  if (route === "/api/demo-images") {
    return Response.json(catalog.map(item => ({...item, url:urlFor(item.filename)})));
  }
  const name = decodeURIComponent(route.slice("/api/demo-images/".length));
  const file = Object.hasOwn(aliases, name) ? aliases[name] : name;
  if (!catalog.some(item => item.filename === file)) return error("样片不存在。", 404);
  const response = await fetch(new URL("samples/" + file, site));
  if (!response.ok) return error("无法载入所选样片。", response.status);
  return new Response(response.body, {headers:{"Content-Type":response.headers.get("Content-Type"),
    "Content-Location":urlFor(file)}});
}

async function inference(clientId, blob) {
  const client = await self.clients.get(clientId);
  if (!client) throw new Error("请在打开的病例页面启动分析。");
  const channel = new MessageChannel();
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      channel.port1.close();
      reject(new Error("模型载入超时，请检查网络连接并重试。"));
    }, 240000);
    channel.port1.onmessage = ({data}) => {
      clearTimeout(timeout);
      channel.port1.close();
      if (data.error) reject(new Error(data.error));
      else resolve(data.result);
    };
    client.postMessage({type:"analyze-film", blob}, [channel.port2]);
  });
}

async function dispatch(request, route, clientId) {
  const method = request.method;
  if (route === "/api/login-options" && method === "GET") return Response.json(team);
  const session = await caches.open("equivision-demo-session" + site.pathname);
  if (route === "/api/login" && method === "POST") {
    const {username, password} = await request.json();
    if (!Object.hasOwn(team, username) || password !== username + "sportsHorse") return error("账号或密码不正确",401);
    const user = {username, display_name:team[username]};
    await session.put(sessionKey, Response.json({user, expires:Date.now()+8*3600000}));
    return Response.json(user);
  }
  if (route === "/api/logout" && method === "POST") {
    await session.delete(sessionKey);
    return new Response(null,{status:204});
  }
  const saved = await (await session.match(sessionKey))?.json();
  if (!saved || saved.expires <= Date.now()) return error("请先登录工作台",401);
  if (route === "/api/me" && method === "GET") return Response.json(saved.user);
  if (method === "GET" && (route === "/api/demo-images" || route.startsWith("/api/demo-images/"))) return samples(route);
  return handleCases(request, route, saved.user, blob => inference(clientId, blob));
}

self.addEventListener("fetch", event => {
  const url = new URL(event.request.url);
  if (url.origin !== site.origin || !url.pathname.startsWith(site.pathname + "api/")) return;
  const route = "/" + url.pathname.slice(site.pathname.length);
  event.respondWith(dispatch(event.request, route, event.clientId).catch(reason => {
    if (reason.name === "QuotaExceededError") return error("浏览器存储空间不足，病例未保存，请释放空间后重试。",507);
    return error(reason.message || "工作台操作失败，请重试。",reason.status || 500);
  }));
});
