/* ch14 后台共享前端鉴权：会话、CSRF、401 跳登录、403 提示、退出。 */
(function () {
  "use strict";

  const state = { session: null, loading: null };

  class ForbiddenError extends Error {
    constructor(detail) {
      super(detail || "权限不足");
      this.name = "ForbiddenError";
      this.status = 403;
    }
  }

  function loginRedirect() {
    const next = location.pathname + location.search;
    location.replace("/admin/login?next=" + encodeURIComponent(next));
  }

  async function getSession(force = false) {
    if (state.session && !force) return state.session;
    if (state.loading && !force) return state.loading;
    state.loading = (async () => {
      const response = await fetch("/api/admin/session", { credentials: "same-origin" });
      if (response.status === 401) {
        loginRedirect();
        throw new Error("未登录");
      }
      if (!response.ok) throw new Error("会话读取失败（" + response.status + "）");
      state.session = await response.json();
      return state.session;
    })();
    try {
      return await state.loading;
    } finally {
      state.loading = null;
    }
  }

  async function request(url, options = {}) {
    const session = await getSession();
    const method = (options.method || "GET").toUpperCase();
    const headers = new Headers(options.headers || {});
    if (!["GET", "HEAD", "OPTIONS"].includes(method)) {
      headers.set("X-CSRF-Token", session.csrf_token);
    }
    const response = await fetch(url, {
      ...options,
      method,
      headers,
      credentials: "same-origin",
    });
    if (response.status === 401) {
      loginRedirect();
      throw new Error("未登录");
    }
    if (response.status === 403) {
      const body = await response.json().catch(() => ({}));
      throw new ForbiddenError(body.detail);
    }
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || ("请求失败（" + response.status + "）"));
    }
    return response;
  }

  async function requestJSON(url, options = {}) {
    return (await request(url, options)).json();
  }

  async function logout() {
    try {
      await request("/api/admin/logout", { method: "POST" });
    } catch (err) {
      /* 会话已失效也继续回登录页 */
    }
    state.session = null;
    location.replace("/admin/login");
  }

  function mountHeader() {
    if (!state.session) return;
    let bar = document.getElementById("admin-bar");
    if (!bar) {
      bar = document.createElement("div");
      bar.id = "admin-bar";
      bar.style.cssText =
        "position:fixed;right:16px;top:12px;z-index:20;display:flex;gap:10px;" +
        "align-items:center;background:#fff;border:1px solid #e5e8ee;border-radius:999px;" +
        "padding:6px 12px;font:12px system-ui;color:#4a5060;" +
        "box-shadow:0 4px 16px rgba(31,45,80,.08)";
      document.body.appendChild(bar);
    }
    bar.innerHTML = "";
    const who = document.createElement("span");
    who.textContent = state.session.username + " · " + state.session.role;
    const out = document.createElement("a");
    out.href = "#";
    out.textContent = "退出";
    out.style.cssText = "color:#315ed6;text-decoration:none";
    out.addEventListener("click", (event) => {
      event.preventDefault();
      logout();
    });
    bar.append(who, out);
  }

  async function init() {
    try {
      await getSession();
      mountHeader();
      return state.session;
    } catch (err) {
      return null;
    }
  }

  function canWrite() {
    return !!state.session && ["operator", "admin"].includes(state.session.role);
  }

  window.AdminAuth = {
    init,
    getSession,
    request,
    requestJSON,
    logout,
    mountHeader,
    canWrite,
    ForbiddenError,
  };
})();
