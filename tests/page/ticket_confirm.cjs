// ch12 页面行为测试:在最小 DOM 垫片里真实执行 static/index.html 的内联脚本,
// 点击按钮后检查实际发出的 fetch 请求体。
// - 工单预览确认必须原样回传 evt.confirm_key;失败重试沿用同键
// - 普通建单 / 退款建单:一次提交稳定键,失败重试同键,新提交换新键
// 输出 JSON 结果;断言失败即非零退出。
"use strict";

const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assert = require("node:assert/strict");

const root = path.resolve(__dirname, "..", "..");
const html = fs.readFileSync(path.join(root, "static", "index.html"), "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
assert.ok(scripts.length >= 1, "index.html 应包含内联脚本");
const pageScript = scripts[scripts.length - 1];

// ---------- 最小 DOM ----------
class TextNode {
  constructor(text) {
    this.nodeType = 3;
    this.textContent = String(text);
    this.parent = null;
  }
  remove() {
    if (this.parent) this.parent._removeChild(this);
  }
}

const SIMPLE_SELECTOR = /^([a-zA-Z0-9]*)(?:\.([\w-]+))?$/;

function matches(el, simple) {
  const m = SIMPLE_SELECTOR.exec(simple);
  if (!m) return false;
  const [, tag, cls] = m;
  if (tag && el.tagName.toLowerCase() !== tag.toLowerCase()) return false;
  if (cls && !el.classList.contains(cls)) return false;
  return true;
}

class Element {
  constructor(tagName) {
    this.nodeType = 1;
    this.tagName = String(tagName || "div").toUpperCase();
    this.children = [];
    this.parent = null;
    this._text = "";
    this._classes = new Set();
    this.disabled = false;
    this.value = "";
    this.onclick = null;
    this.dataset = {};
    this.scrollHeight = 0;
    this.scrollTop = 0;
    this.clientHeight = 0;
    this.listeners = {};
    const self = this;
    this.classList = {
      add(...names) { names.forEach((n) => self._classes.add(n)); },
      remove(...names) { names.forEach((n) => self._classes.delete(n)); },
      contains(name) { return self._classes.has(name); },
      toggle(name) { return self._classes.has(name) ? (self._classes.delete(name), false) : (self._classes.add(name), true); },
    };
  }
  get className() { return [...this._classes].join(" "); }
  set className(v) {
    this._classes = new Set(String(v || "").split(/\s+/).filter(Boolean));
  }
  get textContent() {
    if (this.children.length) return this.children.map((c) => c.textContent).join("");
    return this._text;
  }
  set textContent(v) {
    this.children.forEach((c) => { c.parent = null; });
    this.children = [];
    this._text = String(v == null ? "" : v);
  }
  get innerHTML() {
    return this.children.map((c) => (c.nodeType === 3 ? c.textContent : c.innerHTML)).join("");
  }
  set innerHTML(htmlText) {
    this.children.forEach((c) => { c.parent = null; });
    this.children = [];
    this._text = "";
    parseHTML(String(htmlText || ""), this);
  }
  appendChild(node) {
    if (node && node.nodeType === 3) {
      node.parent = this;
      this.children.push(node);
    } else if (node) {
      node.parent = this;
      this.children.push(node);
    }
    return node;
  }
  append(...nodes) { nodes.forEach((n) => this.appendChild(n)); }
  add(node) {  // <select>.add(option)
    this.appendChild(node);
    if (this.value === "") this.value = node.value;
    return node;
  }
  insertBefore(node, ref) {
    const idx = this.children.indexOf(ref);
    node.parent = this;
    this.children.splice(idx < 0 ? this.children.length : idx, 0, node);
    return node;
  }
  before(node) {
    if (!this.parent) return;
    this.parent.insertBefore(node, this);
  }
  _removeChild(node) {
    const idx = this.children.indexOf(node);
    if (idx >= 0) this.children.splice(idx, 1);
    node.parent = null;
  }
  remove() {
    if (this.parent) this.parent._removeChild(this);
  }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  focus() {}
  _descendants() {
    const out = [];
    const walk = (el) => {
      for (const child of el.children) {
        if (child.nodeType === 1) {
          out.push(child);
          walk(child);
        }
      }
    };
    walk(this);
    return out;
  }
  querySelectorAll(selector) {
    const parts = String(selector || "").split(",").map((s) => s.trim());
    const result = [];
    for (const part of parts) {
      const chain = part.replace(/:not\([^)]*\)/g, "").trim().split(/\s+/);
      const last = chain[chain.length - 1];
      for (const el of this._descendants()) {
        if (matches(el, last) && !result.includes(el)) result.push(el);
      }
    }
    return result;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}

function parseHTML(source, parent) {
  const tokenRe = /<(\/?)([a-zA-Z0-9]+)([^>]*)>|([^<]+)/g;
  const stack = [parent];
  let m;
  while ((m = tokenRe.exec(source)) !== null) {
    const [, closing, tag, attrs, text] = m;
    if (text !== undefined) {
      if (text) stack[stack.length - 1].appendChild(new TextNode(text));
      continue;
    }
    if (closing) {
      if (stack.length > 1) stack.pop();
      continue;
    }
    const el = new Element(tag);
    const cls = /class="([^"]*)"/.exec(attrs || "");
    if (cls) el.className = cls[1];
    stack[stack.length - 1].appendChild(el);
    if (!["input", "br", "img"].includes(tag.toLowerCase())) stack.push(el);
  }
}

// ---------- 沙箱 ----------
const ids = new Map();
const byId = (id) => {
  if (!ids.has(id)) ids.set(id, new Element("div"));
  return ids.get(id);
};
const requests = [];
let fetchMode = "ok";

const document = {
  getElementById: byId,
  createElement: (tag) => new Element(tag),
  createTextNode: (text) => new TextNode(text),
};

function response(ok, body, status) {
  return {
    ok,
    status: status || (ok ? 200 : 500),
    json: async () => body,
  };
}

async function fetchStub(url, options) {
  const request = { url, options, body: JSON.parse((options && options.body) || "{}") };
  requests.push(request);
  if (fetchMode === "fail") return response(false, { detail: "boom" }, 500);
  if (fetchMode === "conflict") return response(false, { detail: "conflict" }, 409);
  if (url === "/api/conversations") return response(true, { ok: true, items: [] });
  if (url.startsWith("/api/conversations/")) return response(true, { messages: [] });
  return response(true, { ticket_no: "T20260101099", status: "待处理" });
}

function Option(text, value) {
  const el = new Element("option");
  el.textContent = text;
  el.value = value === undefined ? text : value;
  return el;
}

const sandbox = {
  document,
  fetch: fetchStub,
  confirm: () => true,
  alert: () => {},
  setInterval: () => 0,
  clearInterval: () => {},
  setTimeout,
  console,
  Option,
  crypto: (() => {
    let n = 0;
    return {
      randomUUID: () => "12345678-90ab-cdef-1234-" + String(++n).padStart(12, "0"),
    };
  })(),
  window: {},
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;

// 脚本尾部的 test hook:把 const state / let lastUserMessage 暴露给测试
const epilogue = `
;globalThis.__t = {
  state,
  setUser(v) { lastUserMessage = v; },
  getUser() { return lastUserMessage; },
};`;

vm.createContext(sandbox);
vm.runInContext(pageScript + epilogue, sandbox, { filename: "index.html#script" });

const KEY_RE = /^[A-Za-z0-9_-]{8,64}$/;

function lastRequest(url) {
  return [...requests].reverse().find((r) => r.url === url);
}

async function main() {
  const results = {};

  // ---------- 1. 工单预览:原样回传 evt.confirm_key ----------
  sandbox.__t.state.sessionId = 42;
  const bubble = sandbox.document.createElement("div");
  bubble.innerHTML = '<div class="content"></div>';
  const evt = { type: "ticket_preview", confirm_key: "preview-key-abc12345",
                ticket_type: "售后", description: "商品坏了要退款",
                conversation_id: 42 };
  sandbox.renderTicketPreview(bubble, evt);
  let card = bubble.querySelector(".ticket-card");
  assert.ok(card, "应渲染工单确认卡");
  card.querySelector(".t-confirm").onclick();
  await new Promise((r) => setTimeout(r, 0));
  let req = lastRequest("/api/tickets/confirm");
  assert.ok(req, "点击确认应发起 confirm 请求");
  assert.equal(req.body.confirm_key, "preview-key-abc12345", "确认必须原样回传预览帧的 confirm_key");
  assert.equal(req.body.conversation_id, 42);
  assert.equal(req.body.description, "商品坏了要退款");
  results.preview_echoes_key = req.body.confirm_key;

  // 失败重试:沿用同一帧的键
  requests.length = 0;
  fetchMode = "fail";
  bubble.innerHTML = '<div class="content"></div>';
  sandbox.renderTicketPreview(bubble, evt);
  card = bubble.querySelector(".ticket-card");
  card.querySelector(".t-confirm").onclick();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(card.querySelector(".t-confirm").disabled, false, "失败后应可重试");
  card.querySelector(".t-confirm").onclick();
  await new Promise((r) => setTimeout(r, 0));
  const retryBodies = requests.filter((r) => r.url === "/api/tickets/confirm").map((r) => r.body);
  assert.equal(retryBodies.length, 2);
  assert.equal(retryBodies[0].confirm_key, "preview-key-abc12345");
  assert.equal(retryBodies[1].confirm_key, retryBodies[0].confirm_key, "失败重试必须沿用同键");
  results.preview_retry_same_key = retryBodies[1].confirm_key;

  // ---------- 2. 普通建单:一次提交稳定键,失败同键重试,新按钮新键 ----------
  fetchMode = "fail";
  requests.length = 0;
  sandbox.__t.setUser("帮我建个工单");
  const btn = sandbox.document.createElement("button");
  await sandbox.createTicket(btn);
  const key1 = btn.dataset.confirmKey;
  assert.match(key1, KEY_RE, "普通建单应带合法 confirm_key");
  assert.equal(requests[0].body.confirm_key, key1);
  assert.equal(requests[0].body.description, "帮我建个工单");
  await sandbox.createTicket(btn);  // 失败后重试
  assert.equal(requests[1].body.confirm_key, key1, "失败重试沿用同键");
  // 成功后再来一次独立提交(新的操作按钮)应换新键
  fetchMode = "ok";
  await sandbox.createTicket(btn);
  assert.equal(btn.dataset.confirmKey, "", "成功后该次提交的键作废");
  const btn2 = sandbox.document.createElement("button");
  await sandbox.createTicket(btn2);
  assert.notEqual(requests[3].body.confirm_key, key1, "新的独立提交使用新键");
  results.normal_submit_key = key1;
  results.normal_new_submission_key = requests[3].body.confirm_key;

  // ---------- 3. 退款建单:一次提交稳定键,失败同键,内容变更/新表单换新键 ----------
  fetchMode = "fail";
  requests.length = 0;
  const bar = sandbox.document.createElement("div");
  const bubble2 = sandbox.document.createElement("div");
  bubble2.innerHTML = '<div class="content"></div>';
  sandbox.renderRefundForm(bar, bubble2, "1001");
  let form = bubble2.querySelector(".refund-form");
  const submit = form.querySelector(".rf-submit");
  submit.onclick({ target: submit });
  await new Promise((r) => setTimeout(r, 0));
  const refundKey = requests[0].body.confirm_key;
  assert.match(refundKey, KEY_RE);
  assert.equal(requests[0].body.ticket_type, "售后");
  assert.ok(requests[0].body.description.includes("订单1001申请退款"));
  submit.onclick({ target: submit });  // 同内容重试
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(requests[1].body.confirm_key, refundKey, "退款失败重试沿用同键");
  // 改备注 = 新提交 → 新键(后端会因描述不同拒绝复用键)
  form.querySelector(".rf-note").value = "补充说明";
  submit.onclick({ target: submit });
  await new Promise((r) => setTimeout(r, 0));
  assert.notEqual(requests[2].body.confirm_key, refundKey, "内容变更视为新提交,换新键");
  // 新表单(新独立提交)即使内容相同也换新键
  const bar3 = sandbox.document.createElement("div");
  const bubble3 = sandbox.document.createElement("div");
  bubble3.innerHTML = '<div class="content"></div>';
  sandbox.renderRefundForm(bar3, bubble3, "1001");
  const submit3 = bubble3.querySelector(".refund-form").querySelector(".rf-submit");
  submit3.onclick({ target: submit3 });
  await new Promise((r) => setTimeout(r, 0));
  assert.notEqual(requests[3].body.confirm_key, refundKey, "新表单是新提交,换新键");
  results.refund_submit_key = refundKey;
  results.refund_changed_key = requests[2].body.confirm_key;
  results.refund_new_form_key = requests[3].body.confirm_key;

  console.log(JSON.stringify({ ok: true, results }));
}

main().catch((err) => {
  console.error(JSON.stringify({ ok: false, error: String(err && err.stack || err) }));
  process.exit(1);
});
