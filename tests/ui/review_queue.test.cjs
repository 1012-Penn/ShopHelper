const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { test } = require('node:test');

class Element {
  constructor(tag = 'div') { this.tagName = tag; this.children = []; this.className = ''; this.disabled = false; this.value = ''; this._text = ''; this._html = ''; this.parts = {}; this.classList = { add() {}, remove() {} }; }
  set innerHTML(value) { this._html = value; this._text = ''; this.children = []; }
  get innerHTML() { return this._html; }
  set textContent(value) { this._text = value; this.children = []; }
  get textContent() { return this._text + this.children.map(x => x.textContent).join(''); }
  appendChild(child) { this.children.push(child); return child; }
  setAttribute(name, value) { this[name] = value; }
  append(...children) { children.forEach(x => this.appendChild(x)); }
  querySelector(selector) { return this.parts[selector] ||= new Element(); }
}
function button(root, text) { return root.children.flatMap(x => [x, ...descendants(x)]).find(x => x.tagName === 'button' && x.textContent.includes(text)); }
function descendants(root) { return root.children.flatMap(x => [x, ...descendants(x)]); }
class ForbiddenError extends Error {
  constructor(detail) { super(detail || '权限不足'); this.name = 'ForbiddenError'; this.status = 403; }
}
async function page(outcome, status = '待审', writable = true) {
  const elements = Object.fromEntries(['items', 'detail', 'title', 'review-status'].map(id => [id, new Element()]));
  elements['review-status'].value = status;
  const row = { id: 1, normalized_question: '运费问题', ai_suggested_answer: '保守答案', original_questions: [], review_status: status, approved_answer: status === '通过' ? '已核准答案' : '', knowledge_chunk_id: status === '通过' ? 7 : null, index_status: status === '通过' ? 'pending' : null };
  const calls = [];
  let initCalls = 0;
  const context = {
    document: { getElementById(id) { return elements[id] ||= new Element(); }, createElement(tag) { const el = new Element(tag); Object.defineProperty(el, 'id', {set(id) { elements[id] = el; }}); return el; }, querySelectorAll() { return []; } },
    console,
    async fetch(url, options = {}) {
      calls.push({ url, options });
      if (options.method === 'POST') return { ok: outcome.code < 400, status: outcome.code, async json() { return outcome.body; } };
      return { ok: true, status: 200, async json() { return url.includes('/1') ? row : { items: [row] }; } };
    },
  };
  context.AdminAuth = {
    ForbiddenError,
    async init() { initCalls += 1; return { username: 'op', role: writable ? 'operator' : 'viewer', csrf_token: 'csrf-test' }; },
    canWrite() { return writable; },
    async requestJSON(url, options = {}) {
      const headers = { ...(options.headers || {}), 'X-CSRF-Token': 'csrf-test' };
      const response = await context.fetch(url, { ...options, headers });
      const body = await response.json();
      if (!response.ok) {
        const message = typeof body.detail === 'string' ? body.detail : `请求失败（HTTP ${response.status}）`;
        throw response.status === 403 ? new ForbiddenError(message) : new Error(message);
      }
      return body;
    },
  };
  vm.createContext(context);
  const html = fs.readFileSync('static/review-queue.html', 'utf8');
  const script = html.match(/<script>([\s\S]*?)<\/script>/)[1].replace(/initPage\(\);\s*$/, 'globalThis.ready = initPage();');
  vm.runInContext(script, context);
  await context.ready;
  await context.openDetail(1, elements.items.children[0]);
  return { elements, calls, context, get initCalls() { return initCalls; } };
}

test('pending approval is never shown as indexed success', async () => {
  const { elements, calls } = await page({ code: 200, body: { approved: true, knowledge_chunk_id: 7, index_status: 'pending' } });
  await button(elements.detail, '通过').onclick();
  assert.match(elements.detail.textContent, /待重试/);
  assert.doesNotMatch(elements.detail.textContent, /已通过并写入知识库/);
  assert.ok(button(elements.detail, '重试'));
  const write = calls.find(x => x.options.method === 'POST');
  assert.equal(write.options.headers['X-CSRF-Token'], 'csrf-test');
});

test('failed approval preserves answer and re-enables the button', async () => {
  const { elements, context, calls } = await page({ code: 409, body: { detail: '答案已核准,请刷新' } });
  const answer = context.document.getElementById('answer');
  answer.value = '人工正在输入的答案';
  const approve = button(elements.detail, '通过');
  await approve.onclick();
  assert.equal(approve.disabled, false);
  assert.equal(JSON.parse(calls.find(x => x.options.method === 'POST').options.body).approved_answer, '人工正在输入的答案');
  assert.equal(answer.value, '人工正在输入的答案');
  assert.match(elements.detail.textContent, /答案已核准/);
  assert.doesNotMatch(elements.detail.textContent, /已通过并写入知识库/);
});

test('done approval displays indexed success', async () => {
  const { elements } = await page({ code: 200, body: { approved: true, knowledge_chunk_id: 7, index_status: 'done' } });
  await button(elements.detail, '通过').onclick();
  assert.match(elements.detail.textContent, /已通过并写入知识库/);
});

test('saved pending reviews remain discoverable and retryable after reload', async () => {
  const { elements, calls } = await page({ code: 200, body: { approved: true, knowledge_chunk_id: 7, index_status: 'done' } }, '通过');
  assert.ok(calls.some(x => x.url.includes('status=') && decodeURIComponent(x.url).includes('通过')));
  const retry = button(elements.detail, '重试');
  assert.ok(retry);
  await retry.onclick();
  assert.ok(calls.some(x => x.url.endsWith('/1/retry-index') && x.options.method === 'POST'));
  assert.match(elements.detail.textContent, /已通过并写入知识库/);
});

test('an old form cannot apply its approval to a newly selected review', async () => {
  const { elements, context } = await page({ code: 200, body: {} });
  const oldApprove = button(elements.detail, '通过');
  let finishDetail, finishApproval;
  context.fetch = (url, options = {}) => new Promise(resolve => {
    if (options.method === 'POST') finishApproval = resolve;
    else if (url.endsWith('/2')) finishDetail = resolve;
    else resolve({ ok: true, async json() { return { items: [] }; } });
  });
  const switching = context.openDetail(2, new Element('button'));
  const oldPosting = oldApprove.onclick();
  finishDetail({ ok: true, async json() { return { id: 2, normalized_question: '第二个问题', ai_suggested_answer: '第二个答案', review_status: '待审', original_questions: [] }; } });
  await switching;
  if (finishApproval) finishApproval({ ok: true, async json() { return { approved: true, knowledge_chunk_id: 7, index_status: 'done' }; } });
  await oldPosting;
  assert.equal(elements.title.textContent, '第二个问题');
  assert.equal(context.document.getElementById('answer').value, '第二个答案');
  assert.doesNotMatch(elements.detail.textContent, /已通过并写入知识库/);
});

test('session initializes once and read-only roles cannot use review write actions', async () => {
  const { elements, initCalls } = await page({ code: 200, body: {} }, '待审', false);
  assert.equal(initCalls, 1);
  assert.ok(button(elements.detail, '通过').disabled);
  assert.ok(button(elements.detail, '驳回').disabled);
  assert.match(elements.detail.textContent, /只读/);
});
