"""ch12 页面行为测试:用 Node 真实执行 static/index.html 的内联脚本并点击按钮,
断言实际发出的请求体包含 confirm_key,且重试同键/新提交换键。

不依赖 jsdom:测试自带最小 DOM 垫片(tests/page/ticket_confirm.cjs)。
Node 不可用时 skip(评测/验收脚本会另查一次)。
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")
HARNESS = ROOT / "tests" / "page" / "ticket_confirm.cjs"


@pytest.mark.skipif(NODE is None, reason="需要 node 运行页面行为测试")
def test_ticket_requests_carry_confirm_key():
    result = subprocess.run([NODE, str(HARNESS)], cwd=ROOT,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True, payload
    results = payload["results"]
    # 预览确认原样回传预览帧的键;失败重试沿用同键
    assert results["preview_echoes_key"] == "preview-key-abc12345"
    assert results["preview_retry_same_key"] == "preview-key-abc12345"
    # 普通建单:首次键合法;新的独立提交换新键
    assert results["normal_submit_key"] != results["normal_new_submission_key"]
    assert len(results["normal_submit_key"]) >= 8
    # 退款:内容变更、新表单都换新键(内容不同不能复用同键,否则后端 409)
    keys = {results["refund_submit_key"], results["refund_changed_key"],
            results["refund_new_form_key"]}
    assert len(keys) == 3
