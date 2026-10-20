"""不依赖浏览器下载的审核页面行为回归;执行实际内联脚本。"""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_review_page_http_outcomes_and_retry():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the DOM contract test")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([node, "--test", "tests/ui/review_queue.test.cjs"],
                            cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
