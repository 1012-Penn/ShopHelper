"""ch11 离线验收脚本:报告每项检查且不依赖 API key。"""
from __future__ import annotations

from scripts import acceptance_ch11


def test_acceptance_reports_each_check(capsys):
    code = acceptance_ch11.main(["--skip-tests", "--skip-snapshot"])
    output = capsys.readouterr().out
    assert code == 0
    for name in ("corpus", "frozen-v1", "regression-v1", "leakage", "budget", "offline-harness"):
        assert f"[ok] {name}" in output


def test_acceptance_offline_does_not_require_api_keys(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("离线验收不应构造 Settings")

    monkeypatch.setattr("app.config.Settings", _boom)
    assert acceptance_ch11.main(["--skip-tests", "--skip-snapshot"]) == 0
