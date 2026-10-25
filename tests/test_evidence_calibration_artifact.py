"""证据闸产物 v2 的加载、校验、回落与原子写测试。"""
import json
from types import SimpleNamespace

from app.graph.builder import _load_evidence_calibration
from scripts.gen_evidence_signals import _atomic_write


def _settings(tmp_path, rerank_floor=0.03):
    return SimpleNamespace(
        evidence_calibration_path=str(tmp_path / "cal.json"),
        rerank_score_floor=rerank_floor,
    )


def _v2(**thresholds):
    base = {
        "top1_floor": 0.10, "effective_score_floor": 0.03,
        "min_effective_count": 1, "margin_floor": 0.05,
        "combined_floor": 0.0, "combined_diagnostic_only": True,
    }
    base.update(thresholds)
    return {"schema_version": 2, "status": "ok", "source": "test", "thresholds": base}


def test_loads_v2_ok(tmp_path):
    settings = _settings(tmp_path)
    (tmp_path / "cal.json").write_text(json.dumps(_v2()), encoding="utf-8")
    calibration = _load_evidence_calibration(settings)
    assert calibration.top1_floor == 0.10
    assert calibration.margin_floor == 0.05
    assert calibration.min_effective_count == 1
    assert calibration.combined_floor == 0.0
    assert calibration.source == "test"


def test_combined_diagnostic_only_forces_zero(tmp_path):
    settings = _settings(tmp_path)
    payload = _v2(combined_floor=0.42, combined_diagnostic_only=True)
    (tmp_path / "cal.json").write_text(json.dumps(payload), encoding="utf-8")
    assert _load_evidence_calibration(settings).combined_floor == 0.0


def test_infeasible_status_falls_back(tmp_path):
    settings = _settings(tmp_path)
    payload = _v2()
    payload["status"] = "infeasible"
    (tmp_path / "cal.json").write_text(json.dumps(payload), encoding="utf-8")
    calibration = _load_evidence_calibration(settings)
    assert calibration.source == "fallback"
    assert calibration.min_effective_count == 1
    assert calibration.combined_floor == 0.15


def test_min_effective_count_zero_falls_back(tmp_path):
    settings = _settings(tmp_path)
    (tmp_path / "cal.json").write_text(
        json.dumps(_v2(min_effective_count=0)), encoding="utf-8")
    calibration = _load_evidence_calibration(settings)
    assert calibration.source == "fallback"
    assert calibration.min_effective_count == 1


def test_legacy_flat_format_loads(tmp_path):
    settings = _settings(tmp_path)
    payload = {"top1_floor": 0.2, "effective_score_floor": 0.03,
               "min_effective_count": 1, "margin_floor": 0.01,
               "combined_floor": 0.15, "source": "legacy"}
    (tmp_path / "cal.json").write_text(json.dumps(payload), encoding="utf-8")
    calibration = _load_evidence_calibration(settings)
    assert calibration.top1_floor == 0.2 and calibration.source == "legacy"


def test_invalid_json_falls_back(tmp_path):
    settings = _settings(tmp_path)
    (tmp_path / "cal.json").write_text("{not json", encoding="utf-8")
    assert _load_evidence_calibration(settings).source == "fallback"


def test_missing_file_falls_back(tmp_path):
    assert _load_evidence_calibration(_settings(tmp_path)).source == "fallback"


def test_atomic_write_leaves_no_tmp(tmp_path):
    path = tmp_path / "artifact.json"
    _atomic_write(path, {"a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}
    assert not (tmp_path / "artifact.json.tmp").exists()
