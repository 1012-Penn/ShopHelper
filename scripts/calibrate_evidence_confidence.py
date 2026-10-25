"""已废弃：请改用 scripts/gen_evidence_signals.py --calibrate。

旧脚本只接受 ch04 正样本，没有同域硬负例、没有校准/留出切分，会产生
“最弱正样本下界”的循环校准产物。v2 校准统一走：

  .venv/bin/python scripts/gen_evidence_signals.py --calibrate \
      --artifact-config config/evidence_calibration.json
"""
import sys


def main() -> int:
    print("scripts/calibrate_evidence_confidence.py 已废弃；"
          "请改用 scripts/gen_evidence_signals.py --calibrate")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
