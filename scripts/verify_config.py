"""
Quick verification script — run before starting data collection.
Checks that all configs and class_labels are consistent after the
alphabet-only update.

Usage:
    python scripts/verify_config.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.class_labels import (
    CLASS_LABELS, NUM_CLASSES,
    SINGLE_HAND_SET, TWO_HAND_SET,
    get_hand_config_summary,
)
from src.utils.config_loader import load_config

print("\n" + "=" * 60)
print("  SignBridge — Configuration Verification")
print("=" * 60)

# ── class_labels.py ──────────────────────────────────────────
print(f"\n[class_labels.py]")
print(f"  NUM_CLASSES     : {NUM_CLASSES}")
print(f"  CLASS_LABELS    : {CLASS_LABELS}")
print(f"\n  {get_hand_config_summary()}")

# ── dataset.yaml ─────────────────────────────────────────────
cfg_d = load_config("dataset", resolve_paths=False)
print(f"\n[dataset.yaml]")
print(f"  num_classes     : {cfg_d['num_classes']}")
yaml_alpha = cfg_d["classes"]["alphabets"]
print(f"  alphabets count : {len(yaml_alpha)}")
print(f"  alphabets       : {yaml_alpha}")
yaml_1h = cfg_d["hand_detection"]["single_hand_classes"]
yaml_2h = cfg_d["hand_detection"]["two_hand_classes"]
print(f"  single_hand     : {yaml_1h}")
print(f"  two_hand        : {yaml_2h}")

# ── model.yaml ───────────────────────────────────────────────
cfg_m = load_config("model", resolve_paths=False)
output_dim = cfg_m["classifier"]["output_dim"]
print(f"\n[model.yaml]")
print(f"  output_dim      : {output_dim}")

# ── Cross-checks ─────────────────────────────────────────────
print("\n[Cross-checks]")
errors = []

if NUM_CLASSES != 26:
    errors.append(f"  FAIL — NUM_CLASSES is {NUM_CLASSES}, expected 26")
else:
    print(f"  PASS — NUM_CLASSES == 26")

if cfg_d["num_classes"] != 26:
    errors.append(f"  FAIL — dataset.yaml num_classes is {cfg_d['num_classes']}, expected 26")
else:
    print(f"  PASS — dataset.yaml num_classes == 26")

if output_dim != 26:
    errors.append(f"  FAIL — model.yaml output_dim is {output_dim}, expected 26")
else:
    print(f"  PASS — model.yaml output_dim == 26")

yaml_1h_set = set(yaml_1h)
yaml_2h_set = set(yaml_2h)
if yaml_1h_set != SINGLE_HAND_SET:
    errors.append(f"  FAIL — single_hand mismatch\n"
                  f"         yaml={sorted(yaml_1h_set)}\n"
                  f"         code={sorted(SINGLE_HAND_SET)}")
else:
    print(f"  PASS — single_hand_classes match: {sorted(SINGLE_HAND_SET)}")

if yaml_2h_set != TWO_HAND_SET:
    errors.append(f"  FAIL — two_hand mismatch\n"
                  f"         yaml={sorted(yaml_2h_set)}\n"
                  f"         code={sorted(TWO_HAND_SET)}")
else:
    print(f"  PASS — two_hand_classes match ({len(TWO_HAND_SET)} letters)")

if set(yaml_alpha) != set(CLASS_LABELS):
    errors.append(f"  FAIL — alphabet list mismatch between yaml and class_labels.py")
else:
    print(f"  PASS — alphabet list consistent across yaml and class_labels.py")

# ── Result ───────────────────────────────────────────────────
print()
if errors:
    print("=" * 60)
    print("  CONFIGURATION ERRORS FOUND:")
    for e in errors:
        print(e)
    print("=" * 60)
    sys.exit(1)
else:
    print("=" * 60)
    print("  ALL CHECKS PASSED — ready to collect data")
    print("=" * 60)
    print()
