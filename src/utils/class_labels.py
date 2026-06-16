"""
class_labels.py
---------------
Canonical ISL class label registry.

Scope: 26 ISL alphabet letters only (A-Z).
Words list has been intentionally removed per project specification.

Hand configuration verified against the ISL alphabet reference image:
  Single hand (1H) : C, I, L, O, U, V
  Double hand (2H) : A, B, D, E, F, G, H, J, K, M, N, P, Q, R, S, T, W, X, Y, Z

Provides
--------
* CLASS_LABELS    — ordered list of 26 ISL class name strings (A–Z)
* CLASS_TO_IDX    — dict mapping class name  → integer index (0–25)
* IDX_TO_CLASS    — dict mapping integer index → class name
* SINGLE_HAND_SET — set of classes requiring exactly 1 hand
* TWO_HAND_SET    — set of classes requiring 2 hands
* NUM_CLASSES     — integer 26
"""

# ---------------------------------------------------------------------------
# Ordered class list  (index = position in this list)
# ---------------------------------------------------------------------------
CLASS_LABELS: list[str] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
]

NUM_CLASSES: int = len(CLASS_LABELS)   # 26

# ---------------------------------------------------------------------------
# Bidirectional lookup tables
# ---------------------------------------------------------------------------
CLASS_TO_IDX: dict[str, int] = {cls: idx for idx, cls in enumerate(CLASS_LABELS)}
IDX_TO_CLASS: dict[int, str] = {idx: cls for idx, cls in enumerate(CLASS_LABELS)}

# ---------------------------------------------------------------------------
# Hand configuration
# Verified against ISL alphabet reference image (provided by user)
#
#  Single hand (1H): C  I  L  O  U  V
#  Double hand (2H): A  B  D  E  F  G  H  J  K  M  N  P  Q  R  S  T  W  X  Y  Z
# ---------------------------------------------------------------------------
SINGLE_HAND_SET: set[str] = {"C", "I", "L", "O", "U", "V"}

TWO_HAND_SET: set[str] = {
    "A", "B", "D", "E", "F", "G", "H", "J", "K",
    "M", "N", "P", "Q", "R", "S", "T", "W", "X", "Y", "Z",
}

# ---------------------------------------------------------------------------
# Sanity checks at import time
# ---------------------------------------------------------------------------
assert SINGLE_HAND_SET | TWO_HAND_SET == set(CLASS_LABELS), (
    "SINGLE_HAND_SET + TWO_HAND_SET must cover all 26 classes exactly.\n"
    f"  Missing : {set(CLASS_LABELS) - (SINGLE_HAND_SET | TWO_HAND_SET)}\n"
    f"  Extra   : {(SINGLE_HAND_SET | TWO_HAND_SET) - set(CLASS_LABELS)}"
)
assert SINGLE_HAND_SET & TWO_HAND_SET == set(), (
    "A class cannot appear in both SINGLE_HAND_SET and TWO_HAND_SET."
)
assert NUM_CLASSES == 26, f"Expected 26 classes, got {NUM_CLASSES}."


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def label_to_index(label: str) -> int:
    """Convert a class name string to its integer index (0–25)."""
    try:
        return CLASS_TO_IDX[label]
    except KeyError:
        raise ValueError(
            f"Unknown class label '{label}'. "
            f"Valid labels: {CLASS_LABELS}"
        )


def index_to_label(index: int) -> str:
    """Convert an integer index (0–25) to its class name string."""
    try:
        return IDX_TO_CLASS[index]
    except KeyError:
        raise ValueError(
            f"Index {index} out of range. Valid range: 0–{NUM_CLASSES - 1}."
        )


def requires_two_hands(label: str) -> bool:
    """Return True if the given ISL letter requires both hands."""
    return label in TWO_HAND_SET


def requires_one_hand(label: str) -> bool:
    """Return True if the given ISL letter requires exactly one hand."""
    return label in SINGLE_HAND_SET


def get_single_hand_labels() -> list[str]:
    """Return the list of single-hand ISL letters: C, I, L, O, U, V."""
    return sorted(SINGLE_HAND_SET, key=lambda x: CLASS_TO_IDX[x])


def get_two_hand_labels() -> list[str]:
    """Return the list of two-hand ISL letters."""
    return sorted(TWO_HAND_SET, key=lambda x: CLASS_TO_IDX[x])


def display_label(label: str) -> str:
    """Return the label as-is (all alphabet labels are already clean)."""
    return label


def get_hand_config_summary() -> str:
    """Return a formatted summary of the hand configuration."""
    single = " ".join(get_single_hand_labels())
    two    = " ".join(get_two_hand_labels())
    return (
        f"ISL Alphabet — 26 classes\n"
        f"  Single hand (1H) [{len(SINGLE_HAND_SET):2d}]: {single}\n"
        f"  Double hand (2H) [{len(TWO_HAND_SET):2d}]: {two}"
    )
