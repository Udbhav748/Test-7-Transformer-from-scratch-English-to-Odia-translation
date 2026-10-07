"""Verifies notebooks/kaggle/en_or_transformer.ipynb's embedded _FILES_B64
mapping is byte-for-byte identical to the current repo source files.
"""
import ast
import base64
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = REPO_ROOT / "notebooks" / "kaggle" / "en_or_transformer.ipynb"


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    src = None
    for c in nb["cells"]:
        if c["cell_type"] == "code" and "_FILES_B64" in "".join(c["source"]):
            src = "".join(c["source"])
            break
    assert src is not None, "could not find a cell containing _FILES_B64"

    tree = ast.parse(src)
    files_b64 = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "_FILES_B64" for t in node.targets
        ):
            files_b64 = ast.literal_eval(node.value)
            break
    assert files_b64 is not None, "could not parse _FILES_B64 literal"

    checked = 0
    mismatches = []
    for path, b64 in files_b64.items():
        decoded = base64.b64decode(b64).decode("utf-8")
        real = (REPO_ROOT / path).read_text(encoding="utf-8")
        checked += 1
        if decoded != real:
            mismatches.append(path)

    print(f"Embedded files checked: {checked}")
    print(f"Exact matches: {checked - len(mismatches)}")
    print(f"Mismatches remaining: {len(mismatches)}")
    if mismatches:
        print("MISMATCHED:", mismatches)

    clean_decoded = base64.b64decode(files_b64["src/data/clean.py"]).decode("utf-8")
    real_clean = (REPO_ROOT / "src/data/clean.py").read_text(encoding="utf-8")
    fix_line = next(
        line for line in real_clean.splitlines() if line.startswith("_EDGE_JOINER_RUN =")
    )
    has_fix = fix_line in clean_decoded
    print("clean.py embedded version has the \\s* edge-joiner fix:", has_fix)

    if mismatches or not has_fix:
        print("RESULT: MISMATCHES FOUND")
        sys.exit(1)
    print("RESULT: ALL EMBEDDED FILES MATCH CURRENT REPOSITORY")


if __name__ == "__main__":
    main()
