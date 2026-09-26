"""Assemble <team>_submission.zip in the structure required by the challenge.

<team>_submission.zip
|-- output/matching_results.tsv, output/candidate_pairs.tsv
|-- code/business_entity_resolution/{src/, README.md, requirements.txt}
`-- Documentation_template.md

usage: python package_submission.py TEAM_NAME [SUBMISSION_VERSION_DIR]
"""
import shutil, sys, zipfile
from pathlib import Path

ROOT = Path(r"D:\Amazon ML Challenge")
team = sys.argv[1] if len(sys.argv) > 1 else "team"
src_out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "output"
zpath = ROOT / f"{team}_submission.zip"
code = ROOT / "code" / "business_entity_resolution"
with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        z.write(src_out / f, f"output/{f}")
    for f in sorted((code / "src").glob("*.py")):
        z.write(f, f"code/business_entity_resolution/src/{f.name}")
    for f in ("README.md", "requirements.txt"):
        z.write(code / f, f"code/business_entity_resolution/{f}")
    z.write(ROOT / "submission_package" / "Documentation_template.md", "Documentation_template.md")
print(zpath, round(zpath.stat().st_size / 1e6, 1), "MB")
with zipfile.ZipFile(zpath) as z:
    for n in z.namelist():
        print("  ", n)
