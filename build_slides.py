"""Build the main deck and the nine topic decks, and copy the PDFs to their shared names.

    python build_slides.py            build everything
    python build_slides.py t5         build one topic deck (any part of its file name)
    python build_slides.py --keep     keep LaTeX intermediate files (default: clean them up)

Needs a LaTeX installation with latexmk and the Metropolis Beamer theme. The
slides read figures from figures/ and numbers from results/, so run
experiments/run_all.py first if those folders are empty.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLIDES = ROOT / "slides"
TOPICS = SLIDES / "topics"
DECKS = SLIDES / "decks"

# topic file stem -> published PDF name
TOPIC_DECKS = {
    "t1_market_risk": "01-var-es-market-risk.pdf",
    "t2_backtesting": "02-backtesting-var-es.pdf",
    "t3_model_risk": "03-model-risk-estimation-risk.pdf",
    "t4_credit_portfolio": "04-credit-portfolio-risk.pdf",
    "t5_pd_validation": "05-pd-model-validation.pdf",
    "t6_counterparty": "06-counterparty-risk-cva.pdf",
    "t7_monte_carlo": "07-monte-carlo-methods.pdf",
    "t8_operational_risk": "08-operational-risk-evt.pdf",
    "t9_stress_testing": "09-stress-testing.pdf",
}


def latexmk(folder: Path, stem: str) -> Path:
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"{stem}.tex"]
    res = subprocess.run(cmd, cwd=folder, capture_output=True, text=True)
    if res.returncode != 0:
        log = folder / f"{stem}.log"
        tail = log.read_text(errors="replace").splitlines()[-30:] if log.exists() else []
        raise SystemExit(f"LaTeX failed for {stem}:\n" + "\n".join(tail))
    return folder / f"{stem}.pdf"


def publish(folder: Path, stem: str, target: Path, keep: bool):
    """Copy the compiled PDF to its shared name; unless keep, remove build products."""
    pdf = latexmk(folder, stem)
    shutil.copyfile(pdf, target)
    if not keep:
        subprocess.run(["latexmk", "-C", f"{stem}.tex"], cwd=folder, capture_output=True)
        for ext in ("bbl", "nav", "snm", "vrb", "run.xml", "bcf"):
            (folder / f"{stem}.{ext}").unlink(missing_ok=True)


def main(argv):
    keep = "--keep" in argv
    args = [a for a in argv[1:] if a != "--keep"]
    only = args[0] if args else None
    DECKS.mkdir(exist_ok=True)
    if only is None or only == "main":
        publish(SLIDES, "main", SLIDES / "stochastic-risk-slides.pdf", keep)
        print("main deck   -> slides/stochastic-risk-slides.pdf")
    for stem, name in TOPIC_DECKS.items():
        if only is not None and only not in stem:
            continue
        publish(TOPICS, stem, DECKS / name, keep)
        print(f"{stem:22s} -> slides/decks/{name}")


if __name__ == "__main__":
    main(sys.argv)
