"""Run every experiment.

    python experiments/run_all.py --compute --jobs 16   simulations, in parallel processes
    python experiments/run_all.py --plot                 figures and numbers from results/raw
    python experiments/run_all.py                        both, one process

The compute step takes about half an hour of CPU time in total (eight minutes on
16 cores); the plot step about fifteen seconds.
The GIF animations are separate: python experiments/animations.py
"""
import importlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent

EXPERIMENTS = [
    "e01_estimators", "e02_scaling", "e03_var_backtests", "e04_es_backtest", "e05_pla",
    "e06_estimation_risk", "e07_model_uncertainty", "e08_asrf", "e09_copula_is",
    "e10_calibration_tests", "e11_discrimination", "e12_exposure", "e13_cva_wwr",
    "e14_nested", "e15_variance_reduction", "e16_lda", "e17_evt", "e18_reverse_stress",
    "e19_regimes", "gallery",
]


def _compute_one(name):
    t0 = time.time()
    res = subprocess.run([sys.executable, str(HERE / f"{name}.py"), "--compute"],
                         capture_output=True, text=True)
    status = "ok" if res.returncode == 0 else "FAILED\n" + res.stderr[-2000:]
    return f"{name:26s} {time.time() - t0:8.1f} s  {status}"


def main(argv):
    only = [a for a in argv if not a.startswith("--") and not a.isdigit()]
    names = [n for n in EXPERIMENTS if not only or any(o in n for o in only)]
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 1
    if "--compute" in argv:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for line in pool.map(_compute_one, names):
                print(line, flush=True)
    if "--plot" in argv or ("--compute" not in argv):
        sys.path.insert(0, str(HERE))
        for name in names:
            mod = importlib.import_module(name)
            if "--compute" not in argv and "--plot" not in argv:
                mod.compute()
            t0 = time.time()
            mod.plot()
            print(f"{name:26s} plot {time.time() - t0:6.1f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
