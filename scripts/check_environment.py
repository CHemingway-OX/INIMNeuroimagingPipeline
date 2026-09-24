"""Check Python imports and command-line entry points without processing data."""
import importlib
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Use Python 3.11 through pixi run check")
    packages = {
        "numpy": "numpy", "scipy": "scipy", "pandas": "pandas",
        "nibabel": "nibabel", "matplotlib": "matplotlib",
        "skimage": "scikit-image", "ants": "antspyx",
    }
    for module, distribution in packages.items():
        importlib.import_module(module)
        print(f"{distribution}: {version(distribution)}", flush=True)
    for script in (
        "run_scripts/run_fastsurfer_docker.py",
        "run_scripts/run_lst_docker.py",
        "run_scripts/run_lesioninpainting.py",
        "Pipeline/generate_structural_qc.py",
        "Pipeline/collect_structural_metrics.py",
    ):
        subprocess.run([sys.executable, str(ROOT / script), "--help"],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["bash", "-n", str(ROOT / "Pipeline/struct_bids.sh")], check=True)
    subprocess.run(["bash", str(ROOT / "Pipeline/struct_bids.sh"), "--help"],
                   check=True, stdout=subprocess.DEVNULL)
    print("Python imports and CLI checks passed. Containers, FreeSurfer and imaging outputs were not tested.")

if __name__ == "__main__":
    main()
