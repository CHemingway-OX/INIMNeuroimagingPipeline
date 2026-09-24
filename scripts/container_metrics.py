"""Run the metrics collector's asegstats2table invocation inside FastSurfer."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.container_runtime import run_container

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--tablefile", required=True)
    args, extra = parser.parse_known_args()
    stats = Path(args.inputs).resolve()
    table = Path(args.tablefile).resolve()
    bootstrap = (
        'if ! command -v asegstats2table >/dev/null 2>&1; then '
        'for setup in "${FREESURFER_HOME:-/nonexistent}/SetUpFreeSurfer.sh" '
        '/usr/local/freesurfer/SetUpFreeSurfer.sh /opt/freesurfer/SetUpFreeSurfer.sh '
        '/freesurfer/SetUpFreeSurfer.sh; do '
        'if [ -f "$setup" ]; then source "$setup"; break; fi; done; fi; '
        'exec asegstats2table "$@"'
    )
    run_container("fastsurfer", ["/bin/bash", "-c", bootstrap, "_", "--inputs",
                  f"/stats/{stats.name}", "--tablefile", f"/table/{table.name}", *extra],
                  [f"{stats.parent}:/stats:ro", f"{table.parent}:/table"])

if __name__ == "__main__":
    main()
