"""Pull fixed OCI images to SIF files. Run on a node with internet access."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.container_runtime import IMAGES, docker_image, image_path, runtime

def main():
    name = runtime()
    if name == "docker":
        raise SystemExit("Set CONTAINER_RUNTIME=singularity (or apptainer) before preparing SIF images.")
    for tool in IMAGES:
        target = image_path(tool)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            source = "docker://" + docker_image(tool)
            metadata = target.with_suffix(".json")
            if target.exists():
                if not metadata.is_file() or json.loads(metadata.read_text())["source"] != source:
                    raise SystemExit(f"Existing {target} has no matching source record. Choose a different *_SIF path.")
                print(f"Keeping {target}", flush=True)
                continue
            subprocess.run([name, "pull", str(target), source], check=True)
            digest = hashlib.sha256()
            with target.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            metadata.write_text(json.dumps({"source": source, "sha256": digest.hexdigest()}, indent=2) + "\n")
            print(f"Prepared {target}", flush=True)

if __name__ == "__main__":
    main()
