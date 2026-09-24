"""Foreground Docker/Singularity invocations shared by all pipeline stages."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
# Registry digests verified on 2026-09-24. Keep images fixed across subject jobs.
IMAGES = {
    "fastsurfer": ("fastsurfer-2.4.2.sif", "deepmi/fastsurfer@sha256:d4f87043faa13ae481d49d4417bf28ac8c9957b17949921189bd1b51ff0fd743"),
    "lst": ("lst-ai-1.2.0.sif", "jqmcginnis/lst-ai@sha256:2b2a0d6454e6efff60a350df3c650ae9584f1926338dec7842f9e71912cc6c85"),
    "lit": ("lit-0.5.0.sif", "deepmi/lit@sha256:7de676ab1a2141bc2fab04719c2a1bcbbedb0e9c2b0cf9b1b84f19a9c2fa746a"),
}

def runtime():
    name = os.environ.get("CONTAINER_RUNTIME", "docker")
    if name not in {"docker", "singularity", "apptainer"}:
        raise ValueError(f"Unsupported CONTAINER_RUNTIME: {name}")
    return name

def image_path(tool):
    default = Path(os.environ.get("CONTAINER_DIR", str(ROOT / "containers"))) / IMAGES[tool][0]
    return Path(os.environ.get(f"{tool.upper()}_SIF", str(default))).expanduser().resolve()

def docker_image(tool):
    return os.environ.get(f"{tool.upper()}_DOCKER_IMAGE", IMAGES[tool][1])

def build_command(tool, args, binds=(), gpu=False, workdir=None):
    name = runtime()
    env = os.environ.copy()
    # Do not inherit host Python paths or stale container-specific overrides.
    for key in list(env):
        if key.startswith(("SINGULARITYENV_", "APPTAINERENV_")):
            env.pop(key)
    if name == "docker":
        cmd = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}"]
        if gpu:
            devices = env.get("CUDA_VISIBLE_DEVICES")
            if devices is not None and not devices.strip():
                raise ValueError("CUDA_VISIBLE_DEVICES is empty for a GPU container")
            cmd += ["--gpus", f'"device={devices}"' if devices is not None else "all"]
        for bind in binds:
            cmd += ["--volume", bind]
        if workdir:
            cmd += ["--workdir", workdir]
        for key in ("OMP_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
            if key in env:
                cmd += ["--env", f"{key}={env[key]}"]
        if not args:
            raise ValueError("Container command is required")
        cmd += ["--entrypoint", args[0], docker_image(tool), *map(str, args[1:])]
    else:
        image = image_path(tool)
        if not image.is_file():
            raise FileNotFoundError(f"Missing image: {image}. Run pixi run prepare-images first.")
        cmd = [name, "exec", "--cleanenv", "--no-mount", "home,cwd"]
        if gpu:
            cmd += ["--nv"]
        prefix = "SINGULARITYENV_" if name == "singularity" else "APPTAINERENV_"
        for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
            if key in env and (key != "CUDA_VISIBLE_DEVICES" or gpu):
                env[prefix + key] = env[key]
        if gpu and env.get("CUDA_VISIBLE_DEVICES") == "":
            raise ValueError("CUDA_VISIBLE_DEVICES is empty for a GPU container")
        for bind in binds:
            cmd += ["--bind", bind]
        cmd += ["--pwd", workdir or "/tmp", str(image), *map(str, args)]
    return cmd, env

def run_container(tool, args, binds=(), gpu=False, workdir=None):
    cmd, env = build_command(tool, args, binds, gpu, workdir)
    print("Command: " + shlex.join(cmd), flush=True)
    return subprocess.run(cmd, env=env, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, choices=IMAGES)
    parser.add_argument("--bind", action="append", default=[])
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--workdir")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command after -- is required")
    try:
        run_container(args.tool, command, args.bind, args.gpu, args.workdir)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)

if __name__ == "__main__":
    main()
