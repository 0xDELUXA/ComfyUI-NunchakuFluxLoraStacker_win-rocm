"""
ComfyUI-Manager install / update hook.

Installs this pack's requirements.txt into the active ComfyUI Python
environment (includes pytorch-lightning and other CCSR deps so the pack
does not fail to import when those packages are missing).
"""
import subprocess
import sys
from pathlib import Path


def main():
    req = Path(__file__).resolve().parent / "requirements.txt"
    print(
        "[ComfyUI-NunchakuFluxLoraStacker] install.py: "
        f"installing -r {req.name} ..."
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(req)],
    )
    print("[ComfyUI-NunchakuFluxLoraStacker] install.py: requirements ready.")


if __name__ == "__main__":
    main()
