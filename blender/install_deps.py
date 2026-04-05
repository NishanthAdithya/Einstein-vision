"""Run this once inside Blender to install Python deps into Blender's Python."""
import subprocess
import sys

packages = ["pyyaml", "msgpack"]
target = sys.exec_prefix + "/Lib/site-packages"

for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--target", target])
    print(f"Installed {pkg} to {target}")
