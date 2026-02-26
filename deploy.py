import os
import sys
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
DIST = ROOT / "dist"

def run(cmd):
    print("\n>>", cmd)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("❌ Command failed")
        sys.exit(result.returncode)

def clean():
    print("🧹 Cleaning old builds...")
    shutil.rmtree(ROOT / "build", ignore_errors=True)
    shutil.rmtree(ROOT / "dist", ignore_errors=True)
    for p in ROOT.glob("*.egg-info"):
        shutil.rmtree(p, ignore_errors=True)

def build():
    print("📦 Building package...")
    run("python -m build")

def upload_test():
    print("🚀 Uploading to TestPyPI...")
    run("twine upload -r testpypi dist/*")

def upload_prod():
    print("🚀 Uploading to PyPI...")
    run("twine upload dist/*")

def main():
    if len(sys.argv) < 2:
        print("Usage: python deploy.py [test | prod]")
        sys.exit(1)

    target = sys.argv[1].lower()

    clean()
    build()

    if target == "test":
        upload_test()
    elif target == "prod":
        upload_prod()
    else:
        print("Invalid target. Use 'test' or 'prod'")
        sys.exit(1)

    print("\n✅ Deploy complete!")

if __name__ == "__main__":
    main()