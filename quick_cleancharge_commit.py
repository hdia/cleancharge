# quick_cleancharge_commit.py

import subprocess
from datetime import datetime
from pathlib import Path

REPO_DIR = Path(r"C:\CleanCharge\cleancharge_GitHub_final")

def run(cmd):
    print(f">> {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=REPO_DIR,
        text=True,
        capture_output=True
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode

def main():
    msg = f"Update CleanCharge repository {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    run("git status")
    run("git add .")

    commit_code = run(f'git commit -m "{msg}"')
    if commit_code != 0:
        print("No commit created, possibly nothing to commit.")

    push_code = run("git push origin main")
    if push_code != 0:
        print("\nPush failed. Run: git pull origin main, resolve any conflict, then push again.")
    else:
        print("\nDone. Changes synced to GitHub.")

if __name__ == "__main__":
    main()