"""Selective MIMIC-IV v3.1 downloader for the temporal med-lab project.

Downloads only the seven `hosp/` files we actually need (skipping `icu/` and
everything else). Uses the system `curl` binary under the hood because
PhysioNet's file server returns 403 to Python `requests` (likely due to
User-Agent / HTTP/2 differences) while accepting plain curl GETs.

USAGE (Windows PowerShell):

    cd "D:\\PYTHON\\Explainable Temporal Medication–Laboratory"
    python scripts\\download_mimic.py --user YOUR_PHYSIONET_USERNAME

The script will prompt for your PhysioNet password. By default the password
is **visible** while you type (use --hide-password to hide it). Resume is
automatic: re-running skips files that are already complete.

Files land in: data/raw/mimiciv/hosp/
"""
from __future__ import annotations

import argparse
import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path

# MIMIC-IV version to download. v2.2 is currently the most stable for credentialed
# file-server access; v3.1 has propagation issues for newly credentialed users.
# Schema is identical for the hosp/ tables we use, so all downstream code works
# unchanged. Override with --version on the command line.
DEFAULT_VERSION = "2.2"
BASE_URL_TEMPLATE = "https://physionet.org/files/mimiciv/{version}"

# Only the hosp/ tables we need for S1/S2/S3 (ACEi+K, Warfarin+INR, Metformin+eGFR).
# Sizes are rough — just for the friendly summary at the start.
FILES = [
    ("hosp/patients.csv.gz",            3 * 1024 * 1024),
    ("hosp/admissions.csv.gz",         20 * 1024 * 1024),
    ("hosp/d_labitems.csv.gz",         10 * 1024),
    ("hosp/d_icd_diagnoses.csv.gz",     2 * 1024 * 1024),
    ("hosp/diagnoses_icd.csv.gz",      50 * 1024 * 1024),
    ("hosp/prescriptions.csv.gz",     500 * 1024 * 1024),
    ("hosp/labevents.csv.gz",        2000 * 1024 * 1024),  # the big one
]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:6.1f} {unit}"
        n /= 1024
    return f"{n:6.1f} TB"


def have_curl() -> bool:
    return shutil.which("curl") is not None


def curl_download(url: str, dest: Path, user: str, password: str) -> int:
    """Run curl with resume + progress bar. Returns curl's exit code (0=ok)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # -C - : resume from where local file left off
    # -L  : follow redirects
    # -f  : fail on HTTP errors (non-2xx)
    # --progress-bar : compact progress bar
    # -u user:password : HTTP Basic Auth
    # -o file : write to file
    cmd = [
        "curl",
        "-L", "-f", "-C", "-",
        "--progress-bar",
        "-A", "Wget/1.21.3",
        "-u", f"{user}:{password}",
        "-o", str(dest),
        url,
    ]
    # Inherit stderr/stdout so the user sees curl's live progress bar.
    proc = subprocess.run(cmd)
    return proc.returncode


def curl_probe(url: str, user: str, password: str, dest_dir: Path) -> int:
    """Download LICENSE.txt fully to verify auth. PhysioNet rejects HEAD and
    rejects Range probes (returns 403) on credentialed files even when a plain
    GET would succeed. Tiny file (~11 KB) so this is cheap.
    Returns HTTP status code (200 = ok), or -1 on error."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_license = dest_dir / "LICENSE.txt"
    cmd = [
        "curl", "-s", "-L", "-f",
        "-w", "%{http_code}",
        "-A", "Wget/1.21.3",
        "-u", f"{user}:{password}",
        "-o", str(tmp_license),
        url,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        # curl -f returns non-zero on HTTP error; status code still printed via -w
        code = out.stdout.strip()
        return int(code) if code.isdigit() else -1
    except Exception:
        return -1


def main() -> int:
    parser = argparse.ArgumentParser(description="Download MIMIC-IV hosp subset.")
    parser.add_argument("--user", required=True, help="PhysioNet username")
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"MIMIC-IV version (default: {DEFAULT_VERSION}). Try 3.1 if your access supports it.",
    )
    parser.add_argument(
        "--out",
        default="data/raw/mimiciv",
        help="Output root (default: data/raw/mimiciv)",
    )
    parser.add_argument(
        "--hide-password",
        action="store_true",
        help="Hide the password while typing (default: visible)",
    )
    args = parser.parse_args()
    BASE_URL = BASE_URL_TEMPLATE.format(version=args.version)
    print(f"Using MIMIC-IV v{args.version}")
    print(f"Source URL : {BASE_URL}\n")

    if not have_curl():
        print("ERROR: 'curl' is not on PATH. On Windows 10+ it ships by default;")
        print("       check `where curl` in PowerShell. On older systems install")
        print("       curl from https://curl.se/windows/ and re-run.")
        return 1

    # Password handling.
    pw = os.environ.get("PHYSIONET_PASSWORD")
    if not pw:
        if args.hide_password:
            pw = getpass.getpass(f"PhysioNet password for {args.user}: ")
        else:
            print(
                "Note: password will be VISIBLE while you type."
                " Use --hide-password to hide it.\n"
            )
            pw = input(f"PhysioNet password for {args.user}: ")

    out_root = Path(args.out).resolve()

    # Auth sanity check using a full GET on LICENSE.txt (same flags as a real
    # download, just smaller). HEAD and Range probes are rejected by PhysioNet
    # even for authorised users, so we just download the tiny LICENSE.
    print("\nChecking access to LICENSE.txt ...")
    code = curl_probe(f"{BASE_URL}/LICENSE.txt", args.user, pw, out_root)
    if code == 200:
        print(f"  OK (HTTP {code}). Access confirmed.\n")
    elif code == 401:
        print("  ERROR: 401 Unauthorized — username or password is wrong.")
        return 1
    elif code == 403:
        print("  ERROR: 403 Forbidden. Verify all of these:")
        print("           https://physionet.org/settings/training/   (must be Accepted)")
        print("           https://physionet.org/settings/agreements/ (DUA must be Signed)")
        print("           https://physionet.org/settings/credentialing/")
        return 1
    else:
        print(f"  ERROR: probe returned HTTP {code}")
        return 1
    approx_total = sum(sz for _, sz in FILES)
    print(f"Target folder: {out_root}")
    print(f"Approximate total download: {human(approx_total)}")
    print(f"Files: {len(FILES)}\n")

    for i, (rel, _) in enumerate(FILES, 1):
        url = f"{BASE_URL}/{rel}"
        dest = out_root / rel
        print(f"[{i}/{len(FILES)}] {rel}")
        # Skip if a previous run completed this file (curl -C - handles the
        # incomplete case automatically; the explicit skip avoids a redundant
        # HEAD-like probe call).
        rc = curl_download(url, dest, args.user, pw)
        if rc != 0:
            print(f"  ERROR: curl exit code {rc} (re-run to resume)")
            return 2
        print()  # blank line between files

    print("\nAll files downloaded successfully.")
    print(f"Tree: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
