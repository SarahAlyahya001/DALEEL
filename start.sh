#!/bin/sh
# Sets up and starts DALEEL on Linux and macOS: sh start.sh (see README.md). Setup and startup are
# checked automatically on Ubuntu; macOS has not been tested.
# The first run downloads Python 3.10.11, the packages in requirements.txt and the two model files
# into this folder (.tools, .venv and weights). Later runs start DALEEL straight away.
# Nothing is installed elsewhere on the computer, and the instance folder (accounts, passwords,
# missions and evidence) is never changed by the setup.
set -eu

ROOT=$(cd "$(dirname "$0")" && pwd)
TOOLS="$ROOT/.tools"
VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"
MARKER="$VENV/daleel-setup.txt"
DATABASE="$ROOT/instance/v2/daleel.db"
SETUP_ONLY=no
[ "${1:-}" = "--setup-only" ] && SETUP_ONLY=yes

# DALEEL was developed, tested and evaluated with Python 3.10.11 and the package versions in
# requirements-lock.txt; PyTorch 2.3.1 does not install on Python 3.13 or later. python.org publishes
# no ready-to-unpack Python for Linux or macOS, so the python-build-standalone build of 3.10.11 is used,
# pinned by the SHA-256 published with its release.
PYTHON_VERSION=3.10.11
PYTHON_RELEASE=https://github.com/astral-sh/python-build-standalone/releases/download/20230507
PYTHON_DIR="$TOOLS/python-$PYTHON_VERSION"
BASE_PYTHON="$PYTHON_DIR/bin/python3"
export PIP_CACHE_DIR="$TOOLS/pip-cache"  # finished downloads are kept if the setup is interrupted
export PIP_CONFIG_FILE=/dev/null         # ignore pip settings made for other projects
export PIP_DISABLE_PIP_VERSION_CHECK=1
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV

fail() {
    printf '\nDALEEL could not start: %s\n' "$1" >&2
    printf 'Run sh start.sh again after fixing the problem; finished downloads are kept.\n' >&2
    exit 1
}

step() { printf '\n%s\n' "$1"; }

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

download() {
    if command -v curl >/dev/null 2>&1; then curl -fsSL "$1" -o "$2"
    else wget -q "$1" -O "$2"
    fi
}

case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)
        PYTHON_FILE=cpython-3.10.11%2B20230507-x86_64-unknown-linux-gnu-install_only.tar.gz
        PYTHON_SHA256=c5bcaac91bc80bfc29cf510669ecad12d506035ecb3ad85ef213416d54aecd79 ;;
    Linux-aarch64 | Linux-arm64)
        PYTHON_FILE=cpython-3.10.11%2B20230507-aarch64-unknown-linux-gnu-install_only.tar.gz
        PYTHON_SHA256=c7573fdb00239f86b22ea0e8e926ca881d24fde5e5890851339911d76110bc35 ;;
    Darwin-arm64)
        PYTHON_FILE=cpython-3.10.11%2B20230507-aarch64-apple-darwin-install_only.tar.gz
        PYTHON_SHA256=8348bc3c2311f94ec63751fb71bd0108174be1c4def002773cf519ee1506f96f ;;
    *) fail "PyTorch 2.3.1 is not available for this computer ($(uname -s) $(uname -m)). Use Docker instead (docs/DEPLOYMENT.md)." ;;
esac

setup_id() {
    printf 'python %s %s %s\n' "$PYTHON_VERSION" "$(sha256 "$ROOT/requirements.txt")" "$(sha256 "$ROOT/requirements-lock.txt")"
}

install_python() {
    [ -x "$BASE_PYTHON" ] && return 0
    echo "Downloading Python $PYTHON_VERSION from github.com/astral-sh/python-build-standalone"
    archive="$TOOLS/python-$PYTHON_VERSION.tar.gz"
    download "$PYTHON_RELEASE/$PYTHON_FILE" "$archive" ||
        fail "Python $PYTHON_VERSION could not be downloaded. Check the internet connection."
    if [ "$(sha256 "$archive")" != "$PYTHON_SHA256" ]; then
        rm -f "$archive"
        fail "the downloaded Python $PYTHON_VERSION failed its SHA-256 check and was deleted."
    fi
    rm -rf "$PYTHON_DIR.part"
    mkdir -p "$PYTHON_DIR.part"
    tar -xzf "$archive" -C "$PYTHON_DIR.part"
    mv "$PYTHON_DIR.part/python" "$PYTHON_DIR"
    rm -rf "$PYTHON_DIR.part" "$archive"
}

# The environment must be built on the Python checked above, not on another Python 3.10.
venv_uses_daleel_python() {
    [ -x "$PYTHON" ] && [ "$("$PYTHON" -c 'import os, sys; print(os.path.samefile(sys.base_prefix, sys.argv[1]))' \
        "$PYTHON_DIR" 2>/dev/null)" = "True" ]
}

install_daleel() {
    step "[1/3] Getting Python $PYTHON_VERSION (first start only)"
    mkdir -p "$TOOLS"
    install_python
    if [ -e "$VENV" ] && ! venv_uses_daleel_python; then
        [ -L "$VENV" ] && fail "the .venv folder is a link to another folder. Remove the link, then run sh start.sh again."
        echo "Replacing the existing .venv folder, which was not made with DALEEL's Python $PYTHON_VERSION."
        echo "Accounts, passwords, missions and evidence are kept in the instance folder and are not affected."
        rm -rf "$VENV"
    fi
    if [ ! -x "$PYTHON" ]; then
        "$BASE_PYTHON" -m venv "$VENV" || fail "the Python $PYTHON_VERSION environment could not be created."
    fi
    step "[2/3] Downloading and installing the Python packages (first start only)"
    echo "The time this takes depends mainly on the speed of the internet connection."
    if [ "$(uname -s)" = Linux ]; then
        # The CPU builds of PyTorch (as in the Dockerfile) avoid several gigabytes of GPU libraries.
        "$PYTHON" -m pip install --timeout 300 --retries 10 --extra-index-url https://download.pytorch.org/whl/cpu \
            --constraint "$ROOT/requirements-lock.txt" torch==2.3.1+cpu torchvision==0.18.1+cpu ||
            fail "the Python packages could not be installed."
    fi
    "$PYTHON" -m pip install --timeout 300 --retries 10 -r "$ROOT/requirements.txt" \
        --constraint "$ROOT/requirements-lock.txt" || fail "the Python packages could not be installed."
    if ! "$PYTHON" -c "import cv2" >/dev/null 2>&1; then
        fail "OpenCV needs two system libraries. On Ubuntu or Debian, run: sudo apt install libgl1 libglib2.0-0"
    fi
    setup_id > "$MARKER"
    step "[3/3] Downloading and checking the two detection models (29 MB)"
}

cd "$ROOT"
if [ -x "$PYTHON" ] && [ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "$(setup_id)" ]; then
    echo "Checking the detection models..."
else
    install_daleel
fi
"$PYTHON" scripts/setup_weights.py >/dev/null || fail "the detection models could not be downloaded or checked."
if [ "$SETUP_ONLY" = yes ]; then
    printf '\nDALEEL is set up. Run sh start.sh to start it.\n'
    exit 0
fi

PORT="${DALEEL_PORT:-5000}"
URL="http://127.0.0.1:$PORT"
if command -v curl >/dev/null 2>&1; then
    if curl -fsS --max-time 3 "$URL/login" 2>/dev/null | grep -q DALEEL; then
        echo "DALEEL is already running at $URL"
        exit 0
    fi
    if curl -s -o /dev/null --max-time 3 "$URL/"; then
        fail "another program is using port $PORT (on macOS, often AirPlay Receiver). Close it, or run: DALEEL_PORT=5050 sh start.sh"
    fi
fi

echo
if [ ! -f "$DATABASE" ]; then
    echo "First start: DALEEL now creates the Administrator account and shows its temporary username"
    echo "and password below, once. Copy them, then open $URL in your browser."
else
    echo "Starting DALEEL at $URL"
fi
echo "Keep this terminal open while you use DALEEL. Press Ctrl+C to stop it."
echo
exec "$PYTHON" run.py
