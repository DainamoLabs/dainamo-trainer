#!/usr/bin/env bash
# Dainamo Trainer launcher: sets up the Python env and web UI on first run (and again when
# requirements.txt or the web UI's source changes, as after a git pull), then starts the server.
# Options pass through, e.g.
#   ./launch.sh --host 0.0.0.0        (see ./launch.sh --help)
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "Creating Python environment…"
  if command -v uv >/dev/null; then
    uv venv --python 3.12 .venv
  else
    python3 -m venv .venv
  fi
fi

if ! cmp -s requirements.txt .venv/requirements.installed; then
  echo "Installing Python packages, PyTorch with CUDA among them. This takes a while the first time…"
  if command -v uv >/dev/null; then
    uv pip install --python .venv/bin/python -r requirements.txt
  else
    .venv/bin/pip install -r requirements.txt
  fi
  cp requirements.txt .venv/requirements.installed
fi

if ! .venv/bin/python -m dainamo.webui check; then
  if command -v npm >/dev/null; then
    if [ -f web/dist/index.html ]; then echo "The web UI changed since it was built. Rebuilding it…"; else echo "Building web UI…"; fi
    (cd web && npm install && npm run build)
    .venv/bin/python -m dainamo.webui built
  elif [ -f web/dist/index.html ]; then
    echo "Note: the web UI changed since it was built, but npm wasn't found to rebuild it, so this is the older UI." >&2
  else
    echo "The web UI isn't built and npm wasn't found. Install Node.js 18 or newer and run this again." >&2
    exit 1
  fi
fi

command -v ffmpeg >/dev/null || echo "Note: ffmpeg wasn't found. Video datasets and video samples need it."

exec .venv/bin/python -m dainamo "$@"
