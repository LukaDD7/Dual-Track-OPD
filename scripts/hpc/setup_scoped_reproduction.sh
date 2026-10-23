#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG="${SCOPED_CONFIG:-$ROOT/configs/external/scoped_expert_reproduction.yaml}"
REPO_ROOT="${SCOPED_REPO_ROOT:-$ROOT/fc-opd-storage/external/scoped/repo}"
STORAGE_ROOT="${SCOPED_STORAGE_ROOT:-$ROOT/fc-opd-storage/external/scoped/storage}"
INSTALL=0

usage() {
    cat <<'EOF'
Usage: bash scripts/hpc/setup_scoped_reproduction.sh [--install]

Clones or updates the pinned NEUIR/SCOPED checkout and prepares the external
storage layout. Pass --install to create .venv-rl and .venv-scoped from the
upstream requirements files.

Environment overrides:
  SCOPED_REPO_ROOT     External upstream checkout path
  SCOPED_STORAGE_ROOT  External data/output/log path
  SCOPED_CONFIG        Local reproduction contract YAML
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install) INSTALL=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

UPSTREAM_URL=$(python -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["scoped_reproduction"]["upstream"]["repo_url"])' "$CONFIG")
COMMIT=$(python -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["scoped_reproduction"]["upstream"]["commit"])' "$CONFIG")

mkdir -p "$(dirname "$REPO_ROOT")" "$STORAGE_ROOT"
if [[ ! -d "$REPO_ROOT/.git" ]]; then
    git clone "$UPSTREAM_URL" "$REPO_ROOT"
else
    git -C "$REPO_ROOT" remote get-url origin >/dev/null
    git -C "$REPO_ROOT" fetch origin
fi

DIRTY=$(git -C "$REPO_ROOT" status --short --untracked-files=all -- \
    ':(exclude).venv-rl' ':(exclude).venv-scoped' \
    ':(exclude)**/__pycache__/**' ':(exclude)**/*.egg-info/**')
if [[ -n "$DIRTY" ]]; then
    echo "SCOPED checkout is dirty; refusing to reset or overwrite it:" >&2
    printf '%s\n' "$DIRTY" >&2
    exit 1
fi

git -C "$REPO_ROOT" checkout "$COMMIT"
git -C "$REPO_ROOT" rev-parse HEAD | grep -qx "$COMMIT" || {
    echo "Failed to pin SCOPED checkout to $COMMIT" >&2
    exit 1
}

if [[ "$INSTALL" == 1 ]]; then
    if ! command -v python3.12 >/dev/null 2>&1; then
        echo "python3.12 is required to create the upstream virtualenvs" >&2
        exit 1
    fi
    python3.12 -m venv "$REPO_ROOT/.venv-rl"
    RL_COMPAT="$STORAGE_ROOT/requirements-rl.compat.txt"
    python -c 'import sys; from pathlib import Path; p=Path(sys.argv[1]); q=Path(sys.argv[2]); q.write_text(p.read_text().replace("packaging==24.2", "packaging==25.0"))' \
        "$REPO_ROOT/requirements-rl.txt" "$RL_COMPAT"
    (cd "$REPO_ROOT" && "$REPO_ROOT/.venv-rl/bin/python" -m pip install -r "$RL_COMPAT")
    python3.12 -m venv "$REPO_ROOT/.venv-scoped"
    SCOPED_COMPAT="$STORAGE_ROOT/requirements-scoped.compat.txt"
    python -c 'import sys; from pathlib import Path; p=Path(sys.argv[1]); q=Path(sys.argv[2]); q.write_text(p.read_text().replace("flash-attn==2.8.3.post1", "flash-attn==2.8.3"))' \
        "$REPO_ROOT/requirements.txt" "$SCOPED_COMPAT"
    FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"
    (cd "$REPO_ROOT" && "$REPO_ROOT/.venv-scoped/bin/python" -m pip install \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0)
    (cd "$REPO_ROOT" && "$REPO_ROOT/.venv-scoped/bin/python" -m pip install "$FLASH_ATTN_WHEEL")
    (cd "$REPO_ROOT" && "$REPO_ROOT/.venv-scoped/bin/python" -m pip install \
        nvidia-cuda-runtime-cu12==12.8.90)
    (cd "$REPO_ROOT" && "$REPO_ROOT/.venv-scoped/bin/python" -m pip install -r "$SCOPED_COMPAT")
fi

python "$ROOT/scripts/hpc/scoped_reproduction.py" \
    --config "$CONFIG" \
    --repo-root "$REPO_ROOT" \
    --storage-root "$STORAGE_ROOT" \
    --stage setup \
    --output "$STORAGE_ROOT/records/setup.json"

echo "SCOPED checkout ready: $REPO_ROOT@$COMMIT"
echo "External storage: $STORAGE_ROOT"
if [[ "$INSTALL" != 1 ]]; then
    echo "Virtualenvs not installed; rerun with --install on the HPC CPU/GPU node."
fi
