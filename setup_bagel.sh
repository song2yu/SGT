#!/usr/bin/env bash
# setup_bagel.sh - Create the BAGEL environment on a local disk via uv and install deps.
# Design principles:
#   - Keep only code / data / weights / checkpoints on EFS
#   - Put the venv on a local disk (overlay or local-ssd) to avoid unsupported
#     symlinks and performance issues
set -euo pipefail

# ========== Config ==========
ENV_NAME="bagel"
PYTHON_VERSION="3.10"
PROJECT_DIR="BAGEL"
FLASH_ATTN_VERSION="2.7.2.post1"

# venv location: prefer local-ssd, fall back to /root (overlay).
# Override via the VENV_ROOT env var, e.g.: VENV_ROOT=/local-ssd/venvs ./setup_bagel.sh
if [[ -n "${VENV_ROOT:-}" ]]; then
    VENV_BASE="${VENV_ROOT}"
elif [[ -d "/local-ssd" && -w "/local-ssd" ]]; then
    VENV_BASE="/local-ssd/venvs"
else
    VENV_BASE="/root/venvs"
fi
VENV_DIR="${VENV_BASE}/${ENV_NAME}"
# ============================

# Force copy mode to avoid symlink failures on EFS / NFS
export UV_LINK_MODE=copy

# Record the absolute project path (script is usually run from the project's
# parent directory or from inside the project itself)
START_DIR="$(pwd)"

echo "========================================"
echo "[INFO] Env name:       ${ENV_NAME}"
echo "[INFO] Python version: ${PYTHON_VERSION}"
echo "[INFO] Project dir:    ${START_DIR}/${PROJECT_DIR}"
echo "[INFO] venv path:      ${VENV_DIR}   (local disk)"
echo "[INFO] UV_LINK_MODE:   ${UV_LINK_MODE}"
echo "========================================"

# 0. Check / install uv
if ! command -v uv &> /dev/null; then
    echo "[INFO] uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[INFO] uv version: $(uv --version)"

# 1. Create the venv parent directory
mkdir -p "${VENV_BASE}"

# 2. Create virtual environment (on local disk)
if [[ -d "${VENV_DIR}" ]]; then
    echo "[WARN] ${VENV_DIR} already exists, skipping creation (remove it first to rebuild)"
else
    echo "[INFO] Creating virtual environment: ${VENV_DIR} (Python ${PYTHON_VERSION})..."
    uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}" --seed
fi

# 3. Activate the environment
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "[INFO] Current Python: $(which python)"
echo "[INFO] Python version: $(python --version)"

# 4. Enter the project directory (code on EFS is fine, only the venv must be off-EFS)
cd "${START_DIR}/${PROJECT_DIR}"

# 5. Install requirements.txt
echo "[INFO] Installing requirements.txt..."
uv pip install -r requirements.txt

# 6. Install flash-attn (requires --no-build-isolation)
echo "[INFO] Installing flash-attn==${FLASH_ATTN_VERSION}..."
uv pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation

# 7. Drop an activation shortcut in the project dir for convenience
ACTIVATE_SHORTCUT="${START_DIR}/${PROJECT_DIR}/activate_env.sh"
cat > "${ACTIVATE_SHORTCUT}" <<EOF
# Usage: source activate_env.sh
export UV_LINK_MODE=copy
source "${VENV_DIR}/bin/activate"
echo "[OK] BAGEL environment activated: \$(which python)"
EOF
chmod +x "${ACTIVATE_SHORTCUT}" 2>/dev/null || \
    echo "[WARN] Could not chmod +x ${ACTIVATE_SHORTCUT} (likely on EFS and not the owner); safe to ignore, just use 'source activate_env.sh'."

echo ""
echo "========================================"
echo "[DONE] BAGEL environment is ready!"
echo ""
echo "venv path: ${VENV_DIR}"
echo ""
echo "Activate with one of:"
echo "  1) source ${VENV_DIR}/bin/activate"
echo "  2) cd ${PROJECT_DIR} && source activate_env.sh"
echo ""
echo "Deactivate: deactivate"
echo "========================================"
