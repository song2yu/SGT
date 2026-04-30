# Usage: source activate_env.sh
# Activate the local OmniGen2 virtualenv. Override VENV_DIR to point elsewhere.
export UV_LINK_MODE=copy
: "${VENV_DIR:=${HOME}/venvs/omnigen2}"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
    echo "[OK] OmniGen2 environment activated: $(which python)"
else
    echo "[WARN] virtualenv not found at ${VENV_DIR}. Set VENV_DIR before sourcing."
fi
