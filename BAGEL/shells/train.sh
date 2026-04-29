#!/bin/bash
# train.sh -- BAGEL training launcher.
#
# Storage policy:
#   Hot-writable directories (support append / rename):
#     $LOCAL_RUN_ROOT/<EXP_NAME>/<TIMESTAMP>/
#       ├── logs/          # log.txt (append-only training logs)
#       ├── checkpoints/   # ema.safetensors / model.safetensors / optimizer*.pt
#       └── wandb/         # local offline cache
#
#   Archive (persistent) directory:
#     $ARCHIVE_ROOT/<EXP_NAME>/<TIMESTAMP>/
#       populated periodically by scripts/sync_runs.sh, and flushed once on exit.
#
# Environment variables you may override:
#   EXP_NAME         -- Experiment name. Default: reca_default.
#   LOCAL_RUN_ROOT   -- Local run-root directory. Default: ./runs.
#   ARCHIVE_ROOT     -- Archive root directory. Default: ./archives.
#   SYNC_TO_ARCHIVE  -- Whether to sync logs/checkpoints to the archive dir.
#                       1 = enable (default); 0 = disable (local-only run).
#   SYNC_INTERVAL    -- Background sync interval in seconds. Default: 60.
#   RESUME_FROM      -- Path to resume weights from. If located on a slow FS,
#                       the script can optionally pre-cache it locally.
#
set -euo pipefail

export master_addr=${master_addr:-localhost}
export master_port=${master_port:-12335}

# -- 1. Runtime directories (local, append / rename friendly) ----------------
EXP_NAME="${EXP_NAME:-reca_panoptic_mixed}"
TIMESTAMP="${DEEPGEN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOCAL_RUN_ROOT="${LOCAL_RUN_ROOT:-./runs}"
RUN_DIR="${LOCAL_RUN_ROOT}/${EXP_NAME}/${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
CKPT_DIR="${RUN_DIR}/checkpoints"
WANDB_DIR="${RUN_DIR}/wandb"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${WANDB_DIR}"

# -- 2. Archive directory (persistent storage) -------------------------------
# If the archive root does not exist (e.g. debugging locally), the background
# sync is automatically disabled.
ARCHIVE_ROOT="${ARCHIVE_ROOT:-./archives}"
ARCHIVE_DIR="${ARCHIVE_ROOT}/${EXP_NAME}/${TIMESTAMP}"

# 0 = run purely local, never sync to archive.
# 1 = enable background rsync + final flush on exit + wandb tar archive.
SYNC_TO_ARCHIVE="${SYNC_TO_ARCHIVE:-1}"

echo "=============== BAGEL training directories ==============="
echo "  EXP_NAME        : ${EXP_NAME}"
echo "  TIMESTAMP       : ${TIMESTAMP}"
echo "  Local run dir   : ${RUN_DIR}"
if [[ "${SYNC_TO_ARCHIVE}" == "1" ]]; then
    echo "  Archive dir     : ${ARCHIVE_DIR}"
    echo "  Archive sync    : enabled (SYNC_TO_ARCHIVE=1)"
else
    echo "  Archive dir     : (disabled, SYNC_TO_ARCHIVE=0; run data is local-only!)"
fi
echo "==========================================================="

# -- 3. Tell the Python side where the "local log root" is, so it does not
#       accidentally write logs onto an object-store-mounted path. ----------
export BAGEL_LOCAL_LOG_ROOT="${LOCAL_RUN_ROOT}/_redirected_logs"

# -- 4. Fully disable wandb (no login, no network, no disk writes). ---------
# This training pipeline does not use wandb. We force-disable it here so that
# `wandb.init()` / `wandb.log()` become no-ops under any condition.
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export WANDB_CONSOLE=off
export WANDB_DISABLE_CODE=true
# Strip any stale wandb configuration inherited from the parent shell.
unset WANDB_API_KEY
unset WANDB_ENTITY
unset WANDB_PROJECT
unset WANDB_RUN_ID
unset WANDB_RESUME
# Keep WANDB_DIR pointing to the local disk (some sub-modules still read this
# variable even in disabled mode; we do not want them writing to object store).
export WANDB_DIR="${WANDB_DIR}"
export WANDB_CACHE_DIR="${WANDB_DIR}/cache"
mkdir -p "${WANDB_CACHE_DIR}"

# -- 5. Start the background rsync sync (local -> archive) ------------------
SYNC_PID=""
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYNC_SCRIPT="${SCRIPT_DIR}/../scripts/sync_runs.sh"

start_bg_sync() {
    if [[ "${SYNC_TO_ARCHIVE}" != "1" ]]; then
        echo "[train.sh] Archive sync disabled (SYNC_TO_ARCHIVE=0); logs/ckpts stay local."
        echo "[train.sh]   To archive manually after training, run:"
        echo "[train.sh]     SRC=${RUN_DIR} DST=${ARCHIVE_DIR} bash ${SYNC_SCRIPT} once"
        echo "[train.sh]     SRC=${RUN_DIR} DST=${ARCHIVE_DIR} bash ${SYNC_SCRIPT} wandb_archive"
        return 0
    fi
    if [[ ! -x "${SYNC_SCRIPT}" ]]; then
        chmod +x "${SYNC_SCRIPT}" 2>/dev/null || true
    fi
    # Only start the sync when the archive root is writable (i.e. properly mounted).
    if mkdir -p "${ARCHIVE_DIR}" 2>/dev/null; then
        SRC="${RUN_DIR}" DST="${ARCHIVE_DIR}" INTERVAL="${SYNC_INTERVAL:-3600}" \
            bash "${SYNC_SCRIPT}" loop \
            > "${LOG_DIR}/sync.out" 2>&1 &
        SYNC_PID=$!
        echo "[train.sh] Background sync started (pid=${SYNC_PID}) -> ${ARCHIVE_DIR}"
    else
        echo "[train.sh] WARNING: archive dir ${ARCHIVE_DIR} is not writable; skipping background sync."
        echo "[train.sh]          Training will continue, but run data stays on local disk only."
    fi
}

final_flush() {
    if [[ "${SYNC_TO_ARCHIVE}" != "1" ]]; then
        # Archive sync explicitly disabled -- nothing to do.
        return 0
    fi
    # The training process has exited; make sure we do one last archive sync.
    if [[ -n "${SYNC_PID}" ]] && kill -0 "${SYNC_PID}" 2>/dev/null; then
        echo "[train.sh] Sending TERM to sync process (pid=${SYNC_PID}) for final flush..."
        kill -TERM "${SYNC_PID}" 2>/dev/null || true
        wait "${SYNC_PID}" 2>/dev/null || true
    fi
    if [[ -d "${ARCHIVE_DIR}" ]]; then
        # 1) Normal files (weights / log shards) -- one-off archive, skip wandb.
        SRC="${RUN_DIR}" DST="${ARCHIVE_DIR}" \
            bash "${SYNC_SCRIPT}" once || true
        # 2) Archive the wandb directory as a single tar (one-shot write is
        #    always safe on append/rename-unfriendly mounts like S3 FUSE).
        SRC="${RUN_DIR}" DST="${ARCHIVE_DIR}" \
            bash "${SYNC_SCRIPT}" wandb_archive || true
        echo "[train.sh] Final archive flush completed: ${ARCHIVE_DIR}"
    fi
}
trap 'final_flush' EXIT INT TERM

start_bg_sync

# -- 6. Prepare weights: optionally cache the model to a fast local disk. ----
# Rationale: reading a ~29GB ema.safetensors directly from a network/object-
# store-mounted filesystem can take 5-30 minutes because of FUSE mmap and
# single-threaded sequential I/O. Copying the weights to a local SSD and
# reading them from there usually finishes loading within 30s-1min.
#
# You can override these paths via environment variables, e.g.:
#   BAGEL_CODE_ROOT=./ \
#   MODEL_PATH=$BAGEL_CODE_ROOT/ckpt/BAGEL-7B-MoT \
#   DATASET_CFG=$BAGEL_CODE_ROOT/data/configs/high_mixed_sam.yaml \
#       bash shells/train.sh
#
# Cache control knobs:
#   USE_LOCAL_CACHE=0   Disable caching and use MODEL_PATH directly (default: on).
#   LOCAL_WEIGHTS_ROOT  Cache root directory (default: ./local_weights).
#   FORCE_REFRESH=1     Force re-copy, ignoring any existing cache.
BAGEL_CODE_ROOT="${BAGEL_CODE_ROOT:-.}"
MODEL_PATH="${MODEL_PATH:-${BAGEL_CODE_ROOT}/ckpt/BAGEL-7B-MoT}"
RESUME_FROM="${RESUME_FROM:-${MODEL_PATH}}"
DATASET_CFG="${DATASET_CFG:-${BAGEL_CODE_ROOT}/data/configs/high_mixed_sam.yaml}"

USE_LOCAL_CACHE="${USE_LOCAL_CACHE:-0}"
LOCAL_WEIGHTS_ROOT="${LOCAL_WEIGHTS_ROOT:-./local_weights}"
FORCE_REFRESH="${FORCE_REFRESH:-0}"

# Mirror the weights directory pointed to by $1 (if remote) onto the local
# cache. Uses flock + a .ready marker for concurrency safety, and validates
# using a "file-count + total-byte-size" fingerprint of the source directory
# so that cache entries are invalidated automatically when the source changes.
#
# Contract: the function prints *exactly one line* on stdout -- the final
#   usable path, to be captured with $(...). All progress logs go to stderr
#   (>&2) so they do not contaminate the return value.
cache_weights_to_local_ssd() {
    local src="$1"

    # Empty input / already under the cache root / caching disabled -> passthrough.
    if [[ -z "$src" || "$USE_LOCAL_CACHE" != "1" ]]; then
        echo "$src"
        return 0
    fi
    case "$src" in
        "${LOCAL_WEIGHTS_ROOT}"/*)
            echo "$src"
            return 0
            ;;
    esac
    if [[ ! -d "$src" ]]; then
        # Not a directory (e.g. a single .safetensors file) -- pass through.
        echo "$src"
        return 0
    fi

    mkdir -p "$LOCAL_WEIGHTS_ROOT" >&2

    # Compute an 8-char hash of the source path to avoid name collisions.
    local abs_src tag dst ready lock
    abs_src="$(readlink -f "$src")"
    tag="$(printf '%s' "$abs_src" | md5sum | awk '{print $1}' | cut -c1-8)"
    dst="${LOCAL_WEIGHTS_ROOT}/$(basename "$abs_src")-${tag}"
    ready="${dst}/.ready"
    lock="${LOCAL_WEIGHTS_ROOT}/.${tag}.lock"

    # Fingerprint = "file-count + total byte-size" of the source directory.
    local fingerprint_src
    fingerprint_src="$(
        cd "$abs_src" && \
        find . -type f -printf '%s\n' 2>/dev/null | \
        awk 'BEGIN{n=0;s=0}{n++;s+=$1}END{printf "files=%d bytes=%d\n",n,s}'
    )"

    # Acquire an advisory file lock to avoid concurrent copies from racing.
    (
        flock -w 3600 200 || {
            echo "[cache_weights] Timed out acquiring lock: $lock" >&2
            exit 1
        }

        local need_copy=1
        if [[ "$FORCE_REFRESH" != "1" && -f "$ready" ]]; then
            local fingerprint_cached
            fingerprint_cached="$(head -n1 "$ready" 2>/dev/null || true)"
            if [[ "$fingerprint_cached" == "$fingerprint_src" ]]; then
                need_copy=0
                echo "[cache_weights] Cache hit: $dst" >&2
            else
                echo "[cache_weights] Fingerprint changed, re-copying" >&2
                echo "    old: $fingerprint_cached" >&2
                echo "    new: $fingerprint_src" >&2
            fi
        fi

        if [[ "$need_copy" == "1" ]]; then
            echo "[cache_weights] First-time copy: $abs_src -> $dst" >&2
            echo "[cache_weights] Fingerprint: $fingerprint_src" >&2
            # Copy into a .tmp directory first, then atomically rename to
            # prevent half-copied artifacts from being reused on interrupt.
            rm -rf "${dst}.tmp" 2>/dev/null || true
            mkdir -p "${dst}.tmp"
            local t0
            t0="$(date +%s)"
            # -a preserves permissions/timestamps; cp stderr goes to the
            # terminal, stdout is discarded.
            if cp -a "$abs_src"/. "${dst}.tmp"/ 1>&2; then
                rm -rf "$dst" 2>/dev/null || true
                mv "${dst}.tmp" "$dst"
                printf '%s\n%s\n' "$fingerprint_src" "copied_at=$(date -Iseconds)" > "$ready"
                local t1=$(($(date +%s) - t0))
                echo "[cache_weights] Copy complete (took ${t1}s): $dst" >&2
            else
                echo "[cache_weights] ERROR: copy failed; falling back to original path: $abs_src" >&2
                rm -rf "${dst}.tmp" 2>/dev/null || true
                exit 1
            fi
        fi
    ) 200>"$lock"
    local rc=$?

    # The sub-shell's flock has already been released; check the final state.
    if [[ $rc -eq 0 && -f "$ready" ]]; then
        echo "$dst"
    else
        echo "$src"
    fi
}

# Run the cache step.
if [[ "${USE_LOCAL_CACHE}" == "1" ]]; then
    echo "[train.sh] Checking local weight cache ..."
    CACHED_MODEL_PATH="$(cache_weights_to_local_ssd "$MODEL_PATH")"
    # If resume_from shares the same source as model_path, reuse the cache;
    # otherwise cache it separately.
    if [[ "$RESUME_FROM" == "$MODEL_PATH" ]]; then
        CACHED_RESUME_FROM="$CACHED_MODEL_PATH"
    else
        CACHED_RESUME_FROM="$(cache_weights_to_local_ssd "$RESUME_FROM")"
    fi
    MODEL_PATH="$CACHED_MODEL_PATH"
    RESUME_FROM="$CACHED_RESUME_FROM"
    echo "[train.sh] Final MODEL_PATH   = $MODEL_PATH"
    echo "[train.sh] Final RESUME_FROM  = $RESUME_FROM"
fi

export PYTHONPATH=.

# -- 7. Launch training -----------------------------------------------------
# auto_resume    resume_model_only
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=8 \
    --master_addr="${master_addr}" \
    --master_port="${master_port}" \
    train/pretrain_unified_navit.py \
    --model_path "${MODEL_PATH}" \
    --dataset_config_file "${DATASET_CFG}" \
    --layer_module Qwen2MoTDecoderLayer \
    --max_latent_size 64 \
    --freeze_vae True \
    --freeze_vit False \
    --freeze_llm False \
    --freeze_und False \
    --finetune_from_ema True \
    --resume_from "${RESUME_FROM}" \
    --results_dir "${LOG_DIR}" \
    --checkpoint_dir "${CKPT_DIR}" \
    --save_every "${SAVE_EVERY:-1000}" \
    --save_after "${SAVE_AFTER:-7000}" \
    --wandb_runid 1 \
    --use_flex \
    --lr 0.0001
# lr: BAGEL official default is 0.00004; in this project we use 0.0001.
