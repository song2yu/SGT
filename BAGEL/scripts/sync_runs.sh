#!/usr/bin/env bash
# sync_runs.sh -- one-way sync from a local disk (SRC) to an archive / object
#                 store (DST).
#
# Background -- handling append/rename-unfriendly mounts (e.g. S3 FUSE):
#   Object-store-backed mounts typically do NOT support file rename, append
#   or overwrite. In practice, rsync-ing onto such a mount means each file
#   can effectively be written only once -- any retry will fail with EPERM
#   (Operation not permitted).
#
# Key design choices:
#   1) rsync uses --inplace + --ignore-existing
#      -- files that already exist on DST are not "updated" (the mount would
#         reject that anyway), which also avoids spamming EPERM errors.
#   2) Directories that are continuously rewritten in place (wandb/,
#      *.sqlite, sync.out, ...) are excluded from the loop sync. They are
#      handled separately at the end of training via the once / wandb_archive
#      modes.
#   3) For wandb/ we build a single tar archive -- one-shot writes are safe
#      on any filesystem, including object-store-backed ones.
#
# Usage:
#   # Background periodic sync (every INTERVAL seconds, excludes wandb)
#   SRC=... DST=... INTERVAL=60 bash scripts/sync_runs.sh loop &
#
#   # One-shot sync (final flush on training exit / crash; excludes wandb)
#   SRC=... DST=... bash scripts/sync_runs.sh once
#
#   # Pack wandb/ into a tar archive (call once at the very end of training)
#   SRC=... DST=... bash scripts/sync_runs.sh wandb_archive
#
set -u  # deliberately NOT -e: occasional non-zero rsync exit codes must not
        # kill the long-running background process.

MODE="${1:-loop}"            # loop | once | wandb_archive
SRC="${SRC:?SRC must be set (local run directory, e.g. ./runs/exp/ts)}"
DST="${DST:?DST must be set (archive directory, e.g. ./archives/exp/ts)}"
INTERVAL="${INTERVAL:-60}"
LOG_TAG="[sync_runs]"

# Common rsync flags:
#   -rltD                          recursive; keep symlinks, mtimes, special files
#                                  (avoid -a so we skip chmod/chown)
#   --inplace                      write the destination file directly, no tmp+rename
#                                  (append/rename-unfriendly mounts do not support rename)
#   --ignore-existing              skip if the destination exists (overwrite is not allowed)
#   --partial                      allow resuming partially transferred files
#   --no-perms/owner/group/...     skip chmod/chown (silently ignored by object-store FUSE)
#   --omit-dir-times               skip directory mtime (may be unsupported)
#   --exclude ...                  ignore runtime caches / continuously-changing files
#
# Note: --ignore-existing also means that locally-modified files are *never*
# re-pushed. This is a required compromise (a re-push would fail anyway).
# Checkpoints are unaffected because every step produces a new directory with
# unique filenames. Logs (log.txt is append-only) ARE affected -- we therefore
# archive them separately via a "tail-rotate" scheme (see rotate_append_log).
RSYNC_COMMON=(
    -rltD
    --inplace
    --ignore-existing
    --partial
    --no-perms --no-owner --no-group --omit-dir-times
    --exclude='*.tmp'
    --exclude='*.swp'
    --exclude='*.sqlite'
    --exclude='*.sqlite-*'
    --exclude='__pycache__/'
    --exclude='.nfs*'
    --exclude='sync.out'          # do not push our own sync logs back
)

# Extra excludes for the loop mode: directories that are continuously
# rewritten are skipped in loop and handled later by wandb_archive / once.
RSYNC_LOOP_EXTRA_EXCLUDES=(
    --exclude='wandb/'
    --exclude='logs/log.txt'      # append-only log; see rotate_append_log
    --exclude='logs/log.txt.*'
)

# --------------------------------------------------------------------
# "Shard archive" for append-only logs: avoid log.txt repeatedly
# triggering EPERM on append-unfriendly mounts.
#
# Approach: at each sync cycle, copy the current content of log.txt into a
# one-shot shard file log.txt.<seq>, then truncate the live log file. Shard
# files are only written once, which is always safe on object-store mounts.
# --------------------------------------------------------------------
rotate_append_log() {
    local log_file="${SRC}/logs/log.txt"
    local part_dir="${SRC}/logs"
    [[ -f "$log_file" ]] || return 0
    # Only rotate non-empty logs.
    if [[ ! -s "$log_file" ]]; then
        return 0
    fi
    local seq
    seq="$(date +%s)"
    local part_file="${part_dir}/log.txt.${seq}"
    # cp a snapshot of the current content, then truncate the live file.
    if cp "$log_file" "$part_file" 2>/dev/null; then
        : > "$log_file" 2>/dev/null || true
    fi
    # Push the freshly-created shard to DST (only log.txt.* one-shot files).
    mkdir -p "${DST}/logs" 2>/dev/null || true
    rsync "${RSYNC_COMMON[@]}" \
        --include='log.txt.*' --exclude='*' \
        "${part_dir}/" "${DST}/logs/" 2>/dev/null || true
}

# --------------------------------------------------------------------
# Compute rsync --exclude fragments for "not-yet-finished" checkpoint
# step directories.
#
# Heuristic: treat checkpoints/<step>/ as complete only when scheduler.pt
# exists (scheduler.pt is the last small file written by the save routine).
# This prevents rsync from uploading half-written large files (ema/
# optimizer) as truncated artifacts -- on append/rename-unfriendly mounts
# such a truncated upload cannot be overwritten later, so it would leave
# the archived checkpoint permanently corrupted.
# --------------------------------------------------------------------
build_incomplete_step_excludes() {
    local ckpt_root="${SRC}/checkpoints"
    [[ -d "$ckpt_root" ]] || return 0
    local step_dir step_name
    for step_dir in "$ckpt_root"/*/; do
        [[ -d "$step_dir" ]] || continue
        step_name="$(basename "$step_dir")"
        if [[ ! -f "${step_dir}scheduler.pt" ]]; then
            # Exclude the full step directory (path relative to SRC).
            printf -- '--exclude=checkpoints/%s/\n' "$step_name"
        fi
    done
}

# --------------------------------------------------------------------
# Loop sync: pushes newly produced files (e.g. weights), but leaves
# wandb/ and the append-only log.txt alone.
# --------------------------------------------------------------------
do_sync_loop() {
    if [[ ! -d "$SRC" ]]; then
        echo "$LOG_TAG SRC not ready yet: $SRC"
        return 0
    fi
    mkdir -p "$DST" 2>/dev/null || true
    # Dynamically exclude incomplete (scheduler.pt missing) checkpoint steps.
    local incomplete_excludes=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && incomplete_excludes+=("$line")
    done < <(build_incomplete_step_excludes)
    if (( ${#incomplete_excludes[@]} > 0 )); then
        echo "$LOG_TAG Skipping incomplete ckpt steps: ${incomplete_excludes[*]#--exclude=checkpoints/}"
    fi
    rsync "${RSYNC_COMMON[@]}" "${RSYNC_LOOP_EXTRA_EXCLUDES[@]}" \
        "${incomplete_excludes[@]}" \
        "$SRC"/ "$DST"/ 2>/dev/null
    local rc=$?
    # rsync return codes 23/24 (partial transfer / vanished source files) are
    # expected on object-store-backed mounts; just emit a short note.
    if [[ $rc -ne 0 && $rc -ne 23 && $rc -ne 24 ]]; then
        echo "$LOG_TAG rsync(loop) rc=$rc"
    fi
    # Also rotate the append-only log once more.
    rotate_append_log
    return 0
}

# --------------------------------------------------------------------
# One-shot flush on training exit: push every new file, still skipping
# wandb/ (handled by wandb_archive).
# --------------------------------------------------------------------
do_sync_once() {
    if [[ ! -d "$SRC" ]]; then
        echo "$LOG_TAG SRC does not exist: $SRC"
        return 0
    fi
    mkdir -p "$DST" 2>/dev/null || true
    # Same skip policy for incomplete ckpt steps (missing scheduler.pt).
    local incomplete_excludes=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && incomplete_excludes+=("$line")
    done < <(build_incomplete_step_excludes)
    if (( ${#incomplete_excludes[@]} > 0 )); then
        echo "$LOG_TAG once: skipping incomplete ckpt steps: ${incomplete_excludes[*]#--exclude=checkpoints/}"
    fi
    rsync "${RSYNC_COMMON[@]}" --exclude='wandb/' \
        "${incomplete_excludes[@]}" \
        "$SRC"/ "$DST"/ 2>/dev/null
    local rc=$?
    if [[ $rc -ne 0 && $rc -ne 23 && $rc -ne 24 ]]; then
        echo "$LOG_TAG rsync(once) rc=$rc"
    fi
    rotate_append_log
    # One more pass: push any remaining tail content of log.txt as log.txt.final.
    if [[ -f "${SRC}/logs/log.txt" && -s "${SRC}/logs/log.txt" ]]; then
        cp "${SRC}/logs/log.txt" "${SRC}/logs/log.txt.final" 2>/dev/null || true
        rsync "${RSYNC_COMMON[@]}" --include='log.txt.final' --exclude='*' \
            "${SRC}/logs/" "${DST}/logs/" 2>/dev/null || true
    fi
    return 0
}

# --------------------------------------------------------------------
# wandb archive: pack wandb/ as a tar and push it to DST. A tar file is
# only created once, so it never triggers append/rename operations.
# --------------------------------------------------------------------
do_wandb_archive() {
    local wandb_src="${SRC}/wandb"
    if [[ ! -d "$wandb_src" ]]; then
        echo "$LOG_TAG wandb dir does not exist, skipping: $wandb_src"
        return 0
    fi
    mkdir -p "${DST}/wandb_archives" 2>/dev/null || true
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local tar_name="wandb_${ts}.tar.gz"
    local tar_dst="${DST}/wandb_archives/${tar_name}"

    # Important: create the tar on local disk (/tmp) first, then cp to DST.
    # tar itself seek/truncates its output while writing, which is not
    # supported by append/rename-unfriendly mounts -- so we never tar
    # directly onto DST.
    local tmp_tar
    tmp_tar="$(mktemp -t "wandb_XXXXXX.tar.gz")"
    echo "$LOG_TAG Packing wandb into ${tmp_tar} ..."
    if tar -czf "$tmp_tar" -C "$SRC" wandb 2>/dev/null; then
        echo "$LOG_TAG Uploading to ${tar_dst} ..."
        # cp is a one-shot sequential write; safe on object-store mounts.
        if cp "$tmp_tar" "$tar_dst" 2>/dev/null; then
            echo "$LOG_TAG wandb archive complete: ${tar_dst}"
        else
            echo "$LOG_TAG WARNING: failed to upload wandb tar: ${tar_dst}"
        fi
        rm -f "$tmp_tar"
    else
        echo "$LOG_TAG WARNING: failed to pack wandb"
        rm -f "$tmp_tar"
        return 1
    fi
}

case "$MODE" in
    once)
        echo "$LOG_TAG One-shot sync $SRC -> $DST"
        do_sync_once
        ;;
    wandb_archive)
        echo "$LOG_TAG wandb archive $SRC/wandb -> $DST/wandb_archives/"
        do_wandb_archive
        ;;
    loop)
        echo "$LOG_TAG Background loop sync started (pid=$$) $SRC -> $DST, interval ${INTERVAL}s (wandb/ excluded)"
        trap 'echo "$LOG_TAG Received exit signal, running final sync..."; do_sync_once; exit 0' TERM INT
        while true; do
            do_sync_loop
            sleep "$INTERVAL" &
            wait $!
        done
        ;;
    *)
        echo "$LOG_TAG Unknown mode: $MODE (expected: loop | once | wandb_archive)" >&2
        exit 2
        ;;
esac
