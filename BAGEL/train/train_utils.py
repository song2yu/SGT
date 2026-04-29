# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import logging
import os


# Storage-compatibility notes:
#   Some filesystems -- typically object-store-backed FUSE mounts -- do NOT
#   support append or rename. Python's ``logging.FileHandler`` opens files
#   with mode 'a' (append) by default, which triggers
#   ``PermissionError: [Errno 1] Operation not permitted`` on such mounts.
#
#   To work around that:
#     1) if ``logging_dir`` points at an object-store-backed mount, the
#        logger transparently redirects the log file into ``LOCAL_LOG_ROOT``
#        (default: ``./bagel_logs``) -- a local directory -- and relies on
#        an external rsync script to archive the logs later;
#     2) we also touch the log file explicitly in mode ``'a'`` so that the
#        ``FileHandler`` does not fail when the file does not exist yet.
#
#   Environment overrides:
#     BAGEL_LOCAL_LOG_ROOT -- local log root directory, default ``./bagel_logs``
#     BAGEL_OBJECT_STORE_PREFIXES -- colon-separated list of path prefixes
#         that should be treated as object-store mounts. If unset, no paths
#         are treated specially and ``logging_dir`` is used as-is.
_LOCAL_LOG_ROOT_DEFAULT = "./bagel_logs"


def _object_store_prefixes():
    raw = os.environ.get("BAGEL_OBJECT_STORE_PREFIXES", "")
    return tuple(p for p in raw.split(":") if p)


def _resolve_safe_log_dir(logging_dir: str) -> str:
    """If ``logging_dir`` lives on an object-store mount, rewrite it to a local dir."""
    abs_dir = os.path.abspath(logging_dir)
    prefixes = _object_store_prefixes()
    if not prefixes:
        return abs_dir
    on_object_store = any(abs_dir.startswith(p) or abs_dir == p.rstrip("/")
                          for p in prefixes)
    if not on_object_store:
        return abs_dir

    local_root = os.environ.get("BAGEL_LOCAL_LOG_ROOT", _LOCAL_LOG_ROOT_DEFAULT)
    # Use the "de-rooted" original path as sub-directory so that runs from
    # different experiments do not overwrite each other.
    suffix = abs_dir.lstrip("/").replace("/", "_")
    redirected = os.path.join(local_root, suffix)
    print(f"[create_logger] logging_dir={abs_dir} is on an object-store mount "
          f"that does not support append; redirecting to local disk at "
          f"{redirected} (use scripts/sync_runs.sh to archive it later).")
    return redirected


def create_logger(logging_dir, rank, filename="log"):
    """Create a logger that writes to both stdout and a log file.

    When ``logging_dir`` lives on an object-store-backed mount that does not
    support append, the log file is redirected to a local directory (see
    ``_resolve_safe_log_dir`` above).
    """
    if rank == 0 and logging_dir is not None:  # real logger
        safe_dir = _resolve_safe_log_dir(logging_dir)
        os.makedirs(safe_dir, exist_ok=True)
        log_path = os.path.join(safe_dir, f"{filename}.txt")

        # Touch the file beforehand. If the file does not yet exist, the
        # FileHandler will open it with O_APPEND, which fails with EPERM on
        # some FUSE mounts. Pre-creating the file avoids that.
        try:
            with open(log_path, "a"):
                pass
        except PermissionError as e:
            # Extreme fallback: if ``safe_dir`` itself is not writable, fall
            # back to /tmp so training does not crash on startup.
            fallback = os.path.join("/tmp", f"bagel_{filename}.txt")
            print(f"[create_logger] log dir {safe_dir} does not support "
                  f"append ({e}); falling back to {fallback}")
            log_path = fallback
            with open(log_path, "a"):
                pass

        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_path, mode="a"),
            ],
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Log file: {log_path}")
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def get_latest_ckpt(checkpoint_dir):
    step_dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if len(step_dirs) == 0:
        return None
    step_dirs = sorted(step_dirs, key=lambda x: int(x))
    latest_step_dir = os.path.join(checkpoint_dir, step_dirs[-1])
    return latest_step_dir
