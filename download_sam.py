"""
Download the Two-hot/SAM-SGT dataset from HuggingFace Hub.

Usage:
    python download_sam_sgt.py                              # default dir: ./data/SAM-SGT
    python download_sam_sgt.py --target-dir /path/to/save   # custom dir
    python download_sam_sgt.py --target-dir ./my_data --use-mirror
"""

import os
import argparse
from huggingface_hub import snapshot_download


def download_dataset(
    repo_id: str,
    target_dir: str,
    use_mirror: bool = False,
    disable_xet: bool = True,
) -> str:
    """
    Download a dataset from HuggingFace Hub.

    Args:
        repo_id: The repository ID on HuggingFace (e.g., "Two-hot/SAM-SGT").
        target_dir: Local directory to save the dataset.
        use_mirror: If True, use hf-mirror.com (useful for users in mainland China).
        disable_xet: If True, disable hf_xet backend to avoid permission errors.

    Returns:
        The absolute path of the downloaded dataset.
    """
    # Disable hf_xet to avoid permission errors in restricted environments
    if disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    # Optionally use HF-Mirror
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    os.makedirs(target_dir, exist_ok=True)

    print(f"📦 Repo ID     : {repo_id}")
    print(f"📁 Target Dir  : {os.path.abspath(target_dir)}")
    print(f"🌐 Endpoint    : {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
    print(f"🚫 Disable Xet : {disable_xet}")
    print("-" * 60)

    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=target_dir,
        resume_download=True,
    )

    print("-" * 60)
    print(f"✅ Download completed: {local_path}")
    return local_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Two-hot/SAM-SGT dataset from HuggingFace."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Two-hot/SAM-SGT",
        help="HuggingFace repo ID (default: Two-hot/SAM-SGT).",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="./data/SAM-SGT",
        help="Directory to save the dataset (default: ./data/SAM-SGT).",
    )
    parser.add_argument(
        "--use-mirror",
        action="store_true",
        help="Use hf-mirror.com endpoint (for users in mainland China).",
    )
    parser.add_argument(
        "--enable-xet",
        action="store_true",
        help="Enable hf_xet backend (disabled by default to avoid permission errors).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    download_dataset(
        repo_id=args.repo_id,
        target_dir=args.target_dir,
        use_mirror=args.use_mirror,
        disable_xet=not args.enable_xet,
    )


if __name__ == "__main__":
    main()