"""Run as the first project-related cell in the v2 Kaggle worker notebook.

In a notebook, use ``%run /kaggle/working/analogical_math_rag/kaggle_resume_v2_bootstrap.py``
after cloning the current repository, before importing ``config`` or ``src``.
The subprocess authenticates the saved manifest and creates an isolated checkout.
"""

from pathlib import Path
import os
import subprocess
import sys


RUN_ID = "layer1-grouping-k5-size2-v2"
ACTIVE_SOURCE = Path("/kaggle/working/analogical_math_rag")
RESUME_SOURCE = Path("/kaggle/working/analogical_math_rag-resume-v2")


def _hub_token():
    token = os.environ.get("HF_SYNC_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return None
    try:
        return UserSecretsClient().get_secret("HF_SYNC_TOKEN")
    except Exception:
        # A public repository or a cached Hugging Face login may still work.
        return None


def prepare_notebook_resume():
    imported = sorted(
        name for name in sys.modules
        if name == "config" or name == "src" or name.startswith("src.")
    )
    if imported:
        raise RuntimeError(
            "Project modules are already imported. Restart the kernel and run "
            "this bootstrap before the setup cell."
        )
    if not (ACTIVE_SOURCE / "prepare_distributed_resume.py").is_file():
        raise RuntimeError(f"Recovery helper is missing from {ACTIVE_SOURCE}.")

    from huggingface_hub import hf_hub_download

    manifest_path = hf_hub_download(
        repo_id="mostafabehroozi/grouping_dist",
        repo_type="dataset",
        filename=f"distributed_runs/{RUN_ID}/manifest.json",
        revision="main",
        token=_hub_token(),
    )
    subprocess.run(
        [
            sys.executable, "-B", str(ACTIVE_SOURCE / "prepare_distributed_resume.py"),
            "--manifest", str(manifest_path),
            "--run-id", RUN_ID,
            "--destination", str(RESUME_SOURCE),
        ],
        cwd=ACTIVE_SOURCE,
        check=True,
    )

    # The notebook and later project imports must use the authenticated source.
    os.chdir(RESUME_SOURCE)
    active_source = ACTIVE_SOURCE.resolve()
    sys.path[:] = [
        str(RESUME_SOURCE),
        *(entry for entry in sys.path if not entry or Path(entry).resolve() != active_source),
    ]
    print(f"Verified {RUN_ID} source: {RESUME_SOURCE}")
    return RESUME_SOURCE


if __name__ == "__main__":
    prepare_notebook_resume()
