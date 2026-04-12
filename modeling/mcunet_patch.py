"""
Work around a bug in mcunet's download_url exception handler: on failure it does
``os.path.join(cached_file, 'download.lock')``, treating the JSON path as a directory,
which raises FileNotFoundError on Windows and masks the real download error.

Apply with :func:`apply_mcunet_download_patch` before importing ``mcunet.model_zoo``.
"""

from __future__ import annotations

import os
import sys

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


def _patched_download_url(
    url: str, model_dir: str = "~/.torch/mcunet", overwrite: bool = False
):
    target_name = url.split("/")[-1]
    base_dir = os.path.expanduser(model_dir)
    try:
        os.makedirs(base_dir, exist_ok=True)
        cached_file = os.path.join(base_dir, target_name)
        if not os.path.exists(cached_file) or overwrite:
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)
        return cached_file
    except Exception as e:
        lock_path = os.path.join(base_dir, "download.lock")
        if os.path.isfile(lock_path):
            try:
                os.remove(lock_path)
            except OSError:
                pass
        sys.stderr.write("Failed to download from url %s\n%s\n" % (url, e))
        return None


def apply_mcunet_download_patch() -> None:
    """
    Replace download_url everywhere mcunet keeps a reference (common_tools,
    utils, model_zoo), and point model_zoo at the current MIT release host.
    """
    import mcunet.utils.common_tools as ct

    ct.download_url = _patched_download_url

    import mcunet.utils as u

    if hasattr(u, "download_url"):
        u.download_url = _patched_download_url

    import mcunet.model_zoo as mz

    mz.download_url = _patched_download_url

    # Older pip builds use https://hanlab.mit.edu/... which now 404s; upstream
    # repo uses https://hanlab18.mit.edu/projects/tinyml/mcunet/release/
    if hasattr(mz, "url_base") and "hanlab18" not in getattr(mz, "url_base", ""):
        mz.url_base = "https://hanlab18.mit.edu/projects/tinyml/mcunet/release/"
