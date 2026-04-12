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
    import mcunet.utils.common_tools as ct

    ct.download_url = _patched_download_url
