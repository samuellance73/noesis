"""
utils/ssl_patch.py
──────────────────
Monkey-patches the global SSL context to bypass strict certificate
verification. Required in NixOS / restricted environments that lack
proper CA bundles.

Call apply() once, as early as possible (before any network imports).
"""

import ssl


def apply() -> None:
    """Disable SSL certificate verification globally."""
    _orig = ssl.create_default_context

    def _unverified_context(*args, **kwargs):
        ctx = _orig(*args, **kwargs)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    ssl.create_default_context = _unverified_context

    try:
        ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]
    except AttributeError:
        pass
