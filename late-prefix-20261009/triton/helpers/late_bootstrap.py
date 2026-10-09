"""Install independent sampler observation and exact runtime decode capture."""
from __future__ import annotations
import os

def install():
    if os.environ.get("GDN_LATE_ACTIVE") != "1":
        return
    import force_decode as common
    import force_decode_v2
    import launch_capture
    import late_capture
    capture_active = os.environ.pop("GDN_CAPTURE_ACTIVE", None)
    try:
        force_decode_v2.install()
    finally:
        if capture_active is not None:
            os.environ["GDN_CAPTURE_ACTIVE"] = capture_active
    if common._PATCHED:
        raise RuntimeError("Late bootstrap must precede actual runner import")
    original = common._patch
    def combined(module):
        original(module)
        late_capture.patch_runner(module)
    common._patch = combined
    launch_capture._choose_capture = late_capture.choose
    launch_capture.install()
