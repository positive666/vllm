"""Enable GDN capture in spawned workers only when explicitly configured."""

import os
import sys
import traceback

if os.environ.get("GDN_CAPTURE_ACTIVE") == "1":
    try:
        import launch_capture

        launch_capture.install()
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        os._exit(97)
