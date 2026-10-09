"""Install teacher force in spawned ranks; startup patch errors are fatal."""

import os
import sys
import traceback

if os.environ.get("GDN_FORCE_ACTIVE") == "1":
    try:
        if os.environ.get("GDN_FORCE_RUNNER") == "v2":
            import force_decode_v2

            force_decode_v2.install()
        else:
            import force_decode

            force_decode.install()
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        os._exit(97)
