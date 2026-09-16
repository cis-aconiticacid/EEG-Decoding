"""D078 uses the assigned physical-GPU0 queue partition and UUID mutex."""

import sys

from d057_common import ROOT

sys.path.insert(0, str(ROOT.parents[1] / "src"))
from eegdecoding import gpu_lock


if __name__ == "__main__":
    gpu_lock.OWNER_THREAD = "D-078"
    args = gpu_lock.build_parser().parse_args()
    args.queue_partition = "gpu0"
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    try:
        raise SystemExit(gpu_lock.run_locked(args))
    except gpu_lock.LockBusy as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(75)
