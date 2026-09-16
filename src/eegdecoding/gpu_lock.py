"""Atomic, non-stealing GPU lock and single-job queue for shared hosts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


OWNER_THREAD = os.environ.get("EEG_LOCK_OWNER", "local-process")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LockBusy(RuntimeError):
    pass


class AtomicDirectoryLock:
    """Acquire via atomic mkdir; never infer recoverability from mtime."""

    def __init__(self, path: Path, metadata: dict[str, object]):
        self.path = path
        self.metadata = metadata
        self.acquired = False

    @property
    def metadata_path(self) -> Path:
        return self.path / "owner.json"

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(self.path)
        except FileExistsError as exc:
            detail = ""
            try:
                detail = self.metadata_path.read_text(encoding="utf-8")
            except OSError:
                pass
            raise LockBusy(f"lock busy: {self.path}; owner={detail}") from exc
        self.acquired = True
        self.write_metadata()

    def write_metadata(self) -> None:
        if not self.acquired:
            raise RuntimeError("cannot write metadata for an unheld lock")
        temp = self.path / f"owner.{os.getpid()}.tmp"
        temp.write_text(json.dumps(self.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, self.metadata_path)

    def heartbeat(self) -> None:
        self.metadata["heartbeat_utc"] = utc_now()
        self.write_metadata()

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            for child in self.path.iterdir():
                if child.is_file():
                    child.unlink()
            self.path.rmdir()
        finally:
            self.acquired = False


def gpu_compute_processes(gpu_uuid: str) -> list[dict[str, str]]:
    """Return live compute processes for a UUID; failure is a hard error."""

    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) == 4 and fields[0] == gpu_uuid:
            rows.append(dict(zip(("gpu_uuid", "pid", "process_name", "used_memory_mib"), fields)))
    return rows


def local_wddm_snapshot(gpu_uuid: str) -> dict[str, object]:
    """Audit a Windows display GPU without treating C+G UI clients as pure compute."""

    if os.name != "nt":
        raise RuntimeError("local_wddm_cg_v1 is Windows-only")
    inventory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gpu_index = None
    memory_used = None
    utilization = None
    for line in inventory.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4 and fields[1] == gpu_uuid:
            gpu_index, memory_used, utilization = int(fields[0]), int(fields[2]), int(fields[3])
            break
    if gpu_index is None:
        raise RuntimeError(f"GPU UUID not found: {gpu_uuid}")
    full = subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True).stdout
    pattern = re.compile(
        r"^\|\s*(\d+)\s+\S+\s+\S+\s+(\d+)\s+([CG](?:\+G)?)\s+(.+?)\s+(?:N/A|\d+MiB)\s*\|\s*$"
    )
    processes = []
    for line in full.splitlines():
        match = pattern.match(line)
        if match and int(match.group(1)) == gpu_index:
            processes.append(
                {"pid": int(match.group(2)), "type": match.group(3), "process_name": match.group(4).strip()}
            )
    eligible = (
        memory_used <= 512
        and utilization <= 5
        and all(process["type"] == "C+G" for process in processes)
    )
    return {
        "timestamp_utc": utc_now(),
        "gpu_uuid": gpu_uuid,
        "gpu_index": gpu_index,
        "memory_used_mib": memory_used,
        "utilization_percent": utilization,
        "existing_processes": processes,
        "eligible": eligible,
        "rule": "local_wddm_cg_v1: only C+G; memory<=512MiB; util<=5%",
    }


def append_audit(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_locked(args: argparse.Namespace, process_query=None) -> int:
    if not args.command:
        raise ValueError("a worker command is required after --")
    lock_root = Path(args.lock_root).resolve()
    partition = getattr(args, "queue_partition", "global")
    if partition not in ("global", "gpu0"):
        raise ValueError(f"unsupported queue partition: {partition}")
    if partition == "gpu0":
        expected_uuid = args.gpu_uuid
        if not expected_uuid or args.eligibility_policy != "strict_compute_apps":
            raise ValueError("gpu0 partition requires --gpu-uuid and strict compute eligibility")
        inventory = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        ).stdout
        mapping = [tuple(field.strip() for field in line.split(",")) for line in inventory.splitlines()]
        if ("0", expected_uuid) not in mapping:
            raise RuntimeError("the requested GPU UUID does not match physical index 0 on this host")
    eligibility_snapshots: list[dict[str, object]] = []
    if args.eligibility_policy == "local_wddm_cg_v1":
        if args.run_kind != "diagnostic_local":
            raise ValueError("local_wddm_cg_v1 is restricted to diagnostic_local runs")
        for index in range(3):
            snapshot = local_wddm_snapshot(args.gpu_uuid)
            eligibility_snapshots.append(snapshot)
            if not snapshot["eligible"]:
                raise LockBusy(f"local WDDM diagnostic eligibility failed: {snapshot}")
            if index < 2:
                time.sleep(2.0)
    common = {
        "owner_thread": OWNER_THREAD,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "gpu_uuid": args.gpu_uuid,
        "queue_partition": partition,
        "run_id": args.run_id,
        "holder_pid": os.getpid(),
        "holder_started_utc": utc_now(),
        "holder_command": sys.argv,
        "worker_command": args.command,
        "heartbeat_utc": utc_now(),
        "run_kind": args.run_kind,
        "eligibility_policy": args.eligibility_policy,
        "eligibility_prelock_snapshots": eligibility_snapshots,
    }
    queue_name = "queue-slot-0.lock" if partition == "global" else "queue-partition-gpu0.lock"
    queue_kind = "global_queue" if partition == "global" else "partition_queue"
    queue_lock = AtomicDirectoryLock(lock_root / queue_name, dict(common, lock_kind=queue_kind))
    gpu_lock = AtomicDirectoryLock(
        lock_root / f"gpu-{args.gpu_uuid}.lock", dict(common, lock_kind="gpu_uuid")
    )
    queue_lock.acquire()
    try:
        gpu_lock.acquire()
        try:
            if args.eligibility_policy == "local_wddm_cg_v1":
                inside_snapshot = local_wddm_snapshot(args.gpu_uuid)
                if not inside_snapshot["eligible"]:
                    raise LockBusy(f"local WDDM eligibility changed inside lock: {inside_snapshot}")
                for lock in (queue_lock, gpu_lock):
                    lock.metadata["eligibility_inside_lock_snapshot"] = inside_snapshot
                    lock.write_metadata()
            else:
                query = gpu_compute_processes if process_query is None else process_query
                occupied = query(args.gpu_uuid)
                if occupied:
                    raise LockBusy(f"GPU {args.gpu_uuid} has live compute processes: {occupied}")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
            worker = subprocess.Popen(args.command, env=env, start_new_session=(os.name == "posix"))
            started = utc_now()
            for lock in (queue_lock, gpu_lock):
                lock.metadata.update(worker_pid=worker.pid, worker_started_utc=started)
                lock.write_metadata()

            stop = threading.Event()

            def beat() -> None:
                while not stop.wait(args.heartbeat_seconds):
                    for lock in (queue_lock, gpu_lock):
                        lock.heartbeat()

            thread = threading.Thread(target=beat, name="gpu-lock-heartbeat", daemon=True)
            thread.start()
            previous_handlers: dict[int, object] = {}

            def forward(signum, _frame) -> None:
                """Keep holding the lock while forwarding termination to the worker group."""

                if worker.poll() is not None:
                    return
                os.killpg(worker.pid, signum)

            if os.name == "posix":
                for signame in ("SIGINT", "SIGTERM", "SIGHUP"):
                    signum = getattr(signal, signame, None)
                    if signum is not None:
                        previous_handlers[signum] = signal.getsignal(signum)
                        signal.signal(signum, forward)
            try:
                exit_code = worker.wait()
            except KeyboardInterrupt:
                if os.name == "posix":
                    os.killpg(worker.pid, signal.SIGINT)
                else:
                    worker.terminate()
                exit_code = worker.wait()
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
                stop.set()
                thread.join(timeout=max(1.0, args.heartbeat_seconds * 2))
            append_audit(
                lock_root / "audit.jsonl",
                {
                    **queue_lock.metadata,
                    "event": "worker_exit",
                    "worker_exit_code": exit_code,
                    "released_utc": utc_now(),
                    "gpu_lock_metadata": gpu_lock.metadata,
                },
            )
            return exit_code
        finally:
            gpu_lock.release()
    finally:
        queue_lock.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-root", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--queue-partition", choices=("global", "gpu0"), default="global")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-kind", choices=("standard", "diagnostic_local"), default="standard")
    parser.add_argument(
        "--eligibility-policy", choices=("strict_compute_apps", "local_wddm_cg_v1"), default="strict_compute_apps"
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    try:
        return run_locked(args)
    except LockBusy as exc:
        print(str(exc), file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
