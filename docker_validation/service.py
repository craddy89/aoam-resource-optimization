"""Compute-bound multi-threaded microservice inside Docker for Linux cgroups validation.

Uses hashlib.pbkdf2_hmac (which releases the CPython GIL in OpenSSL) so worker
threads execute in true OS-level parallel across assigned CPU quotas and trigger
real Linux CFS bandwidth throttling when m_wrk > allocated --cpus.
"""

import collections
import concurrent.futures
import gc
import hashlib
import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


M_WRK = int(os.environ.get("M_WRK", "2"))
PBKDF_ITERS_MEAN = int(os.environ.get("PBKDF_ITERS_MEAN", "25000"))
M_REQ_MB = float(os.environ.get("M_REQ_MB", "4.0"))
M_ALLOC_MB = float(os.environ.get("M_ALLOC_MB", "1.2"))
TAU_GC = float(os.environ.get("TAU_GC", "1.8"))

EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=M_WRK)
BASE_BUF = bytearray(64 * 1024 * 1024)  # 64 MB static runtime baseline buffer
SURVIVOR_RING = collections.deque(maxlen=24)
TRANSIENT_GARBAGE = []
GARBAGE_LOCK = threading.Lock()


def read_first_existing(paths):
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            continue
    return ""


def read_cgroup_stats():
    stats = {}
    stats["cpu.stat"] = read_first_existing([
        "/sys/fs/cgroup/cpu/cpu.stat",
        "/sys/fs/cgroup/cpu,cpuacct/cpu.stat",
        "/sys/fs/cgroup/cpu.stat",
    ])
    stats["cpu.cfs_quota_us"] = read_first_existing([
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu.max",
    ])
    stats["memory.current"] = read_first_existing([
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        "/sys/fs/cgroup/memory.current",
    ])
    stats["memory.peak"] = read_first_existing([
        "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",
        "/sys/fs/cgroup/memory.peak",
    ])
    stats["memory.events"] = read_first_existing([
        "/sys/fs/cgroup/memory/memory.oom_control",
        "/sys/fs/cgroup/memory.events",
    ])
    return stats


def gc_worker():
    """Periodic heap reclamation loop modeling sawtooth GC dynamics."""
    while True:
        time.sleep(random.uniform(0.75 * TAU_GC, 1.25 * TAU_GC))
        with GARBAGE_LOCK:
            if TRANSIENT_GARBAGE:
                # Retain ~12% survivor fraction in ring, drop remaining transient buffers
                keep_n = max(1, int(0.12 * len(TRANSIENT_GARBAGE)))
                for buf in TRANSIENT_GARBAGE[:keep_n]:
                    SURVIVOR_RING.append(buf[: max(1024, len(buf) // 4)])
                TRANSIENT_GARBAGE.clear()
        gc.collect()


def execute_request(iters_scale: float):
    iters = max(500, int(PBKDF_ITERS_MEAN * iters_scale))
    # Active in-flight request buffer touched so Linux RSS pages are committed
    req_bytes = max(4096, int(M_REQ_MB * 1024 * 1024))
    active_buf = bytearray(req_bytes)
    active_buf[0] = 1
    active_buf[-1] = 1
    for off in range(0, req_bytes, 4096):
        active_buf[off] = 42

    # True parallel CPU work releasing GIL inside OpenSSL C extension
    digest = hashlib.pbkdf2_hmac("sha256", b"aoam-workload", b"cgroup-salt", iters)

    alloc_bytes = max(4096, int(M_ALLOC_MB * 1024 * 1024))
    g_buf = bytearray(alloc_bytes)
    for off in range(0, alloc_bytes, 4096):
        g_buf[off] = digest[0]
    with GARBAGE_LOCK:
        TRANSIENT_GARBAGE.append(g_buf)

    del active_buf
    return len(digest)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path.startswith("/work"):
            scale = random.expovariate(1.0)
            fut = EXECUTOR.submit(execute_request, scale)
            res = fut.result()
            body = f'{{"ok":true,"bytes":{res}}}'.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/metrics"):
            body = json.dumps(read_cgroup_stats()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


if __name__ == "__main__":
    t_gc = threading.Thread(target=gc_worker, daemon=True)
    t_gc.start()
    server = ThreadingHTTPServer(("0.0.0.0", 8080), Handler)
    server.serve_forever()
