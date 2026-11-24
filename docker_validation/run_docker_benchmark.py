"""Controlled Linux cgroups v2 validation benchmark for AOAM using Docker.

Calibrates CPU service work to mu = 40.0 req/s per 1.0 vCPU inside an isolated
1-vCPU container, then evaluates unaligned vs. AOAM-aligned thread pools under
open-loop Poisson HTTP traffic and records kernel cpu.stat and memory.peak metrics.
"""

import csv
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(os.path.dirname(BASE_DIR), "results", "docker_cgroup_validation.csv")


def run_cmd(cmd, check=True):
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def parse_kv_block(text):
    out = {}
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            out[parts[0]] = int(parts[1])
    return out


def wait_container_ready(port=18080, timeout=15.0):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=1.0) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("Container did not become ready in time")


def calibrate_iters_per_vcpu(target_ms=25.0):
    """Measure exact PBKDF2 iterations yielding 25.0 ms mean (mu=40 req/s) on 1.0 vCPU."""
    cname = "aoam-calib"
    run_cmd(["docker", "rm", "-f", cname], check=False)
    probe_iters = 20000
    run_cmd([
        "docker", "run", "-d", "--name", cname,
        "--cpus=1.0", "--memory=512m", "-p", "18080:8080",
        "-e", "M_WRK=1", "-e", f"PBKDF_ITERS_MEAN={probe_iters}",
        "-e", "M_REQ_MB=0.5", "-e", "M_ALLOC_MB=0.2",
        "aoam-microbench:latest",
    ])
    try:
        wait_container_ready()
        # Run sequential deterministic calibration directly via docker exec
        res = run_cmd([
            "docker", "exec", cname, "python", "-c",
            f"import hashlib, time; t0=time.perf_counter(); "
            f"[hashlib.pbkdf2_hmac('sha256', b'a', b'b', {probe_iters}) for _ in range(40)]; "
            f"print((time.perf_counter()-t0)/40.0*1000.0)"
        ])
        measured_ms = float(res.stdout.strip())
        cal_iters = int(probe_iters * (target_ms / max(0.5, measured_ms)))
        print(f"Calibrated 1.0 vCPU speed: {probe_iters} iters = {measured_ms:.2f} ms -> target {target_ms} ms = {cal_iters} iters")
        return cal_iters
    finally:
        run_cmd(["docker", "rm", "-f", cname], check=False)


def run_open_loop_load(port, lam_rps, duration_s, seed=42):
    rng = random.Random(seed)
    latencies_ms = []
    errors = [0]
    lock = threading.Lock()
    stop_t = time.perf_counter() + duration_s

    def send_one():
        t_start = time.perf_counter()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/work", timeout=5.0) as resp:
                resp.read()
            dt = (time.perf_counter() - t_start) * 1000.0
            with lock:
                latencies_ms.append(dt)
        except Exception:
            with lock:
                errors[0] += 1

    threads = []
    next_t = time.perf_counter()
    while next_t < stop_t:
        next_t += rng.expovariate(lam_rps)
        wait = next_t - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        th = threading.Thread(target=send_one, daemon=True)
        th.start()
        threads.append(th)

    for th in threads:
        th.join(timeout=6.0)
    return latencies_ms, errors[0]


def benchmark_config(name, u_cpu, m_wrk, mem_limit_mb, lam_rps, cal_iters, duration_s=30.0, seed=101):
    cname = f"aoam-bench-{name}"
    run_cmd(["docker", "rm", "-f", cname], check=False)
    run_cmd([
        "docker", "run", "-d", "--name", cname,
        f"--cpus={u_cpu}", f"--memory={mem_limit_mb}m", f"--memory-swap={mem_limit_mb}m",
        "-p", "18080:8080",
        "-e", f"M_WRK={m_wrk}", "-e", f"PBKDF_ITERS_MEAN={cal_iters}",
        "aoam-microbench:latest",
    ])
    try:
        m_before = wait_container_ready()
        cpu_before = parse_kv_block(m_before["cpu.stat"])

        # Short 4s warm-up
        run_open_loop_load(18080, lam_rps, 4.0, seed=seed - 1)
        m_warm = json.loads(urllib.request.urlopen("http://127.0.0.1:18080/metrics", timeout=2.0).read().decode("utf-8"))
        cpu_start = parse_kv_block(m_warm["cpu.stat"])

        lats, errs = run_open_loop_load(18080, lam_rps, duration_s, seed=seed)

        insp = json.loads(run_cmd(["docker", "inspect", cname]).stdout)[0]
        oom_killed = bool(insp["State"].get("OOMKilled", False))
        exit_code = insp["State"].get("ExitCode", 0)

        p99_ms = float(np.percentile(lats, 99)) if lats else float("inf")
        mean_ms = float(np.mean(lats)) if lats else float("inf")

        if not oom_killed and insp["State"].get("Running", False):
            m_end = json.loads(urllib.request.urlopen("http://127.0.0.1:18080/metrics", timeout=2.0).read().decode("utf-8"))
            cpu_end = parse_kv_block(m_end["cpu.stat"])
            d_periods = max(1, cpu_end.get("nr_periods", 0) - cpu_start.get("nr_periods", 0))
            d_throttled = max(0, cpu_end.get("nr_throttled", 0) - cpu_start.get("nr_throttled", 0))
            # cgroups v1 exposes throttled_time (ns); cgroups v2 exposes throttled_usec (us)
            if "throttled_time" in cpu_end:
                d_thr_ms = max(0, cpu_end.get("throttled_time", 0) - cpu_start.get("throttled_time", 0)) / 1e6
            else:
                d_thr_ms = max(0, cpu_end.get("throttled_usec", 0) - cpu_start.get("throttled_usec", 0)) / 1000.0
            mem_peak_mb = int(m_end.get("memory.peak") or 0) / (1024 * 1024)
            mem_ev = parse_kv_block(m_end.get("memory.events", ""))
        else:
            d_periods, d_throttled, d_thr_ms = 0, 0, 0.0
            mem_peak_mb = float(mem_limit_mb)
            mem_ev = {"oom_kill": 1}

        thr_pct = (d_throttled / d_periods * 100.0) if d_periods > 0 else 0.0
        row = {
            "config": name,
            "u_cpu": u_cpu,
            "m_wrk": m_wrk,
            "mem_limit_mb": mem_limit_mb,
            "lambda_rps": lam_rps,
            "completed_reqs": len(lats),
            "errors": errs,
            "mean_ms": round(mean_ms, 2),
            "p99_ms": round(p99_ms, 2),
            "nr_periods": d_periods,
            "nr_throttled": d_throttled,
            "throttle_pct": round(thr_pct, 2),
            "throttled_ms": round(d_thr_ms, 1),
            "mem_peak_mb": round(mem_peak_mb, 1),
            "oom_killed": oom_killed or (mem_ev.get("oom_kill", 0) > 0) or (exit_code == 137),
        }
        print("Result:", row)
        return row
    finally:
        run_cmd(["docker", "rm", "-f", cname], check=False)


def main():
    print("Building aoam-microbench Docker image...")
    run_cmd(["docker", "build", "-t", "aoam-microbench:latest", BASE_DIR])
    cal_iters = calibrate_iters_per_vcpu(target_ms=25.0)

    configs = [
        ("Static_Unaligned_8", 2, 8, 1024, 55.0, 101),
        ("Oversized_Pool_16", 2, 16, 1024, 55.0, 102),
        ("AOAM_Aligned_2", 2, 2, 512, 55.0, 103),
        ("AOAM_Aligned_3_High", 3, 3, 768, 85.0, 104),
        ("Unaligned_12_High", 3, 12, 768, 85.0, 105),
    ]
    rows = []
    for name, u, m, mem, lam, sd in configs:
        rows.append(benchmark_config(name, u, m, mem, lam, cal_iters, duration_s=30.0, seed=sd))

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved live Docker cgroups results to:", RESULTS_CSV)


if __name__ == "__main__":
    main()
