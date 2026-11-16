"""Independent Discrete-Event Simulator (DES) for quantized containerized
multi-worker service nodes with dynamic memory allocation and reclamation cycles.
"""

import numpy as np
from typing import Dict
from aoam_solver import WorkloadProfile


def simulate_node(
    wp: WorkloadProfile,
    u_cpu: float,
    m_wrk: int,
    m_limit_mb: float,
    sim_time: float = 300.0,
    warmup_frac: float = 0.1,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Simulate incoming Poisson requests, m_wrk parallel workers competing for periodic
    CPU quota Q = u_cpu * Tp, plus dynamic in-flight and transient GC memory tracking.
    """
    rng = np.random.default_rng(seed)
    tp = wp.tp
    q_budget = u_cpu * tp

    # Generate Poisson arrival timestamps
    inter_arrivals = rng.exponential(1.0 / wp.lam, size=int(wp.lam * sim_time * 1.25) + 100)
    arrivals = np.cumsum(inter_arrivals)
    arrivals = arrivals[arrivals < sim_time]
    n_arr = len(arrivals)

    # Generate service times according to squared CV c_s2
    if abs(wp.c_s2 - 1.0) < 1e-6:
        service_works = rng.exponential(1.0 / wp.mu, size=n_arr)
    else:
        shape = 1.0 / wp.c_s2
        scale = wp.c_s2 / wp.mu
        service_works = rng.gamma(shape, scale, size=n_arr)

    queue = []  # elements: [arrival_time, remaining_cpu_work]
    arr_idx = 0
    response_times = []
    throttled_periods = 0
    total_periods = int(sim_time / tp)

    l_samples = []
    mem_samples = []
    oom_events = 0

    # Periodic memory reclamation state
    next_gc_t = wp.tau_gc
    uncollected_garbage_mb = 0.0
    in_oom_cooldown = False

    for p_idx in range(total_periods):
        p_start = p_idx * tp
        p_end = (p_idx + 1) * tp
        quota_rem = q_budget
        curr_t = p_start
        was_throttled = False

        # Trigger garbage collection cycle when timer elapses
        if p_start >= next_gc_t:
            uncollected_garbage_mb *= 0.12  # residual survivor fraction after GC
            next_gc_t += rng.uniform(0.75 * wp.tau_gc, 1.25 * wp.tau_gc)

        while True:
            next_arr_t = arrivals[arr_idx] if arr_idx < n_arr and arrivals[arr_idx] < p_end else p_end
            k_active = min(len(queue), m_wrk)

            if k_active == 0 or quota_rem <= 1e-12:
                if k_active > 0 and quota_rem <= 1e-12 and curr_t < p_end - 1e-12:
                    was_throttled = True
                curr_t = next_arr_t
                if curr_t >= p_end - 1e-12:
                    break
                queue.append([arrivals[arr_idx], service_works[arr_idx]])
                uncollected_garbage_mb += wp.m_alloc_mb
                arr_idx += 1
                continue

            dt_quota = quota_rem / k_active
            dt_comp = min(queue[i][1] for i in range(k_active))
            dt_arr = next_arr_t - curr_t
            dt_end = p_end - curr_t
            dt = min(dt_quota, dt_comp, dt_arr, dt_end)

            if dt > 0.0:
                curr_t += dt
                quota_rem -= k_active * dt
                for i in range(k_active):
                    queue[i][1] -= dt

            # Process completed jobs
            comp_indices = [i for i in range(k_active) if queue[i][1] <= 1e-11]
            if comp_indices:
                for i in sorted(comp_indices, reverse=True):
                    arr_t, _ = queue.pop(i)
                    response_times.append(curr_t - arr_t)

            # Process arrival exactly at curr_t
            if arr_idx < n_arr and abs(curr_t - arrivals[arr_idx]) <= 1e-11 and arrivals[arr_idx] < p_end:
                queue.append([arrivals[arr_idx], service_works[arr_idx]])
                uncollected_garbage_mb += wp.m_alloc_mb
                arr_idx += 1

            # Check quota exhaustion
            if quota_rem <= 1e-11 and len(queue) > 0 and curr_t < p_end - 1e-11:
                was_throttled = True
                while arr_idx < n_arr and arrivals[arr_idx] < p_end:
                    queue.append([arrivals[arr_idx], service_works[arr_idx]])
                    uncollected_garbage_mb += wp.m_alloc_mb
                    arr_idx += 1
                break

            if curr_t >= p_end - 1e-11:
                break

        if p_idx >= int(warmup_frac * total_periods):
            if was_throttled:
                throttled_periods += 1
            n_sys = len(queue)
            l_samples.append(n_sys)

            # Instantaneous coupled memory consumption
            rho_est = min(0.92, wp.lam / max(1e-6, u_cpu * wp.mu))
            active_mb = wp.m_req_mb * n_sys
            gc_pressure_mb = wp.gc_alpha * (active_mb + uncollected_garbage_mb) / max(0.08, 1.0 - rho_est)
            total_mem_mb = wp.m_base_mb + m_wrk * wp.s_stack_mb + active_mb + uncollected_garbage_mb + gc_pressure_mb
            mem_samples.append(total_mem_mb)

            if total_mem_mb > m_limit_mb:
                if not in_oom_cooldown:
                    oom_events += 1
                    in_oom_cooldown = True
                    # Container restart resets active state and heap
                    queue.clear()
                    uncollected_garbage_mb = 0.0
            else:
                in_oom_cooldown = False

    n_warm = int(warmup_frac * len(response_times))
    rt_ms = np.array(response_times[n_warm:]) * 1000.0
    eval_periods = max(1, total_periods - int(warmup_frac * total_periods))
    mem_arr = np.array(mem_samples)
    l_arr = np.array(l_samples)

    return {
        "mean_R_ms": float(np.mean(rt_ms)) if len(rt_ms) else float("inf"),
        "p99_R_ms": float(np.percentile(rt_ms, 99)) if len(rt_ms) else float("inf"),
        "throttle_rate": throttled_periods / eval_periods,
        "mean_L": float(np.mean(l_arr)),
        "mean_mem_mb": float(np.mean(mem_arr)),
        "p99_mem_mb": float(np.percentile(mem_arr, 99)),
        "max_mem_mb": float(np.max(mem_arr)),
        "oom_events": oom_events,
    }
