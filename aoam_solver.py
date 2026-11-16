"""Analytical stochastic model and AOAM optimization solver for
two-resource quantized containerized service nodes.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.special import gammaincc, gammaincinv


@dataclass(frozen=True)
class WorkloadProfile:
    """Stochastic parameters of incoming request stream and resource usage."""
    lam: float            # Arrival rate lambda (req/s)
    mu: float             # Service rate mu per 1.0 vCPU (req/s)
    c_s2: float = 1.0     # Squared coefficient of variation of service time (1.0 = M/M/c)
    tp: float = 0.1       # Scheduler quota replenishment period Tp (s)
    m_base_mb: float = 320.0   # Base resident memory footprint (MB)
    s_stack_mb: float = 1.0    # Per-thread stack memory (MB)
    m_req_mb: float = 8.5      # Active memory per in-flight request (MB)
    m_alloc_mb: float = 2.4    # Transient uncollected allocation per req before GC (MB)
    tau_gc: float = 1.8        # Mean memory reclamation interval (s)
    gc_alpha: float = 0.18     # Nonlinear heap fragmentation / GC pressure coefficient


@dataclass(frozen=True)
class SLAConstraints:
    """Service-level constraints and unit resource costs."""
    t_slo_ms: float = 200.0    # Target 99th percentile response time (ms)
    eps_thr: float = 0.02      # Max allowed period throttling probability
    eps_oom: float = 0.001     # Max allowed memory overflow risk (OOM probability)
    w_cpu: float = 18.0        # Monthly cost weight per 1.0 vCPU ($/vCPU/mo)
    w_mem_gb: float = 6.0      # Monthly cost weight per 1.0 GB RAM ($/GB/mo)


def erlang_c(c: int, a: float) -> float:
    """Compute exact Erlang-C delay probability P(W > 0) for M/M/c."""
    if c <= 0 or a < 0:
        raise ValueError("c must be positive integer and a >= 0")
    rho = a / c
    if rho >= 1.0:
        return 1.0
    if a == 0.0:
        return 0.0
    sum_terms = 1.0
    term = 1.0
    for k in range(1, c):
        term *= a / k
        sum_terms += term
    last_term = term * (a / c) / (1.0 - rho)
    return last_term / (sum_terms + last_term)


def mmc_queue_moments(lam: float, mu: float, c: int, c_s2: float = 1.0) -> Dict[str, float]:
    """
    Compute stationary moments for M/M/c (exact) and M/G/c (Allen-Cunneen two-moment).
    Returns rho, C, mean_W, std_W, mean_R, std_R, mean_L, std_L.
    """
    a = lam / mu
    rho = a / c
    if rho >= 1.0:
        return {
            "rho": rho, "C": 1.0,
            "mean_W": float("inf"), "std_W": float("inf"),
            "mean_R": float("inf"), "std_R": float("inf"),
            "mean_L": float("inf"), "std_L": float("inf"),
        }

    C = erlang_c(c, a)
    # Exact M/M/c waiting time moments
    gamma = c * mu - lam
    mean_w_mmc = C / gamma
    var_w_mmc = (2.0 * C - C * C) / (gamma * gamma)

    # Allen-Cunneen adjustment for general service time variance c_s2
    scale_ac = 0.5 * (1.0 + c_s2)
    mean_w = scale_ac * mean_w_mmc
    # Two-moment variance scaling for waiting time under M/G/c
    var_w = (scale_ac ** 2) * var_w_mmc if c_s2 >= 1.0 else scale_ac * var_w_mmc

    mean_s = 1.0 / mu
    var_s = c_s2 * (mean_s ** 2)

    mean_r = mean_w + mean_s
    var_r = var_w + var_s
    mean_l = lam * mean_r

    # Variance of number in system L (via distributional Little's law approximation)
    var_l = lam * mean_r + (lam ** 2) * var_w * 0.5
    return {
        "rho": rho,
        "C": C,
        "mean_W": mean_w,
        "std_W": math.sqrt(max(0.0, var_w)),
        "mean_R": mean_r,
        "std_R": math.sqrt(max(0.0, var_r)),
        "mean_L": mean_l,
        "std_L": math.sqrt(max(0.0, var_l)),
    }


def mmc_exact_survival(t: float, lam: float, mu: float, c: int) -> float:
    """Exact response time survival probability P(R > t) for M/M/c."""
    if t <= 0.0:
        return 1.0
    a = lam / mu
    if a / c >= 1.0:
        return 1.0
    C = erlang_c(c, a)
    gamma = c * mu - lam
    exp_mu = math.exp(-mu * t)
    if abs(gamma - mu) < 1e-9:
        p_ws_gt_t = (1.0 + mu * t) * exp_mu
    else:
        exp_gamma = math.exp(-gamma * t)
        p_ws_gt_t = (gamma * exp_mu - mu * exp_gamma) / (gamma - mu)
    return (1.0 - C) * exp_mu + C * p_ws_gt_t


def response_time_quantile(q: float, lam: float, mu: float, c: int, c_s2: float = 1.0) -> float:
    """
    Compute q-th quantile of response time R (in seconds).
    Uses exact survival curve bisection for M/M/c (c_s2 == 1.0),
    and Gamma moment-matching over exact M/G/c two-moment approximations otherwise.
    """
    if lam / (c * mu) >= 1.0:
        return float("inf")

    if abs(c_s2 - 1.0) < 1e-6:
        target_tail = 1.0 - q
        lo, hi = 0.0, 5.0
        while mmc_exact_survival(hi, lam, mu, c) > target_tail:
            hi *= 2.0
            if hi > 1e5:
                return float("inf")
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if mmc_exact_survival(mid, lam, mu, c) > target_tail:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    moments = mmc_queue_moments(lam, mu, c, c_s2)
    mean_r = moments["mean_R"]
    var_r = moments["std_R"] ** 2
    if var_r <= 1e-15:
        return mean_r
    shape = (mean_r ** 2) / var_r
    scale = var_r / mean_r
    return float(gammaincinv(shape, q) * scale)


def compound_poisson_throttle_prob(
    lam: float, mu: float, u_cpu: float, m_wrk: int, tp: float, c_s2: float = 1.0
) -> float:
    """
    Compute periodic CPU quota exhaustion probability P_thr.
    If worker pool m_wrk <= floor(u_cpu) and u_cpu is integer, throttling is identically 0.
    Otherwise, models aggregate demand in period Tp as a Compound Poisson process.
    """
    if m_wrk <= int(math.floor(u_cpu + 1e-9)) and abs(u_cpu - round(u_cpu)) < 1e-9:
        return 0.0

    q_budget = u_cpu * tp
    c_eff = max(1, min(m_wrk, int(math.ceil(u_cpu))))
    rho = lam / (u_cpu * mu)
    if rho >= 1.0:
        return 1.0

    # Mean carryover backlog into period boundary
    a = lam / mu
    c_int = max(1, int(round(u_cpu)))
    C = erlang_c(c_int, min(a, c_int * 0.9999))
    backlog_jobs = C * rho / (1.0 - rho)
    eff_n = lam * tp + 0.5 * backlog_jobs

    p_exceed = 0.0
    max_n = int(max(45, eff_n + 9.0 * math.sqrt(eff_n + 1.0)))
    pois_p = math.exp(-eff_n)
    for n in range(1, max_n + 1):
        pois_p *= eff_n / n
        if pois_p < 1e-12 and n > eff_n:
            break
        shape = n / c_s2
        scale = c_s2 / mu
        p_exceed += pois_p * float(gammaincc(shape, q_budget / scale))
    return min(1.0, p_exceed)


def coupled_memory_requirement(
    wp: WorkloadProfile, u_cpu: float, m_wrk: int, eps_oom: float
) -> Dict[str, float]:
    """
    Compute mean, standard deviation, and OOM-safe limit memory (MB)
    via Little's law coupling L(lambda, u_cpu), sawtooth memory reclamation,
    and nonlinear GC pressure near saturation.
    """
    c_int = max(1, int(math.floor(u_cpu)))
    c_eff = min(m_wrk, c_int) if m_wrk < u_cpu else max(1, int(round(u_cpu)))
    qm = mmc_queue_moments(wp.lam, wp.mu, c_eff, wp.c_s2)

    if math.isinf(qm["mean_L"]):
        return {"mean_mb": float("inf"), "std_mb": float("inf"), "limit_mb": float("inf")}

    rho = qm["rho"]
    f_surv = 0.12  # residual survivor fraction after reclamation
    gc_pressure_factor = wp.gc_alpha / max(0.08, 1.0 - rho)

    # 1) Stationary Mean Memory
    mean_active_mb = wp.m_req_mb * qm["mean_L"]
    mean_garbage_mb = ((1.0 + f_surv) / (2.0 * (1.0 - f_surv))) * wp.lam * wp.m_alloc_mb * wp.tau_gc
    mean_pressure_mb = gc_pressure_factor * (mean_active_mb + mean_garbage_mb)
    mean_total_mb = wp.m_base_mb + m_wrk * wp.s_stack_mb + mean_active_mb + mean_garbage_mb + mean_pressure_mb

    # 2) Pre-Reclamation Tail Limit at OOM risk eps_oom
    # Tail quantile multiplier using normal inverse CDF via scipy/erfinv approximation
    from scipy.stats import norm
    z_tail = float(norm.ppf(1.0 - eps_oom))

    # Tail queue occupancy: account for right-skewed M/M/c geometric tail via Gamma quantile
    var_l = max(1e-6, qm["std_L"] ** 2)
    shape_l = (qm["mean_L"] ** 2) / var_l
    scale_l = var_l / max(1e-6, qm["mean_L"])
    n_sys_tail = float(gammaincinv(shape_l, 1.0 - eps_oom) * scale_l)
    active_tail_mb = wp.m_req_mb * n_sys_tail

    # Worst-case pre-reclamation garbage peak (1.25x interval jitter + Poisson burst)
    tau_max = 1.25 * wp.tau_gc
    lam_tau_max = wp.lam * tau_max
    garbage_peak_mb = (
        wp.m_alloc_mb * (lam_tau_max + z_tail * math.sqrt(max(1e-6, lam_tau_max)))
    ) / (1.0 - f_surv)

    tail_pressure_mb = gc_pressure_factor * (active_tail_mb + garbage_peak_mb)
    limit_mb = wp.m_base_mb + m_wrk * wp.s_stack_mb + active_tail_mb + garbage_peak_mb + tail_pressure_mb

    return {
        "rho": rho,
        "mean_L": qm["mean_L"],
        "mean_mb": mean_total_mb,
        "std_mb": math.sqrt((wp.m_req_mb * qm["std_L"]) ** 2 + (0.35 * garbage_peak_mb) ** 2),
        "limit_mb": limit_mb,
    }


def evaluate_configuration(
    wp: WorkloadProfile, sla: SLAConstraints, u_cpu: float, m_wrk: int
) -> Dict[str, float]:
    """Evaluate all analytical performance, risk, and cost metrics for (u_cpu, m_wrk)."""
    c_int = max(1, int(round(u_cpu)))
    c_service = min(m_wrk, c_int)

    p99_base_s = response_time_quantile(0.99, wp.lam, wp.mu, c_service, wp.c_s2)
    p_thr = compound_poisson_throttle_prob(wp.lam, wp.mu, u_cpu, m_wrk, wp.tp, wp.c_s2)

    # Effective tail penalty under unaligned throttling bursts
    throttle_penalty_s = 0.0
    if p_thr > 0.01 and not math.isinf(p99_base_s):
        throttle_penalty_s = 0.5 * wp.tp * p_thr

    p99_ms = (p99_base_s + throttle_penalty_s) * 1000.0
    mem = coupled_memory_requirement(wp, u_cpu, m_wrk, sla.eps_oom)

    if math.isinf(p99_ms) or math.isinf(mem["limit_mb"]):
        cost = float("inf")
        feasible = False
    else:
        cost = sla.w_cpu * u_cpu + sla.w_mem_gb * (mem["limit_mb"] / 1024.0)
        feasible = (
            p99_ms <= sla.t_slo_ms
            and p_thr <= sla.eps_thr
            and mem["rho"] < 1.0
        )

    return {
        "u_cpu": u_cpu,
        "m_wrk": m_wrk,
        "p99_ms": p99_ms,
        "p_thr": p_thr,
        "mean_L": mem.get("mean_L", float("inf")),
        "rho": mem.get("rho", float("inf")),
        "mem_mean_mb": mem["mean_mb"],
        "mem_limit_mb": mem["limit_mb"],
        "cost_usd": cost,
        "feasible": feasible,
    }


def optimize_aoam(
    wp: WorkloadProfile, sla: SLAConstraints, c_max: int = 32
) -> Optional[Dict[str, float]]:
    """
    Formal Algorithm 1 (AOAM):
    Analytically find minimum-cost feasible integer allocation (u_cpu*, m_wrk*, M_lim*)
    satisfying SLA tail latency, throttling bound, and OOM probability bound.

    1) Checks necessary intrinsic feasibility: pure service quantile R_inf(0.99) <= T_SLO.
    2) Uses monotonicity of R_0.99(c) and L(c): sets aligned thread pool m_wrk = c
       to guarantee P_thr == 0 and eliminate excess stack overhead, then finds minimal c*.
    """
    # Necessary intrinsic feasibility condition (c -> inf limit)
    if abs(wp.c_s2 - 1.0) < 1e-6:
        r_inf_ms = (-math.log(1.0 - 0.99) / wp.mu) * 1000.0
    else:
        r_inf_ms = float(gammaincinv(1.0 / wp.c_s2, 0.99) * (wp.c_s2 / wp.mu)) * 1000.0
    if r_inf_ms > sla.t_slo_ms:
        return None

    c_min = int(math.floor(wp.lam / wp.mu)) + 1
    for c in range(c_min, c_max + 1):
        res = evaluate_configuration(wp, sla, u_cpu=float(c), m_wrk=c)
        if res["feasible"]:
            return res
    return None
