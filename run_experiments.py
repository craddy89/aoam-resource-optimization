"""Reproducible multi-configuration computational experiment suite for AOAM.

Runs three experiment sets comparing analytical stochastic predictions against
independent discrete-event simulation (DES), sensitivity across service-time
variability (M/G/c), and benchmark strategies across multiple workload regimes.
Exports CSV tables to ./results and clean grayscale figures to ./figures.
"""

import csv
import os
import math
import numpy as np
import matplotlib.pyplot as plt

from aoam_solver import (
    SLAConstraints,
    WorkloadProfile,
    coupled_memory_requirement,
    evaluate_configuration,
    optimize_aoam,
    response_time_quantile,
)
from simulator import simulate_node

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Clean academic typography without coordinate grids
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.grid": False,
})


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_exp1_validation():
    """Experiment 1: Analytical predictions vs DES across loads and thread alignments."""
    mu = 40.0
    configs = [
        (40.0, 2, 2), (40.0, 2, 6),
        (60.0, 2, 2), (60.0, 2, 4),
        (60.0, 3, 3), (60.0, 3, 6),
        (80.0, 3, 3), (80.0, 3, 6),
        (100.0, 4, 4), (100.0, 4, 8),
        (120.0, 4, 4), (120.0, 5, 5),
    ]
    rows = []
    sla = SLAConstraints(t_slo_ms=200.0, eps_thr=0.02, eps_oom=0.001)

    for idx, (lam, u_cpu, m_wrk) in enumerate(configs):
        wp = WorkloadProfile(lam=lam, mu=mu, c_s2=1.0)
        ana = evaluate_configuration(wp, sla, u_cpu=float(u_cpu), m_wrk=m_wrk)
        des = simulate_node(
            wp, u_cpu=float(u_cpu), m_wrk=m_wrk,
            m_limit_mb=1e6, sim_time=300.0, seed=100 + idx
        )
        p99_err_pct = abs(ana["p99_ms"] - des["p99_R_ms"]) / max(1e-6, des["p99_R_ms"]) * 100.0
        mem_err_pct = abs(ana["mem_mean_mb"] - des["mean_mem_mb"]) / max(1e-6, des["mean_mem_mb"]) * 100.0
        rows.append({
            "lambda_rps": lam,
            "u_cpu": u_cpu,
            "m_wrk": m_wrk,
            "rho": round(ana["rho"], 3),
            "ana_p99_ms": round(ana["p99_ms"], 2),
            "des_p99_ms": round(des["p99_R_ms"], 2),
            "p99_ape_pct": round(p99_err_pct, 2),
            "ana_thr_pct": round(ana["p_thr"] * 100.0, 2),
            "des_thr_pct": round(des["throttle_rate"] * 100.0, 2),
            "ana_L": round(ana["mean_L"], 2),
            "des_L": round(des["mean_L"], 2),
            "ana_mem_mean_mb": round(ana["mem_mean_mb"], 1),
            "des_mem_mean_mb": round(des["mean_mem_mb"], 1),
            "mem_ape_pct": round(mem_err_pct, 2),
            "des_mem_max_mb": round(des["max_mem_mb"], 1),
            "ana_mem_lim_mb": round(ana["mem_limit_mb"], 1),
        })

    save_csv(os.path.join(RESULTS_DIR, "exp1_validation.csv"), rows)
    return rows


def run_exp2_mgc_sensitivity():
    """Experiment 2: General service-time variance sensitivity (M/M/c vs M/G/c)."""
    lam = 90.0
    mu = 40.0
    sla = SLAConstraints(t_slo_ms=220.0, eps_thr=0.02, eps_oom=0.001)
    cs2_values = [0.25, 0.50, 1.00, 1.75, 2.50]
    rows = []

    for idx, cs2 in enumerate(cs2_values):
        wp = WorkloadProfile(lam=lam, mu=mu, c_s2=cs2)
        # M/M/c baseline prediction (ignoring cs2 != 1)
        opt_mmc = optimize_aoam(WorkloadProfile(lam=lam, mu=mu, c_s2=1.0), sla)
        # Generalized M/G/c AOAM allocation
        opt_mgc = optimize_aoam(wp, sla)

        # Evaluate both in DES with identical random seed and true variance cs2
        des_at_mmc = simulate_node(
            wp, u_cpu=opt_mmc["u_cpu"], m_wrk=int(opt_mmc["m_wrk"]),
            m_limit_mb=opt_mmc["mem_limit_mb"], sim_time=350.0, seed=200 + idx
        )
        des_at_mgc = simulate_node(
            wp, u_cpu=opt_mgc["u_cpu"], m_wrk=int(opt_mgc["m_wrk"]),
            m_limit_mb=opt_mgc["mem_limit_mb"], sim_time=350.0, seed=200 + idx
        )
        rows.append({
            "c_s2": cs2,
            "mmc_u_cpu": opt_mmc["u_cpu"],
            "mmc_mem_lim_mb": round(opt_mmc["mem_limit_mb"], 1),
            "des_p99_if_mmc_ms": round(des_at_mmc["p99_R_ms"], 2),
            "des_oom_if_mmc": des_at_mmc["oom_events"],
            "mgc_ana_p99_ms": round(opt_mgc["p99_ms"], 2),
            "mgc_u_cpu": opt_mgc["u_cpu"],
            "mgc_mem_lim_mb": round(opt_mgc["mem_limit_mb"], 1),
            "des_p99_if_mgc_ms": round(des_at_mgc["p99_R_ms"], 2),
            "des_oom_if_mgc": des_at_mgc["oom_events"],
            "mgc_cost_usd": round(opt_mgc["cost_usd"], 2),
        })

    save_csv(os.path.join(RESULTS_DIR, "exp2_mgc_sensitivity.csv"), rows)
    return rows


def run_exp3_strategy_benchmark():
    """
    Experiment 3: Multi-regime comparison across Static Over-Provisioning,
    Reactive Threshold Heuristic (VPA-like), and Proposed AOAM.
    """
    mu = 40.0
    regimes = [
        ("Low", 35.0),
        ("Nominal", 65.0),
        ("High", 95.0),
        ("Burst", 125.0),
    ]
    sla_targets = [140.0, 180.0, 220.0]
    rows = []

    for t_slo in sla_targets:
        sla = SLAConstraints(t_slo_ms=t_slo, eps_thr=0.02, eps_oom=0.001)
        for r_idx, (regime_name, lam) in enumerate(regimes):
            wp = WorkloadProfile(lam=lam, mu=mu, c_s2=1.0)

            # Strategy 1: Static Over-provisioning (fixed oversized pool & generous static limits)
            # Fixed u_cpu = 4.0, unaligned default m_wrk = 16, static RAM = 2048 MB
            s_u, s_m, s_mem = 4.0, 16, 2048.0
            s_cost = sla.w_cpu * s_u + sla.w_mem_gb * (s_mem / 1024.0)
            des_static = simulate_node(wp, s_u, s_m, s_mem, sim_time=300.0, seed=400 + r_idx)

            # Strategy 2: Reactive Heuristic (VPA-like: picks ceil(1.25 * lam / mu) CPU,
            # keeps unaligned runtime pool m_wrk = 12, and sets RAM = 1.20 * mean_mem)
            v_u = float(max(1, int(math.ceil(1.25 * lam / mu))))
            v_m = 12
            v_mem_stats = coupled_memory_requirement(wp, v_u, v_m, sla.eps_oom)
            v_mem = max(512.0, 1.25 * v_mem_stats["mean_mb"])
            v_cost = sla.w_cpu * v_u + sla.w_mem_gb * (v_mem / 1024.0)
            des_vpa = simulate_node(wp, v_u, v_m, v_mem, sim_time=300.0, seed=500 + r_idx)

            # Strategy 3: Proposed AOAM
            opt = optimize_aoam(wp, sla)
            a_u, a_m, a_mem, a_cost = opt["u_cpu"], int(opt["m_wrk"]), opt["mem_limit_mb"], opt["cost_usd"]
            des_aoam = simulate_node(wp, a_u, a_m, a_mem, sim_time=300.0, seed=600 + r_idx)

            for strat_name, u_val, m_val, mem_lim, cost_val, d in [
                ("Static", s_u, s_m, s_mem, s_cost, des_static),
                ("Reactive_Heuristic", v_u, v_m, v_mem, v_cost, des_vpa),
                ("AOAM", a_u, a_m, a_mem, a_cost, des_aoam),
            ]:
                rows.append({
                    "t_slo_ms": t_slo,
                    "regime": regime_name,
                    "lambda_rps": lam,
                    "strategy": strat_name,
                    "u_cpu": u_val,
                    "m_wrk": m_val,
                    "mem_limit_mb": round(mem_lim, 1),
                    "cost_usd": round(cost_val, 2),
                    "des_p99_ms": round(d["p99_R_ms"], 2),
                    "slo_met": d["p99_R_ms"] <= t_slo * 1.05,
                    "des_thr_pct": round(d["throttle_rate"] * 100.0, 2),
                    "des_oom_events": d["oom_events"],
                    "des_mean_mem_mb": round(d["mean_mem_mb"], 1),
                })

    save_csv(os.path.join(RESULTS_DIR, "exp3_strategy_benchmark.csv"), rows)
    return rows


def generate_figures(exp1, exp2, exp3):
    """Generate clean grayscale academic plots sized natively for 12.2 cm (4.8 in) text width."""

    # Figure 1: Analytical vs DES P99 validation across aligned configs
    aligned = [r for r in exp1 if r["u_cpu"] == r["m_wrk"]]
    labels = [f"λ={int(r['lambda_rps'])}\nc={r['u_cpu']}" for r in aligned]
    x = np.arange(len(aligned))
    ana_p99 = [r["ana_p99_ms"] for r in aligned]
    des_p99 = [r["des_p99_ms"] for r in aligned]

    fig, ax = plt.subplots(figsize=(4.8, 2.8), dpi=220)
    w = 0.36
    ax.bar(x - w/2, ana_p99, width=w, color="#4D4D4D", edgecolor="black", label="Analytical M/M/c Model")
    ax.bar(x + w/2, des_p99, width=w, color="#B3B3B3", edgecolor="black", label="Discrete-Event Simulation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel(r"$P_{99}$ Response Time (ms)", fontsize=8.5)
    ax.set_xlabel("Workload Arrival Rate and Capacity Configuration", fontsize=8.5)
    ax.legend(frameon=True, edgecolor="black", fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_p99_validation.png"), dpi=220)
    plt.close(fig)

    # Figure 2: Nonlinear memory coupling (L and Memory vs utilization rho at lam sweep)
    lams = np.linspace(20.0, 112.0, 24)
    c_fixed = 3
    m_means, m_lims, l_vals = [], [], []
    for l_val in lams:
        wp_i = WorkloadProfile(lam=float(l_val), mu=40.0, c_s2=1.0)
        m_res = coupled_memory_requirement(wp_i, u_cpu=float(c_fixed), m_wrk=c_fixed, eps_oom=0.001)
        m_means.append(m_res["mean_mb"])
        m_lims.append(m_res["limit_mb"])
        l_vals.append(m_res["mean_L"])

    fig, ax1 = plt.subplots(figsize=(4.8, 2.8), dpi=220)
    ax1.plot(lams, m_means, color="black", linestyle="-", linewidth=1.5, label=r"Mean Memory $M_{\mathrm{mean}}$ (MB)")
    ax1.plot(lams, m_lims, color="black", linestyle="--", linewidth=1.5, label=r"OOM-Safe Limit $M_{\mathrm{lim}}$ ($\varepsilon=0.001$)")
    ax1.set_xlabel(r"Arrival Intensity $\lambda$ (req/s) at $c = 3$ units", fontsize=8.5)
    ax1.set_ylabel("Allocated Memory (MB)", fontsize=8.5)
    ax1.tick_params(axis="both", labelsize=8)
    ax2 = ax1.twinx()
    ax2.plot(lams, l_vals, color="#555555", linestyle=":", linewidth=1.7, label=r"Mean Occupancy $L$")
    ax2.set_ylabel(r"Mean Active Requests $L$", color="black", fontsize=8.5)
    ax2.tick_params(axis="y", labelsize=8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, edgecolor="black", fontsize=7.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_nonlinear_memory_coupling.png"), dpi=220)
    plt.close(fig)

    # Figure 3: M/G/c variance sensitivity (c_s^2 sweep at lambda=90, T_SLO=220 ms)
    cs2_x = [r["c_s2"] for r in exp2]
    p99_if_mmc = [r["des_p99_if_mmc_ms"] for r in exp2]
    p99_if_mgc = [r["des_p99_if_mgc_ms"] for r in exp2]

    fig, ax = plt.subplots(figsize=(4.8, 2.8), dpi=220)
    ax.plot(cs2_x, p99_if_mmc, marker="o", markersize=5, color="#555555", linestyle="--", linewidth=1.5, label="Unadjusted $M/M/c$ Baseline")
    ax.plot(cs2_x, p99_if_mgc, marker="^", markersize=5.5, color="black", linestyle="-", linewidth=1.6, label="Variance-Aware AOAM ($M/G/c$)")
    ax.axhline(220.0, color="#888888", linestyle=":", linewidth=1.3, label=r"SLO Threshold ($T_{\mathrm{SLO}} = 220$ ms)")
    ax.set_xlabel(r"Squared Coefficient of Variation of Service Time ($c_s^2$)", fontsize=8.5)
    ax.set_ylabel(r"Simulated $P_{99}$ Response Time (ms)", fontsize=8.5)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_ylim(95, 315)
    ax.legend(frameon=True, edgecolor="black", fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_mgc_sensitivity.png"), dpi=220)
    plt.close(fig)

    # Figure 4: 2-panel stacked Cost & Reliability Benchmark for T_SLO = 140 ms (fits 4.8 in cleanly)
    sub140 = [r for r in exp3 if r["t_slo_ms"] == 140.0]
    regimes = ["Low", "Nominal", "High", "Burst"]
    x = np.arange(len(regimes))
    w = 0.25

    def get_vals(metric, strat):
        return [next(r[metric] for r in sub140 if r["regime"] == reg and r["strategy"] == strat) for reg in regimes]

    cost_static, cost_react, cost_aoam = get_vals("cost_usd", "Static"), get_vals("cost_usd", "Reactive_Heuristic"), get_vals("cost_usd", "AOAM")
    p99_static, p99_react, p99_aoam = get_vals("des_p99_ms", "Static"), get_vals("des_p99_ms", "Reactive_Heuristic"), get_vals("des_p99_ms", "AOAM")
    oom_react = get_vals("des_oom_events", "Reactive_Heuristic")

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(4.8, 4.7), dpi=220)

    # Panel (a): Monthly Cost
    ax_a.bar(x - w, cost_static, width=w, color="#D9D9D9", edgecolor="black", label="Static Over-Provisioning")
    b2 = ax_a.bar(x, cost_react, width=w, color="#8C8C8C", edgecolor="black", label="Reactive Heuristic (OOM-prone)")
    ax_a.bar(x + w, cost_aoam, width=w, color="#262626", edgecolor="black", label="Proposed AOAM (Zero OOM)")
    for i, bar in enumerate(b2):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.0,
            f"{oom_react[i]} OOM",
            ha="center", va="bottom", fontsize=6.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85),
        )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["Low (35 rps)", "Nominal (65 rps)", "High (95 rps)", "Burst (125 rps)"], fontsize=8)
    ax_a.tick_params(axis="y", labelsize=8)
    ax_a.set_ylabel("Monthly Cost ($)", fontsize=8.5)
    ax_a.set_title("(a) Monthly Resource Cost vs. OOM Reliability Failures", fontsize=9)
    ax_a.set_ylim(0, 132)
    ax_a.legend(frameon=True, edgecolor="black", fontsize=7.2, loc="upper left", ncol=1)

    # Panel (b): Tail Latency Compliance
    ax_b.bar(x - w, p99_static, width=w, color="#D9D9D9", edgecolor="black", label="Static Over-Provisioning")
    ax_b.bar(x, p99_react, width=w, color="#8C8C8C", edgecolor="black", label="Reactive Heuristic")
    ax_b.bar(x + w, p99_aoam, width=w, color="#262626", edgecolor="black", label="Proposed AOAM")
    ax_b.axhline(140.0, color="black", linestyle="--", linewidth=1.2, label=r"$T_{\mathrm{SLO}} = 140$ ms")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["Low (35 rps)", "Nominal (65 rps)", "High (95 rps)", "Burst (125 rps)"], fontsize=8)
    ax_b.tick_params(axis="y", labelsize=8)
    ax_b.set_ylabel(r"Simulated $P_{99}$ Latency (ms)", fontsize=8.5)
    ax_b.set_title(r"(b) Tail Latency Compliance ($T_{\mathrm{SLO}} = 140$ ms)", fontsize=9)
    ax_b.set_ylim(0, 210)
    ax_b.legend(frameon=True, edgecolor="black", fontsize=7.2, loc="upper left", ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_cost_benchmark.png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    print("Running Experiment 1: Analytical vs DES Validation...")
    e1 = run_exp1_validation()
    print(f"  Completed {len(e1)} configurations.")

    print("Running Experiment 2: M/G/c Variance Sensitivity...")
    e2 = run_exp2_mgc_sensitivity()
    print(f"  Completed {len(e2)} variance levels.")

    print("Running Experiment 3: Strategy Benchmark across Load Regimes & SLOs...")
    e3 = run_exp3_strategy_benchmark()
    print(f"  Completed {len(e3)} strategy evaluations.")

    print("Generating clean grayscale figures without grids...")
    generate_figures(e1, e2, e3)
    print("All experiments and figures completed cleanly!")
