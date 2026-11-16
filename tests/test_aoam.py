"""Unit tests verifying mathematical properties of AOAM analytical model
and consistency with the independent discrete-event simulator.
"""

import math
import unittest
from aoam_solver import (
    SLAConstraints,
    WorkloadProfile,
    compound_poisson_throttle_prob,
    coupled_memory_requirement,
    erlang_c,
    optimize_aoam,
    response_time_quantile,
)
from simulator import simulate_node


class TestAOAMModel(unittest.TestCase):
    def test_erlang_c_reduces_to_mm1(self):
        """For c = 1, Erlang-C delay probability equals rho = lam / mu."""
        for lam in [10.0, 25.0, 35.0]:
            mu = 40.0
            c_val = erlang_c(1, lam / mu)
            self.assertAlmostEqual(c_val, lam / mu, places=10)

    def test_mm1_exact_quantile(self):
        """For M/M/1, P(R > t) = exp(-(mu - lam)*t), so R_q = -ln(1-q)/(mu - lam)."""
        lam, mu, q = 20.0, 50.0, 0.99
        expected = -math.log(1.0 - q) / (mu - lam)
        computed = response_time_quantile(q, lam, mu, c=1, c_s2=1.0)
        self.assertAlmostEqual(computed, expected, places=7)

    def test_monotonicity_of_latency_and_memory(self):
        """Increasing CPU allocation strictly decreases P99 latency and mean L."""
        wp = WorkloadProfile(lam=70.0, mu=40.0)
        p99_2 = response_time_quantile(0.99, wp.lam, wp.mu, c=2)
        p99_3 = response_time_quantile(0.99, wp.lam, wp.mu, c=3)
        p99_4 = response_time_quantile(0.99, wp.lam, wp.mu, c=4)
        self.assertGreater(p99_2, p99_3)
        self.assertGreater(p99_3, p99_4)

        m2 = coupled_memory_requirement(wp, u_cpu=2.0, m_wrk=2, eps_oom=0.001)
        m3 = coupled_memory_requirement(wp, u_cpu=3.0, m_wrk=3, eps_oom=0.001)
        self.assertGreater(m2["mean_L"], m3["mean_L"])
        self.assertGreater(m2["limit_mb"], m3["limit_mb"])

    def test_aligned_pool_zero_throttling(self):
        """When m_wrk <= u_cpu for integer quota, throttling must be 0 in both math and DES."""
        wp = WorkloadProfile(lam=60.0, mu=40.0)
        p_thr = compound_poisson_throttle_prob(wp.lam, wp.mu, u_cpu=3.0, m_wrk=3, tp=0.1)
        self.assertEqual(p_thr, 0.0)
        des = simulate_node(wp, u_cpu=3.0, m_wrk=3, m_limit_mb=1200.0, sim_time=120.0, seed=1)
        self.assertEqual(des["throttle_rate"], 0.0)

    def test_aoam_solver_satisfies_slo_and_oom_in_des(self):
        """Optimal AOAM allocation must satisfy P99 <= T_SLO and zero OOM in simulation."""
        wp = WorkloadProfile(lam=80.0, mu=40.0, c_s2=1.0)
        sla = SLAConstraints(t_slo_ms=150.0, eps_thr=0.02, eps_oom=0.001)
        opt = optimize_aoam(wp, sla)
        self.assertIsNotNone(opt)
        self.assertTrue(opt["feasible"])
        des = simulate_node(
            wp,
            u_cpu=opt["u_cpu"],
            m_wrk=int(opt["m_wrk"]),
            m_limit_mb=opt["mem_limit_mb"],
            sim_time=250.0,
            seed=7,
        )
        self.assertLessEqual(des["p99_R_ms"], sla.t_slo_ms * 1.1)
        self.assertEqual(des["throttle_rate"], 0.0)
        self.assertEqual(des["oom_events"], 0)

    def test_infeasible_intrinsic_slo_returns_none(self):
        """If pure service-time P99 exceeds T_SLO, no finite c can satisfy SLA."""
        # mu=40 (mean 25ms), exp P99 is 115.13ms > 100ms
        wp = WorkloadProfile(lam=20.0, mu=40.0, c_s2=1.0)
        sla = SLAConstraints(t_slo_ms=100.0)
        self.assertIsNone(optimize_aoam(wp, sla))


if __name__ == "__main__":
    unittest.main()
