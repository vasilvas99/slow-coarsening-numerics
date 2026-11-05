import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator
from scipy.integrate import solve_ivp

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=10)
parser.add_argument("--l0", type=float, default=3.0)
parser.add_argument("--t-final", type=float, default=50000.0)
parser.add_argument("--seed", type=int, default=314)
parser.add_argument("--output-a", type=str, default="fig1_ode_convergence_a.png")
parser.add_argument("--output-b", type=str, default="fig1_ode_convergence_b.png")
args = parser.parse_args()

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    }
)


def e1N_equilibrium(N, l0):
    total = N * l0
    c = total + N - 2
    ws = 2 * total / (c + np.sqrt(c**2 - 4 * (N - 1) * total))
    return ws, total - (N - 1) * ws


def initial_state(N, l0, rng_seed=314):
    noise = np.random.default_rng(rng_seed).normal(0, 0.1, N)
    w = l0 + noise - noise.mean()

    assert np.any(w > 0), "Initial terrace widths should be > 0"

    return w


def h(w):
    return w ** (-2) - w ** (-1)


def potential(w):
    return 1 / w + np.log(w)


def rhs(t, w):
    h_w = h(w)
    return -np.roll(h_w, -1) + 2 * h_w - np.roll(h_w, 1)


def run_integration(N, l0, t_final, rng_seed=314):
    times = np.geomspace(0.01, t_final, 4000)
    w_initial = initial_state(N, l0, rng_seed)

    sol = solve_ivp(
        rhs,
        (0, t_final),
        w_initial,
        method="LSODA",
        t_eval=times,
        rtol=1e-10,
        atol=1e-12,
        max_step=100,
    )

    return times, sol


def plot_fig_trajectories(w, times, ws_eq, wl_eq, l0, plot_path, dpi=600):
    plt.figure(figsize=(5, 4))

    plt.plot(times, np.array(w).T, lw=1.2)

    plt.axhline(ws_eq, color="k", ls="--", lw=1.2, label=r"$w_S,\ w_L$ (theory)")
    plt.axhline(wl_eq, color="k", ls="--", lw=1.2)
    plt.axhline(l0, color="gray", ls=":", lw=0.8, label=r"$\ell_0$")

    plt.xscale("log")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$w_n(t)$")
    plt.gca().xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    plt.gca().xaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10), numticks=100)
    )

    plt.legend(loc="center left")

    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_fig_energy_minimization(w, times, plot_path, dpi=600):
    energy = potential(w).sum(axis=0)

    plt.figure(figsize=(5, 4))
    plt.plot(times, energy, color="tab:red", lw=1.5)

    plt.xscale("log")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\sum_n V(w_n)$")
    plt.gca().xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    plt.gca().xaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10), numticks=100)
    )

    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main(
    N,
    l0,
    t_final,
    rng_seed=314,
    output_a="fig1_ode_convergence_a.png",
    output_b="fig1_ode_convergence_b.png",
):
    ws_eq, wl_eq = e1N_equilibrium(N, l0)
    times, sol = run_integration(N, l0, t_final, rng_seed)
    plot_fig_trajectories(sol.y, times, ws_eq, wl_eq, l0, output_a)
    plot_fig_energy_minimization(sol.y, times, output_b)


main(args.N, args.l0, args.t_final, args.seed, args.output_a, args.output_b)
