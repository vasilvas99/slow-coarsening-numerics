import argparse

import numpy as np
from scipy.optimize import root

parser = argparse.ArgumentParser()
parser.add_argument(
    "OUTPUT_TRAJECTORY",
    type=str,
    help="The path to the output trajectory",
)
parser.add_argument("--eps", type=float, default=0.0015)
parser.add_argument("--l0", type=float, default=3.0)
parser.add_argument("--N", type=int, default=4)
parser.add_argument("--NX", type=int, default=300)
parser.add_argument("--dx", type=float, default=None)
args = parser.parse_args()

TARGET_TIMES = np.unique(
    np.concatenate(
        [
            np.linspace(0.0, 1.0, 80),
            np.geomspace(1.0, 10.0, 40),
            np.geomspace(10.0, 100.0, 20),
            np.geomspace(100.0, 1000.0, 15),
            np.geomspace(1000.0, 10000.0, 10),
            np.geomspace(10000.0, 100000.0, 10),
        ]
    )
)
STEP_FACTOR = 0.0005


def initial_state(N, NX, l0, rng_seed=314):
    rng = np.random.default_rng(rng_seed)
    widths = np.full((N, NX), l0)  # constant initial widths

    for n in range(N):
        raw = rng.normal(0, 0.15 * l0, NX)  # gaussian noise
        for j in range(NX):
            widths[n, j] += (
                raw[(j - 1) % NX] + raw[j] + raw[(j + 1) % NX]
            ) / 3  # which we then smooth a bit

    for j in range(NX):
        widths[:, j] = np.clip(widths[:, j], 0.3, None)  # minimum width is 0.3
        widths[:, j] *= (
            N * l0 / widths[:, j].sum()
        )  # renormalize such that N*l0 originally is recovered

    return np.cumsum(widths, axis=0)  # sum terrace widths to step positions


def h(w):
    return w ** (-2) - w ** (-1)


# the matrices U are with indices n - rows, j - cols
def residual(u, u_old, dt, dx, eps, N, l0):
    u_nm1_j = np.roll(u, 1, axis=0)
    u_nm1_j[0, :] -= N * l0

    w = u - u_nm1_j
    h_w = h(w)
    h_wp1 = np.roll(h_w, -1, axis=0)

    u_n_jm1 = np.roll(u, 1, axis=1)
    u_n_jp1 = np.roll(u, -1, axis=1)

    lap = u_n_jm1 - 2 * u + u_n_jp1
    nonlinear = h_w - h_wp1

    res = u - u_old - (dt * eps / (dx**2)) * lap - (dt / eps) * nonlinear

    return res


def flatten_resd(u, u_old, dt, dx, shape, eps, N, l0):
    return residual(u.reshape(shape), u_old, dt, dx, eps, N, l0).ravel()


def euler_step(u, dt, dx, eps, N, l0):
    sol = root(
        flatten_resd,
        x0=u.ravel(),
        args=(u, dt, dx, u.shape, eps, N, l0),
        method="krylov",
        options={"fatol": 1e-10},
    )
    return sol.x.reshape(u.shape)


def run_euler(
    N, NX, l0, eps, dx, rng_seed=314, trajectory_file="solution_trajectory.npz"
):
    u = initial_state(N, NX, l0, rng_seed)
    snapshots = [u.copy()]
    t = 0.0
    step = 0

    for target_time in TARGET_TIMES[1:]:
        while t < target_time:
            dt = min(STEP_FACTOR * max(t, 0.01), target_time - t)
            u = euler_step(u, dt, dx, eps, N, l0)
            t += dt
            step += 1
            if step % 200 == 0:
                print(f"At {step = } at time = {t}")

        print(f"Target time {float(target_time)} reached")
        snapshots.append(u.copy())

    np.savez_compressed(
        trajectory_file,
        t=TARGET_TIMES,
        u=np.asarray(snapshots),
        x=np.arange(NX) * dx,
        N=N,
        Nx=NX,
        eps=eps,
        l0=l0,
        seed=rng_seed,
        step_factor=STEP_FACTOR,
    )


run_euler(
    args.N,
    args.NX,
    args.l0,
    args.eps,
    args.dx if args.dx is not None else 1.0 / args.NX,
    trajectory_file=args.OUTPUT_TRAJECTORY,
)
