import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axis import XTick
#from Tools.demo.spreadsheet import xml2align
from scipy.signal import savgol_filter

E1_mean  = 172.3e9
E2_mean  = 10.2e9
v12_mean = 0.25
G12_mean = 5.58e9

Xt_mean = 1923e6
Xc_mean = 1480e6
Yt_mean = 84e6
Yc_mean = 220e6
S_mean  = 144.5e6

E1_std  = 9.28e9
E2_std  = 3.58e9
v12_std = 0.018
G12_std = 1.97e9

Xt_std = 188.3e6
Yt_std = 8.2e6
S_std  = 7.33e6

# assume std equal mean * 0.1
Xc_std = 0.10 * Xc_mean   # assumed
Yc_std = 0.10 * Yc_mean   # assumed

E1f = 225e9
v12f = 0.2


half_stack = [0, 90, 45, -45, 0, 90, 45, -45]
stack = half_stack + half_stack[::-1]
nplies = len(stack)

t_ply = 0.125e-3

# Functions for Q, ABD matrx tranformation and etc are directly from the previous question
def get_Q(E1, E2, v12, G12):
    v21 = v12 * E2 / E1
    d = 1.0 - v12 * v21
    return np.array([
        [E1 / d,       v12 * E2 / d, 0.0],
        [v12 * E2 / d, E2 / d,       0.0],
        [0.0,          0.0,          G12]
    ])


def transform_Q(Q, angle_deg):
    t = np.radians(angle_deg)
    m, n = np.cos(t), np.sin(t)

    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0]
    ])

    T = np.array([
        [m**2,     n**2,      2.0 * m * n],
        [n**2,     m**2,     -2.0 * m * n],
        [-m*n,     m*n,       m**2 - n**2]
    ])

    return np.linalg.inv(T) @ Q @ R @ T @ np.linalg.inv(R)


def local_stress(Q, ply_angle_deg, strain_global):
    t = np.radians(ply_angle_deg)
    m, n = np.cos(t), np.sin(t)

    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0]
    ])

    T = np.array([
        [m**2,     n**2,      2.0 * m * n],
        [n**2,     m**2,     -2.0 * m * n],
        [-m*n,     m*n,       m**2 - n**2]
    ])

    return Q @ R @ T @ np.linalg.inv(R) @ strain_global


def puck_FF(s1, s2, Xt, Xc, E1, v12):
    k = (v12 - v12f * 1.1 * E1 / E1f)

    if s1 >= 0.0:
        FI = (s1 - k * s2) / Xt
    else:
        FI = -(s1 - k * s2) / Xc

    return FI >= 1.0


def puck_IFF(s2, s12, Yt, Yc, S):
    sigma_23 = S / (2.0 * 0.2) * (np.sqrt(1.0 + 2.0 * 0.2 * Yc / S) - 1.0)
    p23 = 0.2 * sigma_23 / abs(S)

    if abs(s12) < 1e-14:
        r = np.inf
    else:
        r = abs(s2 / s12)

    crit = sigma_23 / abs(S)

    if s2 >= 0.0:
        val = np.sqrt((s12 / S)**2 + (1.0 - 0.3 * Yt / S)**2 * (s2 / Yt)**2) + 0.3 * s2 / S
        return val > 1.0

    elif r <= crit:
        val = (np.sqrt(s12**2 + (0.2 * s2)**2) + 0.2 * s2) / S
        return val > 1.0

    else:
        val = ((s12 / (2.0 * (1.0 + p23) * S))**2 + (s2 / Yc)**2) * Yc / (-s2)
        return val > 1.0


def sample_positive_normal(mean, std, min_value, rng):
    while True:
        x = rng.normal(mean, std)
        if x > min_value:
            return x

# sample material properties randomly based on std
def sample_material_properties(rng):

    props = {
        "E1":  sample_positive_normal(E1_mean,  E1_std,  1e9,  rng),
        "E2":  sample_positive_normal(E2_mean,  E2_std,  1e8,  rng),
        "v12": sample_positive_normal(v12_mean, v12_std, 0.01, rng),
        "G12": sample_positive_normal(G12_mean, G12_std, 1e8,  rng),
        "Xt":  sample_positive_normal(Xt_mean,  Xt_std,  1e6,  rng),
        "Xc":  sample_positive_normal(Xc_mean,  Xc_std,  1e6,  rng),
        "Yt":  sample_positive_normal(Yt_mean,  Yt_std,  1e6,  rng),
        "Yc":  sample_positive_normal(Yc_mean,  Yc_std,  1e6,  rng),
        "S":   sample_positive_normal(S_mean,   S_std,   1e6,  rng),
    }
    return props

# runs 1 iteration to check for FPF failure, general approach is the same from 2a
def FPF_iteration(N_load, rng):
    props = sample_material_properties(rng)

    E1  = props["E1"]
    E2  = props["E2"]
    v12 = props["v12"]
    G12 = props["G12"]

    Xt = props["Xt"]
    Xc = props["Xc"]
    Yt = props["Yt"]
    Yc = props["Yc"]
    S  = props["S"]

    # build local and laminate stiffness
    Q_local = get_Q(E1, E2, v12, G12)
    A = np.zeros((3, 3))

    for ang in stack:
        Qbar = transform_Q(Q_local, ang)
        A += Qbar * t_ply

    # load: resultant N at +45 degrees to X-axis, sin and cos are squared because it's a tensor transformation and not a simple projection
    N_total = N_load * 1e3  # N/mm -> N/m
    load_ang = np.radians(45)
    Nx = N_total * np.cos(load_ang) ** 2
    Ny = N_total * np.sin(load_ang) ** 2
    Ns = N_total * np.cos(load_ang) * np.sin(load_ang)

    # using try to avoid singular matrix problems
    try:
        strain = np.linalg.solve(A, np.array([Nx, Ny, Ns]))
    except:
        print("")

    # first ply failure check
    for ang in stack:
        s1, s2, s12 = local_stress(Q_local, ang, strain)

        ff = puck_FF(s1, s2, Xt, Xc, E1, v12)
        iff = puck_IFF(s2, s12, Yt, Yc, S)

        if ff or iff:
            return 1, E1, E2, v12, G12, Xt, Xc, Yt, Yc, S

    return 0, E1, E2, v12, G12, Xt, Xc, Yt, Yc, S

# 30000 iterations
def monte_carlo_failure_rate(N_load, n_iter, seed, checkpoint):
    rng = np.random.default_rng(seed)
    failures = 0
    x_vals = []
    pf_vals = []

    E1l = []
    E2l = []
    v12l = []
    G12l = []
    Xtl = []
    Xcl = []
    Ytl = []
    Ycl = []
    Sl = []

    for i in range(1, n_iter + 1):
        Xi, E1, E2, v12, G12, Xt, Xc, Yt, Yc, S = FPF_iteration(N_load, rng)
        failures += Xi

        E1l.append(E1)
        E2l.append(E2)
        v12l.append(v12)
        G12l.append(G12)
        Xtl.append(Xt)
        Xcl.append(Xc)
        Ytl.append(Yt)
        Ycl.append(Yc)
        Sl.append(S)

        # failure rate computed everytime when a checkpoint is reach
        if i % checkpoint == 0 or i == 1:
            pf = failures / i
            x_vals.append(i)
            pf_vals.append(pf)

    return (np.array(x_vals), np.array(pf_vals), np.array(E1l), np.array(E2l), np.array(v12l),
            np.array(G12l), np.array(Xtl), np.array(Xcl), np.array(Ytl), np.array(Ycl), np.array(Sl))


if __name__ == "__main__":
    n_iter = 30000
    checkpoint = 100

    x1, pf1, E1, E2, v12, G12, Xt, Xc, Yt, Yc, S = monte_carlo_failure_rate(
        N_load=783.0,
        n_iter=n_iter,
        seed=1,
        checkpoint=checkpoint
    )

    x2, pf2, _, _, _, _, _, _, _, _, _ = monte_carlo_failure_rate(
        N_load=1219.0,
        n_iter=n_iter,
        seed=2,
        checkpoint=checkpoint
    )

    print(f"Final failure rate for N = 783 N/mm  : {pf1[-1]:.6f}")
    print(f"Final failure rate for N = 1219 N/mm : {pf2[-1]:.6f}")
    outfile = "convergence_output.npz"

    np.savez(
        outfile,
        x1 = x1,
        pf1 = pf1,
        x2 = x2,
        pf2 = pf2,
        E1 = E1,
        E2 = E2,
        v12 = v12,
        G12 = G12,
        Xt = Xt,
        Xc = Xc,
        Yt = Yt,
        Yc = Yc,
        S = S
    )

    print(f"Results exported to {outfile}")