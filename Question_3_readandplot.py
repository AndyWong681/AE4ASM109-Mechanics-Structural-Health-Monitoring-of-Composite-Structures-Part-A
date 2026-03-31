import matplotlib.pyplot as plt
import numpy as np

data = np.load("convergence_output.npz")
x1 = data["x1"]
pf1 = data["pf1"]
x2 = data["x2"]
pf2 = data["pf2"]
E1 = data["E1"]
E2 = data["E2"]
v12 = data["v12"]
G12 = data["G12"]
Xt = data["Xt"]
Xc = data["Xc"]
Yt = data["Yt"]
Yc = data["Yc"]
S = data["S"]

E1_mean  = 172.3e9
E2_mean  = 10.2e9
v12_mean = 0.25
G12_mean = 5.58e9


def plot_failure_rate(x1, pf1, x2, pf2):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(x1, 100*pf1, label='N = 783 N/mm', linewidth=2.0, color='red', zorder = 1)
    plt.plot(x2, 100*pf2, label='N = 1219 N/mm', linewidth=2.0, color='blue', zorder = 1)

    plt.axhline(y=100*pf1[-1], linestyle='--', color='grey', linewidth=1.5, zorder = 0, label = f"N = 783 N/mm, final Pf at {round(100*pf1[-1], 2)} [%]")
    plt.axhline(y=100*pf2[-1], linestyle='-.', color='grey', linewidth=1.5, zorder = 0, label = f"N = 1219 N/mm, final Pf at {round(100*pf2[-1], 2)} [%]")

    plt.xlabel('iterations [-]', fontsize = 10)
    plt.ylabel("$P_{f}$ [%]", fontsize = 10)
    #plt.title('Monte Carlo convergence of first ply failure probability')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize = 10)
    plt.tight_layout()
    plt.show()
    fig.savefig(("3_MCconv"), dpi=200)

    rel_diff_pf1 = np.abs(np.diff(pf1) / pf1[:-1])
    rel_diff_pf2 = np.abs(np.diff(pf2) / pf2[:-1])

    for k in range(len(rel_diff_pf1) - 1, -1, -1):
        if rel_diff_pf1[k] >= 0.005:
            print(f"pf1 convergence at iteration: {np.arange(len(rel_diff_pf1))[k] + 1}")
            break

    for k in range(len(rel_diff_pf2) - 1, -1, -1):
        if rel_diff_pf2[k] >= 0.005:
            print(f"pf2 convergence at iteration: {np.arange(len(rel_diff_pf2))[k] + 1}")
            break

    fig = plt.figure(figsize=(6, 5))
    plt.plot(x1[1:], rel_diff_pf1 * 100, label='relative difference N = 783 N/mm', linewidth=2, color='blue')
    plt.axhline(y=0.5, linestyle='--', color='grey', label='0.5% criterion')
    plt.xlabel('iteration [-]', fontsize = 14)
    plt.ylabel('relative difference [%]', fontsize = 14)
    plt.legend(fontsize = 11)
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)
    plt.grid(True)
    plt.show()
    fig.savefig(("3_783rd"), dpi=200)

    fig = plt.figure(figsize=(6, 5))
    plt.plot(x2[1:], rel_diff_pf2 * 100, label='relative difference N = 1219 N/mm', linewidth=2, color='blue')
    plt.axhline(y=0.5, linestyle='--', color='grey', label='0.5% criterion')
    plt.xlabel('iteration [-]', fontsize = 14)
    plt.ylabel('relative difference [%]', fontsize = 14)
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)
    plt.legend(fontsize = 11)
    plt.grid(True)
    plt.show()
    fig.savefig(("3_1219rd"), dpi=200)


# The addition of Xt, Xc, Yt, Yc, S are written by AI
def property_conv(E1, E2, v12, G12, Xt, Xc, Yt, Yc, S):
    # running mean
    E1_rm = np.cumsum(E1) / np.arange(1, len(E1) + 1)
    E2_rm = np.cumsum(E2) / np.arange(1, len(E2) + 1)
    v12_rm = np.cumsum(v12) / np.arange(1, len(v12) + 1)
    G12_rm = np.cumsum(G12) / np.arange(1, len(G12) + 1)
    Xt_rm = np.cumsum(Xt) / np.arange(1, len(Xt) + 1)
    Xc_rm = np.cumsum(Xc) / np.arange(1, len(Xc) + 1)
    Yt_rm = np.cumsum(Yt) / np.arange(1, len(Yt) + 1)
    Yc_rm = np.cumsum(Yc) / np.arange(1, len(Yc) + 1)
    S_rm = np.cumsum(S) / np.arange(1, len(S) + 1)

    pltarr = np.array([E1_rm, E2_rm, v12_rm, G12_rm, Xt_rm, Xc_rm, Yt_rm, Yc_rm, S_rm], dtype=object)
    leg_arr = np.array(["E1", "E2", "v12", "G12", "Xt", "Xc", "Yt", "Yc", "S"])
    unit_arr = np.array([" [GPa]", " [GPa]", " [-]", " [GPa]", " [MPa]", " [MPa]", " [MPa]", " [MPa]", " [MPa]"])
    mean_arr = np.array([172.3e9, 10.2e9, 0.25, 5.58e9, 1500e6, 1200e6, 50e6, 250e6, 70e6])

    for i in range(len(pltarr)):
        if i == 2:          # v12: dimensionless
            scale = 1
        elif i <= 3:        # E1, E2, G12: GPa
            scale = 1e9
        else:               # Xt, Xc, Yt, Yc, S: MPa
            scale = 1e6

        rd = 100*np.abs(np.diff(pltarr[i]) / pltarr[i][:-1])
        fig = plt.figure(figsize=(6, 5))
        plt.plot(np.arange(len(rd)), rd, label=leg_arr[i] + ' relative difference', linewidth=2, color='blue')
        plt.axhline(y=0.5, linestyle='--', color='grey', label="0.5% criterion")
        plt.xlabel('iteration [-]', fontsize=12)
        plt.ylabel('relative difference [%]', fontsize=12)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.show()
        fig.savefig(("3_" + leg_arr[i] + "conv"), dpi=200)

        # relative difference and convergence

        iter = 0
        for k in range(len(rd)):
            if rd[k] >= 0.5:
                iter = k
        print(f"{leg_arr[i]} convergence at iteration: {iter}")



print(f"Final failure rate for N = 783 N/mm  : {pf1[-1]:.6f}")
print(f"Final failure rate for N = 1219 N/mm : {pf2[-1]:.6f}")
plot_failure_rate(x1, pf1, x2, pf2)
property_conv(E1, E2, v12, G12, Xt, Xc, Yt, Yc, S)