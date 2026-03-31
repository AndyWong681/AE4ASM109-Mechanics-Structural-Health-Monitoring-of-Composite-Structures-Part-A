import numpy as np
import matplotlib.pyplot as plt

# load saved data
data = np.load("failure_envelope_results.npz")

xyload_FPF = data["xyload_FPF"]
xyload_LPF = data["xyload_LPF"]
xystrain_FPF = data["xystrain_FPF"]
xystrain_LPF = data["xystrain_LPF"]
xyld_FPF = data["xyld_FPF"]
xyld_LPF = data["xyld_LPF"]

ysload_FPF = data["ysload_FPF"]
ysload_LPF = data["ysload_LPF"]
ysstrain_FPF = data["ysstrain_FPF"]
ysstrain_LPF = data["ysstrain_LPF"]
ysld_FPF = data["ysld_FPF"]
ysld_LPF = data["ysld_LPF"]


def plot_failure_envelope(mode, load_FPF, load_LPF):

    ind = 0
    xlab = "$σ_{x}$ [MPa]"
    ylab = "$σ_{y}$ [MPa]"
    if mode == "nyns":
        ind = 1
        xlab = "$σ_{y}$ [MPa]"
        ylab = "$τ_{xy}$ [MPa]"

    fig = plt.figure(figsize=(6,6))

    if len(load_FPF) > 0:
        plt.fill(load_FPF[:,0 + ind], load_FPF[:,1 + ind],
                 color='blue', alpha=0.1)
        plt.plot(load_FPF[:,0 + ind], load_FPF[:,1 + ind], 'b--',
                 label="First ply failure FPF")

    if len(load_LPF) > 0:
        plt.fill(load_LPF[:,0 + ind], load_LPF[:,1 + ind],
                 color='red', alpha=0.1)
        plt.plot(load_LPF[:,0 + ind], load_LPF[:,1 + ind], 'r-',
                 label="Last ply failure LPF")

    plt.xlabel(xlab, fontsize=12)
    plt.ylabel(ylab,fontsize=12)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.legend(fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plt.show()
    fig.savefig(("2a_fe_" + mode), dpi=200)


def plot_strain(mode, strain_FPF, strain_LPF, load_FPF, load_LPF):

    def compute_angle(load, mode):

        if mode == "nxny":
            ang = np.rad2deg(np.arctan2(load[:, 1], load[:, 0]))

        elif mode == "nyns":
            ang = np.rad2deg(np.arctan2(load[:, 2], load[:, 1]))

        return (ang + 360) % 360

    ang_FPF = compute_angle(load_FPF, mode)
    ang_LPF = compute_angle(load_LPF, mode)

    order_FPF = np.argsort(ang_FPF)
    order_LPF = np.argsort(ang_LPF)

    ang_FPF = ang_FPF[order_FPF]
    strain_FPF = strain_FPF[order_FPF]

    ang_LPF = ang_LPF[order_LPF]
    strain_LPF = strain_LPF[order_LPF]

    strain_sym = ["$ε_{x}$", "$ε_{y}$", "$γ_{xy}$"]

    for i in range(3):

        fig = plt.figure(figsize=(7.5, 5))

        if len(strain_FPF):
            plt.plot(ang_FPF, strain_FPF[:,i], 'b--',
                     label="First ply failure FPF",)

        if len(strain_LPF):
            plt.plot(ang_LPF, strain_LPF[:,i], 'r-',
                     label="Last ply failure LPF")

        plt.xlabel("θᵣ [°]",fontsize=12)
        plt.ylabel(f"{strain_sym[i]} [ε]",fontsize=12)
        plt.xlim(0, 360)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.legend(fontsize=12)
        plt.grid(True)

        plt.show()
        fig.savefig(("2a_e" + str(i) + mode), dpi=200)


def plot_load(mode, load_FPF, load_LPF):
    def compute_angle(load, mode):

        if mode == "nxny":
            ang = np.rad2deg(np.arctan2(load[:, 1], load[:, 0]))

        elif mode == "nyns":
            ang = np.rad2deg(np.arctan2(load[:, 2], load[:, 1]))

        return (ang + 360) % 360

    def compute_mag(load, mode):
        if mode == "nxny":
            mag = np.sqrt(np.square(load[:, 1]) + np.square(load[:, 0]))

        elif mode == "nyns":
            mag = np.sqrt(np.square(load[:, 2]) + np.square(load[:, 1]))

        return mag


    ang_FPF = compute_angle(load_FPF, mode)
    ang_LPF = compute_angle(load_LPF, mode)
    mag_FPF = compute_mag(load_FPF, mode)
    mag_LPF = compute_mag(load_LPF, mode)

    order_FPF = np.argsort(ang_FPF)
    order_LPF = np.argsort(ang_LPF)

    ang_FPF = ang_FPF[order_FPF]
    ang_LPF = ang_LPF[order_LPF]
    mag_FPF = mag_FPF[order_FPF]
    mag_LPF = mag_LPF[order_LPF]

    lddiff = abs(mag_FPF - mag_LPF)

    fig = plt.figure(figsize=(6, 6))
    plt.plot(ang_FPF, mag_FPF, 'b--', label="First ply failure FPF")
    plt.plot(ang_LPF, mag_LPF, 'r-', label="First ply failure FPF")
    plt.plot(ang_FPF, lddiff, '--', label="Load magnitude difference", color = 'grey')
    plt.xlabel("θᵣ [°]", fontsize = 12)
    plt.ylabel("load magnitude [MPa]", fontsize = 12)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.xlim(0, 360)

    plt.grid(True)
    plt.legend(fontsize = 10)

    plt.show()
    fig.savefig(("2b_load_" + mode), dpi=200)

    print(f"mode {mode} | Largest load difference: {np.max(lddiff)} [MPa] | at {ang_FPF[np.argmax(lddiff)]} [°]"
          f" ")


plot_failure_envelope("nxny", xyload_FPF, xyload_LPF)
plot_failure_envelope("nyns", ysload_FPF, ysload_LPF)

plot_strain("nxny", xystrain_FPF, xystrain_LPF, xyld_FPF, xyld_LPF)
plot_strain("nyns", ysstrain_FPF, ysstrain_LPF, ysld_FPF, ysld_LPF)

plot_load("nxny", xyload_FPF, xyload_LPF)
plot_load("nyns", ysload_FPF, ysload_LPF)