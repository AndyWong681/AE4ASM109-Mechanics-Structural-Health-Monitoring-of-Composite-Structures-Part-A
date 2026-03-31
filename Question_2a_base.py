import numpy as np
import matplotlib.pyplot as plt
import math

# material properties
E1 = 172.3e9
E2 = 10.2e9
v12 = 0.25
G12 = 5.58e9

Xt = 1923e6
Xc = 1480e6
Yt = 84e6
Yc = 220e6
S = 144.5e6

E1f = 225e9
v12f = 0.2

# ply configuration
stack = [0, 45, -45, 90, 30]
stack = stack + stack[::-1]
stack = stack * 3

print(stack)

nplies = len(stack)

t_ply = 0.125e-3
t_tot = t_ply * nplies
z = np.linspace(-nplies/2*t_ply, nplies/2*t_ply, nplies+1)

#Assume symmetric condition, A matrix can be separated alone due to no coupling

# compute Q matrix
def get_Q(E1, E2, v12, G12):
    if E1 > 0:
        v21 = v12 * E2 / E1
    else:
        v21 = 0
    d = 1 - v12 * v21
    return np.array([
        [E1/d, v12*E2/d, 0],
        [v12*E2/d, E2/d, 0],
        [0, 0, G12]
    ])


# rotate Q matrix
def transform_Q(Q, angle):

    t = np.radians(angle)
    m, n = np.cos(t), np.sin(t)
    R = np.array([[1, 0, 0]
                     , [0, 1, 0]
                     , [0, 0, 2]])
    T = np.array([
        [m**2, n**2, 2*m*n],
        [n**2, m**2, -2*m*n],
        [-m*n, m*n, m**2 - n**2]
    ])

    return np.linalg.inv(T) @ Q @ R @ T @ np.linalg.inv(R)

# compute local stress
def local_stress(Q, i, strain):
    t = np.radians(stack[i])
    m, n = np.cos(t), np.sin(t)
    R = np.array([[1, 0, 0]
                     , [0, 1, 0]
                     , [0, 0, 2]])
    T = np.array([
        [m ** 2, n ** 2, 2 * m * n],
        [n ** 2, m ** 2, -2 * m * n],
        [-m * n, m * n, m ** 2 - n ** 2]
    ])

    # An alternative method of computing Q and transformations are found in https://www.scirp.org/journal/paperinformation?paperid=3170
    # The results should be the same as in the lecture slides

    return Q @ R @ T @ np.linalg.inv(R) @ strain


# puck criteria
def puck_FF(s1, s2):
    k = (v12 - v12f * 1.1 * E1 / E1f)

    if s1 >= 0:
        FI = (s1 - k * s2) / Xt
    else:
        FI = -(s1 - k * s2) / Xc

    return FI >= 1


def puck_IFF(s2, s12):
    sigma_23 = S / (2 * 0.2) * (np.sqrt(1 + 2 * 0.2 * Yc / S) - 1)
    p23 = 0.2 * sigma_23 / abs(S)

    if abs(s12) < 1e-12:
        r = np.inf
    else:
        r = abs(s2 / s12)

    sigma_12c = S * np.sqrt(1 + 2 * p23)
    crit = sigma_23 / abs(sigma_12c)

    if s2 >= 0:
        val = np.sqrt((s12/S)**2 + (1 - 0.3*Yt/S)**2 * (s2/Yt)**2) + 0.3*s2/S
        return val > 1

    elif r <= crit:
        val = (np.sqrt(s12**2 + (0.2*s2)**2) + 0.2*s2) / S
        return val > 1

    else:
        val = ((s12/(2*(1+p23)*S))**2 + (s2/Yc)**2) * Yc / (-s2)
        return val > 1


def compute_fail_envelope(mode):

    plot_ang = np.linspace(0, 2 * np.pi, 360)
    load_FPF = []
    load_LPF = []
    strain_FPF = []
    strain_LPF = []
    loaddirec_FPF = []
    loaddirec_LPF = []

    # evaluate each angle from 0 to 360
    for angle in plot_ang:

        print(f"computing mode: {mode} | angle: {angle * 180 / math.pi} [deg]")

        if mode == "nxny":
            loaddirec = np.array([np.cos(angle), np.sin(angle), 0])
        elif mode == "nyns":
            loaddirec = np.array([0, np.cos(angle), np.sin(angle)])

        ply_stat = np.zeros(nplies)
        FPF = False
        # load start at 1200000, so load doesn't start at 0 to reduce runtime
        load = 1200000
        strain_str = []

        # at given angle, gradually add load until laminate fails, document FPF and LPF
        while load <= 1e7:
            load += 500
            Nx, Ny, Ns = load * loaddirec[0], load * loaddirec[1], load * loaddirec[2]

            #Construct ABD matrix

            A_mat = np.zeros((3,3))

            for plyind in range(nplies):
                Q = get_Q(E1, E2, v12, G12)
                if ply_stat[plyind] == 2:  # Failed
                    Q = np.zeros((3,3))
                elif ply_stat[plyind] == 1: # Matrix failure
                    Q[1, 1] *= 0.15
                    Q[0, 1] *= 0.15
                    Q[1, 0] *= 0.15

                Q_trans = transform_Q(Q, stack[plyind])
                A_mat += Q_trans * t_ply

            # try statemnet used to avoid singular matrix problem breaking the code
            try:
                strain = np.linalg.solve(A_mat, [Nx, Ny, Ns])
            except:
                break

            for plyind in range(nplies):
                Q = get_Q(E1, E2, v12, G12)
                if ply_stat[plyind] == 2:  # Failed
                    Q = np.zeros((3,3))
                elif ply_stat[plyind] == 1: # IFF failure
                    Q[1, 1] *= 0.15
                    Q[0, 1] *= 0.15
                    Q[1, 0] *= 0.15
                stress_local = local_stress(Q, plyind, strain)

                # test stress against puck criterias

                if puck_FF(stress_local[0], stress_local[1]):
                    if FPF == False:
                        FPF = True
                        load_FPF.append([Nx, Ny, Ns])
                        strain_FPF.append(strain)
                        loaddirec_FPF.append(loaddirec)

                    ply_stat[plyind] = 2

                elif puck_IFF(stress_local[1], stress_local[2]):
                    if ply_stat[plyind] == 1: # matrix fails for the second time
                        ply_stat[plyind] = 2
                    else:
                        if FPF == False:
                            FPF = True
                            load_FPF.append([Nx, Ny, Ns])
                            strain_FPF.append(strain)
                            loaddirec_FPF.append(loaddirec)
                        ply_stat[plyind] = 1

            # record strain in every load increment
            strain_str.append(strain)

            if np.all(ply_stat == 2):
                load_LPF.append([Nx, Ny, Ns])
                # the strain at the moment when the laminate fails is very unstable due to ABD matrix singularity issues, therefore I've decided to use the strain from a couple of previous steps for the strain plot
                strain_LPF.append(strain_str[-5])
                loaddirec_LPF.append(loaddirec)
                break

    return np.array(load_FPF), np.array(load_LPF), np.array(strain_FPF), np.array(strain_LPF), np.array(loaddirec_FPF), np.array(loaddirec_LPF)



xyload_FPF, xyload_LPF, xystrain_FPF, xystrain_LPF, xyld_FPF, xyld_LPF = compute_fail_envelope("nxny")
ysload_FPF, ysload_LPF, ysstrain_FPF, ysstrain_LPF, ysld_FPF, ysld_LPF = compute_fail_envelope("nyns")

xyload_FPF, xyload_LPF = xyload_FPF/(1e6 * t_tot), xyload_LPF/(1e6 * t_tot)
ysload_FPF, ysload_LPF = ysload_FPF/(1e6 * t_tot), ysload_LPF/(1e6 * t_tot)

outfile = "failure_envelope_results.npz"

# saving the data into a file, so the plotting can be done separately without having to wait 20 minutes everytime
np.savez(
    outfile,

    xyload_FPF = xyload_FPF,
    xyload_LPF = xyload_LPF,
    xystrain_FPF = xystrain_FPF,
    xystrain_LPF = xystrain_LPF,
    xyld_FPF = xyld_FPF,
    xyld_LPF = xyld_LPF,

    ysload_FPF = ysload_FPF,
    ysload_LPF = ysload_LPF,
    ysstrain_FPF = ysstrain_FPF,
    ysstrain_LPF = ysstrain_LPF,
    ysld_FPF = ysld_FPF,
    ysld_LPF = ysld_LPF

)

print(f"Results exported to {outfile}")


'''
def plot_failure_envelope(mode, load_FPF, load_LPF):

    ind = 0
    xlab = "σₓ [MPa]"
    ylab = "σᵧ [MPa]"
    if mode == "nyns":
        ind = 1
        xlab = "σᵧ [MPa]"
        ylab = "γₓᵧ [MPa]"

    plt.figure(figsize=(10,10))

    if len(load_FPF) > 0:
        plt.fill(load_FPF[:,0 + ind]/1e3, load_FPF[:,1 + ind]/1e3,
                 color='blue', alpha=0.1)
        plt.plot(load_FPF[:,0 + ind]/1e3, load_FPF[:,1 + ind]/1e3, 'b--', label="First ply failure FPF")

    if len(load_LPF) > 0:
        plt.fill(load_LPF[:,0 + ind]/1e3, load_LPF[:,1 + ind]/1e3,
                 color='red', alpha=0.1)
        plt.plot(load_LPF[:,0 + ind]/1e3, load_LPF[:,1 + ind]/1e3, 'r-', label="Last ply failure LPF")

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plt.show()


def plot_strain(mode, strain_FPF, strain_LPF, load_FPF, load_LPF):

    def compute_angle(load, mode):

        if mode == "nxny":
            ang = np.rad2deg(np.arctan2(load[:, 1], load[:, 0]))  # Ny / Nx

        elif mode == "nyns":
            ang = np.rad2deg(np.arctan2(load[:, 2], load[:, 1]))  # Ns / Ny

        return (ang + 360) % 360

    ang_FPF = compute_angle(load_FPF, mode)
    ang_LPF = compute_angle(load_LPF, mode)

    order_FPF = np.argsort(ang_FPF)
    order_LPF = np.argsort(ang_LPF)

    ang_FPF = ang_FPF[order_FPF]
    strain_FPF = strain_FPF[order_FPF]

    ang_LPF = ang_LPF[order_LPF]
    strain_LPF = strain_LPF[order_LPF]

    strain_sym = ["εₓ", "εᵧ", "γₓᵧ"]

    for i in range(3):

        plt.figure(figsize=(10,10))

        if len(strain_FPF):
            plt.plot(ang_FPF, strain_FPF[:,i], 'b--', label="First ply failure FPF")

        if len(strain_LPF):
            plt.plot(ang_LPF, strain_LPF[:,i], 'r-', label="Last ply failure LPF")

        plt.xlabel("load angle [deg]")
        plt.ylabel(f"{strain_sym[i]} [ε]")
        plt.xlim(0, 360)
        plt.grid(True)
        plt.legend()

        plt.show()
'''