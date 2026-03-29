import numpy as np
import matplotlib.pyplot as plt
from utils import *
from UD_constants import *
from mpl_toolkits.mplot3d import Axes3D


def compute_engineering_constants():
    """
    Compute 10 engineering constants over a grid of (phi, theta).

    Returns
    -------
    phi_values : 1D ndarray
        Outer-loop angle values.
    theta_values : 1D ndarray
        Inner-loop angle values.
    results : dict
        Dictionary containing 10 arrays, each of shape (n_phi, n_theta).
        Row index -> phi
        Col index -> theta
    """
    t = 0.125e-3  # plate thickness [m]

    phi_values = np.arange(-90, 91, 1)      # -90, -89, ..., 90
    theta_values = np.arange(-90, 91, 1)    # -90, -89, ..., 90

    results = {
        "Ex": [],
        "Ey": [],
        "vxy": [],
        "vyx": [],
        "Gxy": [],
        "Ex_f": [],
        "Ey_f": [],
        "vxy_f": [],
        "vyx_f": [],
        "Gxy_f": [],
    }

    Q11, Q12, Q22, Q66, _ = local_elastic_property(E1, E2, G12, v12)

    for phi in phi_values:
        temp = {key: [] for key in results.keys()}

        for theta in theta_values:
            angle_arr = np.array([
                theta, -theta, phi, phi, phi, phi, phi, phi, -theta, theta,
                theta, -theta, phi, phi, phi, phi, phi, phi, -theta, theta
            ])

            zcoord_arr = zcoordinate(angle_arr, t)
            Q_overall = Q_transformed(Q11, Q12, Q22, Q66, angle_arr)
            ABD_matrix = ABD_Calc(Q_overall, zcoord_arr)

            Ex, Ey, vxy, vyx, Gxy, Ex_f, Ey_f, vxy_f, vyx_f, Gxy_f = Equvalent_properties(
                ABD_matrix, zcoord_arr
            )

            temp["Ex"].append(Ex)
            temp["Ey"].append(Ey)
            temp["vxy"].append(vxy)
            temp["vyx"].append(vyx)
            temp["Gxy"].append(Gxy)
            temp["Ex_f"].append(Ex_f)
            temp["Ey_f"].append(Ey_f)
            temp["vxy_f"].append(vxy_f)
            temp["vyx_f"].append(vyx_f)
            temp["Gxy_f"].append(Gxy_f)

        for key in results.keys():
            results[key].append(temp[key])

    for key in results.keys():
        results[key] = np.array(results[key])

    return phi_values, theta_values, results


def plot_surface(phi_values, theta_values, Z, zlabel="", title="", convert_to_GPa=False,
                 elev=28, azim=-55, save_path=None):
    """
    Improved 3D surface plot.
    Z must have shape (n_phi, n_theta).

    Parameters
    ----------
    phi_values : ndarray
    theta_values : ndarray
    Z : 2D ndarray
        shape = (n_phi, n_theta)
    zlabel : str
    title : str
    convert_to_GPa : bool
        If True, divide Z by 1e9
    elev, azim : float
        3D view angles
    save_path : str or None
        If provided, save figure to this path
    """
    import matplotlib.ticker as mticker
    from matplotlib import cm

    X, Y = np.meshgrid(theta_values, phi_values)
    Z_plot = Z / 1e9 if convert_to_GPa else Z

    # Optional: make the 3D figure look more polished
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig = plt.figure(figsize=(11.5, 8.0), dpi=160)
    ax = fig.add_subplot(111, projection='3d')

    # smoother-looking surface
    surf = ax.plot_surface(
        X, Y, Z_plot,
        cmap=cm.viridis,
        rstride=2,
        cstride=2,
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=0.96
    )

    # Better viewing angle
    ax.view_init(elev=elev, azim=azim)

    # Labels
    ax.set_xlabel(r'$\theta$ [$^\circ$]', labelpad=12)
    ax.set_ylabel(r'$\phi$ [$^\circ$]', labelpad=12)
    ax.set_zlabel(zlabel, labelpad=10)
    ax.set_title(title, pad=18)

    # Ticks
    ax.set_xticks(np.arange(-90, 91, 30))
    ax.set_yticks(np.arange(-90, 91, 30))

    # Cleaner panes
    ax.xaxis.pane.set_alpha(0.08)
    ax.yaxis.pane.set_alpha(0.08)
    ax.zaxis.pane.set_alpha(0.04)

    # Light grid
    ax.grid(True, alpha=0.25)

    # Better aspect ratio
    try:
        ax.set_box_aspect((1.25, 1.25, 0.8))
    except Exception:
        pass

    # nicer z tick formatting
    ax.zaxis.set_major_locator(mticker.MaxNLocator(6))

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.08, aspect=24)
    cbar.set_label(zlabel)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_contour(phi_values, theta_values, Z, colorbar_label="", title=""):
    """
    Filled contour plot.
    Z must have shape (n_phi, n_theta).
    """
    X, Y = np.meshgrid(theta_values, phi_values)

    plt.figure(figsize=(9, 6))
    contour = plt.contourf(X, Y, Z, levels=40, cmap='viridis')
    cbar = plt.colorbar(contour)
    if colorbar_label:
        cbar.set_label(colorbar_label)

    plt.xlabel(r'$\theta$ [$^\circ$]')
    plt.ylabel(r'$\phi$ [$^\circ$]')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_variable_group(phi_values, theta_values, plot_data, theta_to_plot=None, fig_title=None):
    """
    Plot one group of variables using subplots.

    Parameters
    ----------
    phi_values : ndarray
    theta_values : ndarray
    plot_data : list of tuples
        [(data_array, title), ...]
    theta_to_plot : list
        theta values to show in legend
    fig_title : str
        optional overall figure title
    """
    if theta_to_plot is None:
        theta_to_plot = [0, 15, 30, 45, 60]

    theta_indices = []
    for th in theta_to_plot:
        idx = np.where(theta_values == th)[0]
        if len(idx) == 0:
            raise ValueError(f"Theta value {th} not found in theta_values.")
        theta_indices.append(idx[0])

    nplots = len(plot_data)

    # Layout for 5 plots
    if nplots == 5:
        nrows, ncols = 3, 2
    else:
        ncols = 3
        nrows = int(np.ceil(nplots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows))
    axes = np.array(axes).flatten()

    for i, (data, title) in enumerate(plot_data):
        ax = axes[i]

        for th, idx in zip(theta_to_plot, theta_indices):
            ax.plot(phi_values, data[:, idx], linewidth=1.8, label=fr'$\theta={th}^\circ$')

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r'$\phi$ [$^\circ$]')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=8)

    for j in range(nplots, len(axes)):
        axes[j].axis('off')

    # if fig_title is not None:
    #     fig.suptitle(fig_title, fontsize=14)
    #     plt.tight_layout(rect=[0, 0, 1, 0.96])
    # else:
    
    plt.tight_layout()

    plt.show()


def plot_all_subplots(phi_values, theta_values, results, theta_to_plot=None):
    """
    Plot in-plane and flexural engineering constants separately.
    """
    if theta_to_plot is None:
        theta_to_plot = [-60, -45, -30, -15, 0, 15, 30, 45, 60]

    # In-plane variables
    inplane_data = [
        (results["Ex"]   / 1e9, r'In-plane $E_x$ [GPa]'),
        (results["Ey"]   / 1e9, r'In-plane $E_y$ [GPa]'),
        (results["Gxy"]  / 1e9, r'In-plane $G_{xy}$ [GPa]'),
        (results["vxy"],        r'In-plane $\nu_{xy}$ [-]'),
        (results["vyx"],        r'In-plane $\nu_{yx}$ [-]'),
    ]

    # Flexural variables
    flexural_data = [
        (results["Ex_f"]   / 1e9, r'Flexural $E_{x_f}$ [GPa]'),
        (results["Ey_f"]   / 1e9, r'Flexural $E_{y_f}$ [GPa]'),
        (results["Gxy_f"]  / 1e9, r'Flexural $G_{xy_f}$ [GPa]'),
        (results["vxy_f"],        r'Flexural $\nu_{xy_f}$ [-]'),
        (results["vyx_f"],        r'Flexural $\nu_{yx_f}$ [-]'),
    ]

    plot_variable_group(
        phi_values,
        theta_values,
        inplane_data,
        theta_to_plot=theta_to_plot,
        fig_title="In-plane engineering constants"
    )

    plot_variable_group(
        phi_values,
        theta_values,
        flexural_data,
        theta_to_plot=theta_to_plot,
        fig_title="Flexural engineering constants"
    )


def plot_single_constant(phi_values, theta_values, results, constant_name,
                         theta_to_plot=None, title=None, ylabel=None,
                         convert_to_GPa=True, save_path=None):
    """
    Plot a single engineering constant versus phi for selected theta values.
    Optimized for exporting figures into LaTeX.

    Parameters
    ----------
    phi_values : ndarray
    theta_values : ndarray
    results : dict
        Output of compute_engineing_constants()
    constant_name : str
        One of:
        "Ex", "Ey", "Gxy", "vxy", "vyx",
        "Ex_f", "Ey_f", "Gxy_f", "vxy_f", "vyx_f"
    theta_to_plot : list
        Example: [-45, -30, 0, 30, 45]
    title : str
        Optional custom title
    ylabel : str
        Optional custom y-axis label
    convert_to_GPa : bool
        If True, stiffness/modulus quantities are divided by 1e9
    save_path : str or None
        If provided, save figure to this path
    """
    valid_constants = ["Ex", "Ey", "Gxy", "vxy", "vyx",
                       "Ex_f", "Ey_f", "Gxy_f", "vxy_f", "vyx_f"]

    if constant_name not in valid_constants:
        raise ValueError(f"constant_name must be one of {valid_constants}")

    if theta_to_plot is None:
        theta_to_plot = [-45, -30, 0, 30, 45]

    data = results[constant_name].copy()

    stiffness_constants = ["Ex", "Ey", "Gxy", "Ex_f", "Ey_f", "Gxy_f"]

    if constant_name in stiffness_constants and convert_to_GPa:
        data = data / 1e9
        default_unit = "[GPa]"
    elif constant_name in stiffness_constants:
        default_unit = "[Pa]"
    else:
        default_unit = "[-]"

    default_labels = {
        "Ex": r"$E_x$",
        "Ey": r"$E_y$",
        "Gxy": r"$G_{xy}$",
        "vxy": r"$\nu_{xy}$",
        "vyx": r"$\nu_{yx}$",
        "Ex_f": r"$E_{x_f}$",
        "Ey_f": r"$E_{y_f}$",
        "Gxy_f": r"$G_{xy_f}$",
        "vxy_f": r"$\nu_{xy_f}$",
        "vyx_f": r"$\nu_{yx_f}$"
    }

    if ylabel is None:
        ylabel = f"{default_labels[constant_name]} {default_unit}"

    if title is None:
        title = default_labels[constant_name]

    theta_indices = []
    for th in theta_to_plot:
        idx = np.where(theta_values == th)[0]
        if len(idx) == 0:
            raise ValueError(f"Theta value {th} not found in theta_values.")
        theta_indices.append(idx[0])

    # Global style tuning for LaTeX export
    plt.rcParams.update({
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    })

    fig = plt.figure(figsize=(10, 7), dpi=200)

    for th, idx in zip(theta_to_plot, theta_indices):
        plt.plot(phi_values, data[:, idx], linewidth=2.2, label=fr'$\theta={th}^\circ$')

    plt.xlabel(r'$\phi$ [$^\circ$]', fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    # plt.title(title, fontsize=20, pad=10)
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.legend(fontsize=12, loc='best')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_all_constants_individually(phi_values, theta_values, results, theta_to_plot=None, save_folder=None):
    """
    Plot all 10 engineering constants one by one using large single plots.
    Optionally save them automatically with assignment-style filenames.
    """
    import os

    if theta_to_plot is None:
        theta_to_plot = [0, 15, 30, 45, 60, 75, 90]

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

    constants_to_plot = [
        ("Ex",    r"In-plane $E_x$",        r"$E_x$ [GPa]", True,  "1a_Ex.png"),
        ("Ey",    r"In-plane $E_y$",        r"$E_y$ [GPa]", True,  "1a_Ey.png"),
        ("Gxy",   r"In-plane $G_{xy}$",     r"$G_{xy}$ [GPa]", True,  "1a_Gxy.png"),
        ("vxy",   r"In-plane $\nu_{xy}$",   r"$\nu_{xy}$ [-]", False, "1a_vxy.png"),
        ("vyx",   r"In-plane $\nu_{yx}$",   r"$\nu_{yx}$ [-]", False, "1a_vyx.png"),
        ("Ex_f",  r"Flexural $E_{x_f}$",    r"$E_{x_f}$ [GPa]", True,  "1a_Exf.png"),
        ("Ey_f",  r"Flexural $E_{y_f}$",    r"$E_{y_f}$ [GPa]", True,  "1a_Eyf.png"),
        ("Gxy_f", r"Flexural $G_{xy_f}$",   r"$G_{xy_f}$ [GPa]", True,  "1a_Gxyf.png"),
        ("vxy_f", r"Flexural $\nu_{xy_f}$", r"$\nu_{xy_f}$ [-]", False, "1a_vxyf.png"),
        ("vyx_f", r"Flexural $\nu_{yx_f}$", r"$\nu_{yx_f}$ [-]", False, "1a_vyxf.png"),
    ]

    for constant_name, title, ylabel, convert_to_GPa, filename in constants_to_plot:
        save_path = None if save_folder is None else os.path.join(save_folder, filename)

        plot_single_constant(
            phi_values,
            theta_values,
            results,
            constant_name=constant_name,
            theta_to_plot=theta_to_plot,
            # title=title,
            ylabel=ylabel,
            convert_to_GPa=convert_to_GPa,
            save_path=save_path
        )


def main(plot_mode="subplots"):
    """
    plot_mode options:
    - "subplots"
    - "surface"
    - "contour"
    """
    phi_values, theta_values, results = compute_engineering_constants()

    if plot_mode == "subplots":
        plot_all_subplots(
            phi_values,
            theta_values,
            results,
            theta_to_plot=[0, 15, 30, 45, 60, 75, 90]
        )

    elif plot_mode == "surface":
        plot_surface(
            phi_values,
            theta_values,
            results["Ex"],
            zlabel=r"$E_x$ [GPa]",
            # title=r"3D variation of in-plane modulus $E_x$ with $\theta$ and $\phi$",
            convert_to_GPa=True
        )

    elif plot_mode == "contour":
        plot_contour(
            phi_values,
            theta_values,
            results["Ex"],
            colorbar_label=r"$E_x$ [Pa]",
            title=r"Contour of in-plane engineering constant $E_x$"
        )
    
    elif plot_mode == "single":
        # Example for symmetry explanation:
        # choose both negative and positive theta values
        plot_single_constant(
            phi_values,
            theta_values,
            results,
            constant_name="Ex",
            theta_to_plot=[-60, -45, -30, -15, 0, 15, 30, 45, 60],
            # title=r"In-plane modulus $E_x$",
            ylabel=r"$E_x$ [GPa]",
            convert_to_GPa=True
        )

    elif plot_mode == "individual":
        plot_all_constants_individually(
            phi_values,
            theta_values,
            results,
            theta_to_plot=[0, 15, 30, 45, 60, 75, 90],
            save_folder="Figures"
    )

    else:
        raise ValueError("plot_mode must be one of: 'subplots', 'surface', 'contour', 'single', 'individual'")


if __name__ == "__main__":
    main(plot_mode="single")
    print("sfgg done")
