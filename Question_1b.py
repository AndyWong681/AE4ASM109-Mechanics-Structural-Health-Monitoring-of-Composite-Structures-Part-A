import numpy as np
import matplotlib.pyplot as plt
import math
from utils import *
from UD_constants import * 

angle_arr_deg = np.array([0, 45, -45, -45, 90, -60, 30, 0])
angle_arr_rad = np.deg2rad(angle_arr_deg) # convert to radians
Nx = 3e2 # N/m
Ny = 0 # N/m
Nxy = 25 # N/m
Mx = 0 # N
My = 18e3 # N
Mxy = 0 # N
t = 0.125e-3 # m

N_vec = Applied_Loading(Nx, Ny, Nxy, Mx, My, Mxy)

zcoord_arr = zcoordinate(angle_arr_rad, t)   
zcoordinates_ply_group = zcoordinates_ply_group1b(angle_arr_rad, t, zcoordinate) # z-coordinates of each ply group (for strain calculation)
Q11, Q12, Q22, Q66, Q0 = local_elastic_property(E1, E2, G12, v12)            
Q_overall = Q_transformed(Q11, Q12, Q22, Q66, angle_arr_deg) 
ABD_matrix = ABD_Calc(Q_overall, zcoord_arr) 


ABD_inv = np.linalg.inv(ABD_matrix)
strain_global_vec = np.dot(ABD_inv, N_vec)
layer_strain_global = Strain_ply_calculation_1b(strain_global_vec, zcoordinates_ply_group)
strain_local_lst_large = Global_to_local_strain1b(layer_strain_global, angle_arr_deg)  #the gloval_to_local_strain1b require degree angle array
stress_local_lst_large = Stress_ply_calculation_1b(strain_local_lst_large, Q0)



# Plot strain vs thickness for each component
def plot_strain_x_vs_thickness(strain_local_lst_large, zcoordinates_ply_group, angle_arr_deg):
    plt.figure(figsize=(10, 6))
    first_z_start = None
    last_z_end = None
    for i in range(len(angle_arr_deg)):
        strain_x_vals = strain_local_lst_large[i][:, 0] * 1e6  # Convert to microstrains
        z_vals = zcoordinates_ply_group[i] * 1e3  # Convert to mm
        plt.plot(strain_x_vals, z_vals, label=f'Ply {i+1} ({angle_arr_deg[i]}°)', linewidth=2)
        # Store first and last z values
        if i == 0:
            first_z_start = z_vals[0]
        if i == len(angle_arr_deg) - 1:
            last_z_end = z_vals[-1]
        # Connect first ply start to 0
        if i == 0:
            plt.plot([0, strain_x_vals[0]], [z_vals[0], z_vals[0]], 'k--', linewidth=1, alpha=0.7)
        # Add dashed line connecting to next ply
        if i < len(angle_arr_deg) - 1:
            next_strain = strain_local_lst_large[i+1][0, 0] * 1e6
            current_z = z_vals[-1]
            next_z = zcoordinates_ply_group[i+1][0] * 1e3
            plt.plot([strain_x_vals[-1], next_strain], [current_z, next_z], 'k--', linewidth=1, alpha=0.5)
        # Connect last ply end to 0
        if i == len(angle_arr_deg) - 1:
            plt.plot([0, strain_x_vals[-1]], [z_vals[-1], z_vals[-1]], 'k--', linewidth=1, alpha=0.7)
    # Add vertical line connecting first start to last end at x=0
    if first_z_start is not None and last_z_end is not None:
        plt.axline((0, first_z_start), (0, last_z_end), color='k', linewidth=1, alpha=0.7)
    plt.xlabel('Strain X (με)', fontsize=12)
    plt.ylabel('Thickness (mm)', fontsize=12)
    plt.title('Strain X vs Thickness', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_strain_y_vs_thickness(strain_local_lst_large, zcoordinates_ply_group, angle_arr_deg):
    plt.figure(figsize=(10, 6))
    first_z_start = None
    last_z_end = None
    for i in range(len(angle_arr_deg)):
        strain_y_vals = strain_local_lst_large[i][:, 1] * 1e6  # Convert to microstrains
        z_vals = zcoordinates_ply_group[i] * 1e3  # Convert to mm
        plt.plot(strain_y_vals, z_vals, label=f'Ply {i+1} ({angle_arr_deg[i]}°)', linewidth=2)
        # Store first and last z values
        if i == 0:
            first_z_start = z_vals[0]
        if i == len(angle_arr_deg) - 1:
            last_z_end = z_vals[-1]
        # Connect first ply start to 0
        if i == 0:
            plt.plot([0, strain_y_vals[0]], [z_vals[0], z_vals[0]], 'k--', linewidth=1, alpha=0.7)
        # Add dashed line connecting to next ply
        if i < len(angle_arr_deg) - 1:
            next_strain = strain_local_lst_large[i+1][0, 1] * 1e6
            current_z = z_vals[-1]
            next_z = zcoordinates_ply_group[i+1][0] * 1e3
            plt.plot([strain_y_vals[-1], next_strain], [current_z, next_z], 'k--', linewidth=1, alpha=0.5)
        # Connect last ply end to 0
        if i == len(angle_arr_deg) - 1:
            plt.plot([0, strain_y_vals[-1]], [z_vals[-1], z_vals[-1]], 'k--', linewidth=1, alpha=0.7)
    # Add vertical line connecting first start to last end at x=0
    if first_z_start is not None and last_z_end is not None:
        plt.axline((0, first_z_start), (0, last_z_end), color='k', linewidth=1, alpha=0.7)
    plt.xlabel('Strain Y (με)', fontsize=12)
    plt.ylabel('Thickness (mm)', fontsize=12)
    plt.title('Strain Y vs Thickness', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_strain_xy_vs_thickness(strain_local_lst_large, zcoordinates_ply_group, angle_arr_deg):
    plt.figure(figsize=(10, 6))
    first_z_start = None
    last_z_end = None
    for i in range(len(angle_arr_deg)):
        strain_xy_vals = strain_local_lst_large[i][:, 2] * 1e6  # Convert to microstrains
        z_vals = zcoordinates_ply_group[i] * 1e3  # Convert to mm
        plt.plot(strain_xy_vals, z_vals, label=f'Ply {i+1} ({angle_arr_deg[i]}°)', linewidth=2)
        # Store first and last z values
        if i == 0:
            first_z_start = z_vals[0]
        if i == len(angle_arr_deg) - 1:
            last_z_end = z_vals[-1]
        # Connect first ply start to 0
        if i == 0:
            plt.plot([0, strain_xy_vals[0]], [z_vals[0], z_vals[0]], 'k--', linewidth=1, alpha=0.7)
        # Add dashed line connecting to next ply
        if i < len(angle_arr_deg) - 1:
            next_strain = strain_local_lst_large[i+1][0, 2] * 1e6
            current_z = z_vals[-1]
            next_z = zcoordinates_ply_group[i+1][0] * 1e3
            plt.plot([strain_xy_vals[-1], next_strain], [current_z, next_z], 'k--', linewidth=1, alpha=0.5)
        # Connect last ply end to 0
        if i == len(angle_arr_deg) - 1:
            plt.plot([0, strain_xy_vals[-1]], [z_vals[-1], z_vals[-1]], 'k--', linewidth=1, alpha=0.7)
    # Add vertical line connecting first start to last end at x=0
    if first_z_start is not None and last_z_end is not None:
        plt.axline((0, first_z_start), (0, last_z_end), color='k', linewidth=1, alpha=0.7)
    plt.xlabel('Strain XY (με)', fontsize=12)
    plt.ylabel('Thickness (mm)', fontsize=12)
    plt.title('Strain XY vs Thickness', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Plot stress vs thickness for each component
def plot_stress_x_vs_thickness(stress_local_lst_large, zcoordinates_ply_group, angle_arr_deg):
    plt.figure(figsize=(10, 6))
    first_z_start = None
    last_z_end = None
    for i in range(len(angle_arr_deg)):
        stress_x_vals = stress_local_lst_large[i][:, 0] / 1e6  # Convert to MPa
        z_vals = zcoordinates_ply_group[i] * 1e3  # Convert to mm
        plt.plot(stress_x_vals, z_vals, label=f'Ply {i+1} ({angle_arr_deg[i]}°)', linewidth=2)
        # Store first and last z values
        if i == 0:
            first_z_start = z_vals[0]
        if i == len(angle_arr_deg) - 1:
            last_z_end = z_vals[-1]
        # Connect first ply start to 0
        if i == 0:
            plt.plot([0, stress_x_vals[0]], [z_vals[0], z_vals[0]], 'k--', linewidth=1, alpha=0.7)
        # Add dashed line connecting to next ply
        if i < len(angle_arr_deg) - 1:
            next_stress = stress_local_lst_large[i+1][0, 0] / 1e6
            current_z = z_vals[-1]
            next_z = zcoordinates_ply_group[i+1][0] * 1e3
            plt.plot([stress_x_vals[-1], next_stress], [current_z, next_z], 'k--', linewidth=1, alpha=0.5)
        # Connect last ply end to 0
        if i == len(angle_arr_deg) - 1:
            plt.plot([0, stress_x_vals[-1]], [z_vals[-1], z_vals[-1]], 'k--', linewidth=1, alpha=0.7)
    # Add vertical line connecting first start to last end at x=0
    if first_z_start is not None and last_z_end is not None:
        plt.axline((0, first_z_start), (0, last_z_end), color='k', linewidth=1, alpha=0.7)
    plt.xlabel('Stress X (MPa)', fontsize=12)
    plt.ylabel('Thickness (mm)', fontsize=12)
    plt.title('Stress X vs Thickness', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_stress_y_vs_thickness(stress_local_lst_large, zcoordinates_ply_group, angle_arr_deg):
    plt.figure(figsize=(10, 6))
    first_z_start = None
    last_z_end = None
    for i in range(len(angle_arr_deg)):
        stress_y_vals = stress_local_lst_large[i][:, 1] / 1e6  # Convert to MPa
        z_vals = zcoordinates_ply_group[i] * 1e3  # Convert to mm
        plt.plot(stress_y_vals, z_vals, label=f'Ply {i+1} ({angle_arr_deg[i]}°)', linewidth=2)
        # Store first and last z values
        if i == 0:
            first_z_start = z_vals[0]
        if i == len(angle_arr_deg) - 1:
            last_z_end = z_vals[-1]
        # Connect first ply start to 0
        if i == 0:
            plt.plot([0, stress_y_vals[0]], [z_vals[0], z_vals[0]], 'k--', linewidth=1, alpha=0.7)
        # Add dashed line connecting to next ply
        if i < len(angle_arr_deg) - 1:
            next_stress = stress_local_lst_large[i+1][0, 1] / 1e6
            current_z = z_vals[-1]
            next_z = zcoordinates_ply_group[i+1][0] * 1e3
            plt.plot([stress_y_vals[-1], next_stress], [current_z, next_z], 'k--', linewidth=1, alpha=0.5)
        # Connect last ply end to 0
        if i == len(angle_arr_deg) - 1:
            plt.plot([0, stress_y_vals[-1]], [z_vals[-1], z_vals[-1]], 'k--', linewidth=1, alpha=0.7)
    # Add vertical line connecting first start to last end at x=0
    if first_z_start is not None and last_z_end is not None:
        plt.axline((0, first_z_start), (0, last_z_end), color='k', linewidth=1, alpha=0.7)
    plt.xlabel('Stress Y (MPa)', fontsize=12)
    plt.ylabel('Thickness (mm)', fontsize=12)
    plt.title('Stress Y vs Thickness', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_stress_xy_vs_thickness(stress_local_lst_large, zcoordinates_ply_group, angle_arr_deg):
    plt.figure(figsize=(10, 6))
    first_z_start = None
    last_z_end = None
    for i in range(len(angle_arr_deg)):
        stress_xy_vals = stress_local_lst_large[i][:, 2] / 1e6  # Convert to MPa
        z_vals = zcoordinates_ply_group[i] * 1e3  # Convert to mm
        plt.plot(stress_xy_vals, z_vals, label=f'Ply {i+1} ({angle_arr_deg[i]}°)', linewidth=2)
        # Store first and last z values
        if i == 0:
            first_z_start = z_vals[0]
        if i == len(angle_arr_deg) - 1:
            last_z_end = z_vals[-1]
        # Connect first ply start to 0
        if i == 0:
            plt.plot([0, stress_xy_vals[0]], [z_vals[0], z_vals[0]], 'k--', linewidth=1, alpha=0.7)
        # Add dashed line connecting to next ply
        if i < len(angle_arr_deg) - 1:
            next_stress = stress_local_lst_large[i+1][0, 2] / 1e6
            current_z = z_vals[-1]
            next_z = zcoordinates_ply_group[i+1][0] * 1e3
            plt.plot([stress_xy_vals[-1], next_stress], [current_z, next_z], 'k--', linewidth=1, alpha=0.5)
        # Connect last ply end to 0
        if i == len(angle_arr_deg) - 1:
            plt.plot([0, stress_xy_vals[-1]], [z_vals[-1], z_vals[-1]], 'k--', linewidth=1, alpha=0.7)
    # Add vertical line connecting first start to last end at x=0
    if first_z_start is not None and last_z_end is not None:
        plt.axline((0, first_z_start), (0, last_z_end), color='k', linewidth=1, alpha=0.7)
    plt.xlabel('Stress XY (MPa)', fontsize=12)
    plt.ylabel('Thickness (mm)', fontsize=12)
    plt.title('Stress XY vs Thickness', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



# Create all 6 plots
plot_strain_x_vs_thickness(strain_local_lst_large, zcoordinates_ply_group, angle_arr_deg)
plot_strain_y_vs_thickness(strain_local_lst_large, zcoordinates_ply_group, angle_arr_deg)
plot_strain_xy_vs_thickness(strain_local_lst_large, zcoordinates_ply_group, angle_arr_deg)
plot_stress_x_vs_thickness(stress_local_lst_large, zcoordinates_ply_group, angle_arr_deg)
plot_stress_y_vs_thickness(stress_local_lst_large, zcoordinates_ply_group, angle_arr_deg)
plot_stress_xy_vs_thickness(stress_local_lst_large, zcoordinates_ply_group, angle_arr_deg)

