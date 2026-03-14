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
Q11, Q12, Q22, Q66, Q0 = local_elastic_property(E1, E2, G12, v12)            
Q_overall = Q_transformed(Q11, Q12, Q22, Q66, angle_arr_deg) 
ABD_matrix = ABD_Calc(Q_overall, zcoord_arr) 

ABD_inv = np.linalg.inv(ABD_matrix)
strain_global_vec = np.dot(ABD_inv, N_vec)
layer_strain_global = Strain_ply_calculation(strain_global_vec, zcoord_arr)

layer_strain_lst = []
layer_stress_lst = []
for plynum in range(len(angle_arr_rad)):
    layer_strain_lst.append(np.dot(strainTOstrain_trans(angle_arr_rad[plynum]), layer_strain_global[plynum]))
    layer_stress_lst.append(np.dot(Q0, layer_strain_lst[plynum]))

print("Strain in each layer:")
for i in range(len(angle_arr_rad)):
    print(f"Layer {i+1} (angle {angle_arr_deg[i]} degrees): strain x {layer_strain_lst[i][0]*10e-6} [με]| strain y {layer_strain_lst[i][1]*10e-6} [με]| strain xy {layer_strain_lst[i][2]*10e-6} [με]")   

print("\nStress in each layer:")
for i in range(len(angle_arr_rad)):
    print(f"Layer {i+1} (angle {angle_arr_deg[i]} degrees): stress x {layer_stress_lst[i][0]*10e-6} [MPa]| stress y {layer_stress_lst[i][1]*10e-6} [MPa]| stress xy {layer_stress_lst[i][2]*10e-9} [GPa]")


