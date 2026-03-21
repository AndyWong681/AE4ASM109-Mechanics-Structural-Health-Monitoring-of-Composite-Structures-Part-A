import numpy as np
import matplotlib.pyplot as plt
import math
from utils import *
from UD_constants import * 

#For the laminate [0/±45/90/30]3s plot the biaxial stress failure envelopes for 
#axial loading Nx-Ny utilizing Puck failure criterion.

t = 0.125e-3 # m
angle_arr_deg = np.array([0, 45, -45, 90, 30])
angle_arr_deg = 3*(angle_arr_deg + angle_arr_deg [::-1])

zcoord_arr = zcoordinate(angle_arr_rad, t)    
Q11, Q12, Q22, Q66, Q0 = local_elastic_property(E1, E2, G12, v12)   
Q_overall = Q_transformed(Q11, Q12, Q22, Q66, angle_arr_deg) 
ABD_matrix = ABD_Calc(Q_overall, zcoord_arr) 

# Define the range of Nx and Ny values
Nx = np.linspace(-1e6, 1e6, 100)
Ny = np.linspace(-1e6, 1e6, 100)
Nx_grid, Ny_grid = np.meshgrid(Nx, Ny)

#or the laminate [0/±45/90/30]3s plot the biaxial stress failure envelopes for
#i. axial loading Nx-Ny utilizing Puck failure criterion.
#Department of Aerospace Structures & Materials Dr. Dimitrios Zarouchas (d.zarouchas@tudelft.nl)
#ii. axial loading Ny-Ns utilizing Puck failure criterion
#You should indicate the FPF and LPF in each plot and for each loading ratio.
#Report the global failure strains as well (for each loading step where failure occurs)


#For the analysis of the LPF, you should use the following sudden degradation rule:
#You zero all the elastic properties of the failed lamina if Tension or Compression FF is
#observed, otherwise you should use 0.15 degradation factor for the transverse elastic
#properties. If the same lamina fails for 2nd time (no matter the failure mode), zero all the
#properties.
#Furthermore, use the following values of the compression strengths
#Xc=1480 MPa
#Yc=220 MPa


#Puck failure criterion for axial loading Nx-Ny
def puck_failure_criterion_FF():
    
    if sigma_1 >= 0:  # Tension
        X = 1500e6  # Tension strength in Pa
    else:  # Compression
        X = -1480e6  # Compression strength in Pa

    FF = 1 / X * (sigma_1 - (v21 - v21f * 1.1 * E1 / E1_f) * (sigma_2))

    if FF >= 1:
        return True # Failure occurs
    else:
        return False  # No failure



def puck_failure_criterion_IFF(sigma_2, sigma_12, X_c, X_t, S):
    
    sigma_23 = sigma_12 / (2 * 0.2) * (np.sqrt(1 + 2 * 0.2 * X_c / S) - 1)
    p_23 =  0.2 * sigma_23 / abs(S)
    r = abs(sigma_2 / sigma_12)
    criterion = sigma_23 / abs(S)

    IFF_A = 0
    IFF_B = 0
    IFF_C = 0

    if sigma_2 > 0:
        IFF_A = np.sqrt((sigma_12 / S) ** 2 + (1 - 0.3 * X_t / S) ** 2 * (sigma_2 / X_t) ** 2) + 0.3 * sigma_2 / S
    elif sigma_2 < 0 and r <= criterion:
        IFF_B = 1 / S * np.sqrt(sigma_12 ** 2 + (0.2 * sigma_2) ** 2) + 0.2 * sigma_2 
    else:
        IFF_C = ((sigma_12 / (2 * (1 + p_23 * S))) ** 2 + (sigma_2 / X_c) ** 2 ) * X_c / (-sigma_2)
    

    if IFF_A > 1 or IFF_B > 1 or IFF_C > 1:
        return True  #True if failure occurs
    else:
        return False  
    


def update_properties(lamina_index, failure_mode):

    if failure_mode == 'FF':
        # Zero all properties for the failed lamina
        Q_overall[lamina_index] = np.zeros_like(Q_overall[lamina_index])
    elif failure_mode == 'IFF':
        # Apply 0.15 degradation factor to transverse properties
        Q_overall[lamina_index][[0,1],[1,0],[1,1]] *= 0.15 # Assuming transverse properties are at indices 1 and 2
    


def laminate_iter(): 
    Failure = False

    ply_mat_fail = np.zeros_like(angle_arr_deg, dtype=bool)  # To track the number of failures for each lamina
    ply_fail = np.zeros_like(angle_arr_deg, dtype=bool)  # To track if a lamina has failed at least once

    while Failure == False:
        # Implement criteria for each lamina and update the properties based on the degradation rule
        for i in range(len(angle_arr_deg)):
            # Calculate the stresses for the current lamina
            sigma_1, sigma_2, sigma_12 = calculate_stresses(ABD_matrix, Nx_grid, Ny_grid, i)
            
            # Check for failure using Puck's criteria
            if puck_failure_criterion_FF(sigma_1, sigma_2):
                # Update properties based on degradation rule
                update_properties(i, 'FF')
                
            elif puck_failure_criterion_IFF(sigma_2, sigma_12):
                ply_mat_fail[i] = True
                update_properties(i, 'IFF')

        if np.all(ply_fail == True):
            Failure = True  


