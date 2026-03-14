import numpy as np
import matplotlib.pyplot as plt
import math
from utils import *
from UD_constants import * 
from mpl_toolkits.mplot3d import Axes3D

def main():
    # engineering constants should be 3d arrays 

    t = 0.125e-3 # in meters, thickness of the plate
    Ex_arr = []
    Ey_arr = []
    vxy_arr = []
    vyx_arr = []
    Gxy_arr = []
    
    Ex_f_arr = []
    Ey_f_arr = []
    vxy_f_arr = []
    vyx_f_arr = []
    Gxy_f_arr = []
    
    Q11, Q12, Q22, Q66, _ = local_elastic_property(E1, E2, G12, v12)

    for phi in range(-90, 91, 10): 
        Ex_temp = []
        Ey_temp = []
        vxy_temp = []
        vyx_temp = []
        Gxy_temp = []

        Ex_f_temp = []
        Ey_f_temp = []
        vxy_f_temp = []
        vyx_f_temp = []
        Gxy_f_temp = []

        for theta in range(-90, 91):
                angle_arr = np.array([theta, -theta , phi, phi, phi, phi, phi, phi, -theta, theta,
                                    theta, -theta , phi, phi, phi, phi, phi, phi, -theta, theta])
                
                zcoord_arr = zcoordinate(angle_arr, t)
                
                Q_overall = Q_transformed(Q11, Q12, Q22, Q66, angle_arr) 
                ABD_matrix = ABD_Calc(Q_overall, zcoord_arr)    # For each angle combination, we will have a different ABD matrix, which will give us different engineering constants.
                Ex, Ey, vxy, vyx, Gxy, Ex_f, Ey_f, vxy_f, vyx_f, Gxy_f = Equvalent_properties(ABD_matrix, zcoord_arr)

                Ex_temp.append(Ex)  # For each theta, we will have 181 different phi values, and thus 181 different engineering constants.
                Ey_temp.append(Ey)
                vxy_temp.append(vxy)
                vyx_temp.append(vyx)
                Gxy_temp.append(Gxy)
                
                Ex_f_temp.append(Ex_f)
                Ey_f_temp.append(Ey_f)
                vxy_f_temp.append(vxy_f)
                vyx_f_temp.append(vyx_f)
                Gxy_f_temp.append(Gxy_f)    

        Ex_arr.append(Ex_temp)
        Ey_arr.append(Ey_temp)
        vxy_arr.append(vxy_temp)
        vyx_arr.append(vyx_temp)
        Gxy_arr.append(Gxy_temp)
        Ex_f_arr.append(Ex_f_temp)
        Ey_f_arr.append(Ey_f_temp)
        vxy_f_arr.append(vxy_f_temp)
        vyx_f_arr.append(vyx_f_temp)
        Gxy_f_arr.append(Gxy_f_temp)
        # Use the theta loop to append the 181 engineering constants in a 181*181 array, where the rows correspond to theta and the columns correspond to phi.

        
        
    
    # print(Ex_arr)
    # plt.figure(figsize=(10, 6))
    # for i in range(10):
    #     plt.plot(range(-90, 91), Ex_arr[i], label='Ex')
    # plt.xlabel('Theta [degrees]')
    # plt.ylabel('Ex [Pa]')
    # plt.show()
    # Create x and y axes from -90 to 90
    
    Ex_arr = np.array(Ex_arr)
    x = np.linspace(-90, 90, Ex_arr.shape[1])
    y = np.linspace(-90, 90, Ex_arr.shape[0])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot with color based on Z
    surf = ax.plot_surface(X, Y, Ex_arr, cmap='viridis', edgecolor='none')

    # Colorbar showing Z scale
    fig.colorbar(surf, ax=ax, shrink=0.6, label="Z value")
    plt.show()

    # # Plot contour map
    # plt.figure()
    # plt.contourf(X, Y, Ex_arr, levels=50)
    # plt.colorbar()

    # plt.xlabel("phi")
    # plt.ylabel("theta")
    # plt.title("Contour Map")

    # plt.show()

    


if __name__=="__main__":
    main()
