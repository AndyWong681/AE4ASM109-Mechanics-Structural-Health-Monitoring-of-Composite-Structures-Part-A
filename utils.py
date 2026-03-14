import numpy as np 
import math
import matplotlib.pyplot as plt

def sin(angle): # in radians

    return np.sin(angle)

def cos(angle): # in radians

    return np.cos(angle)

def stressTOstress_trans(angle): # 3x3 matrix angle transformation from stress to strain
    m = cos(-angle)
    n = sin(-angle)
    return np.array([[m**2, n**2, 2*m*n],[n**2, m**2, -2*m*n],[-m*n, m*n, m**2 - n**2]])

def strainTOstrain_trans(angle): # 3x3 matrix angle transformation from strain to stress
    m = cos(-angle)
    n = sin(-angle)
    return np.array([[m**2, n**2, m*n],[n**2, m**2, -m*n],[-2*m*n, 2*m*n, m**2 - n**2]])

def stressTOstress6x6_trans(angle): # 6x6 matrix angle transformation from stress to strain
    basemat6x6 = np.zeros((6,6))
    basemat6x6[0:3, 0:3] = stressTOstrain_trans(angle)
    basemat6x6[3:6, 3:6] = stressTOstrain_trans(angle)
    return basemat6x6

def strainTOstrain6x6_trans(angle): # 6x6 matrix angle transformation from strain to stress
    basemat6x6 = np.zeros((6,6))
    basemat6x6[0:3, 0:3] = strainTOstress_trans(angle)
    basemat6x6[3:6, 3:6] = strainTOstress_trans(angle)
    return basemat6x6



def zcoordinate(theta_lst, t):
    
    z_coordinate = []
    
    for i in range(len(theta_lst) + 1):
        
        z = len(theta_lst) / 2 * t - i * t
        
        z_coordinate.append(z)

    return np.array(z_coordinate)



def local_elastic_property(E1, E2, G12, v12):
    
    v21 = E2 * v12 / E1
    Q = 1 - v12 * v21 
    Q11 = E1 * 1 / Q
    Q22 = E2 * 1 / Q
    Q12 = v12 * E2 * 1 / Q
    Q66 = G12
    
    return Q11, Q12, Q22, Q66, np.array([[Q11, Q12, 0],[Q12, Q22, 0],[0, 0, Q66]])


def Q_transformed(Q11, Q12, Q22, Q66, theta_lst):

    Q_overall_lst = []

    for i in range(len(theta_lst)):


        m = np.cos(np.radians(float(theta_lst[i])))
        n = np.sin(np.radians(float(theta_lst[i])))

        Qxx = Q11*m**4 + 2*(Q12 + 2*Q66)*m**2*n**2 + Q22*n**4
        Qxy = (Q11 + Q22 - 4*Q66)*m**2*n**2 + Q12*(m**4 + n**4)
        Qyy = Q11*n**4 + 2*(Q12 + 2*Q66)*m**2*n**2 + Q22*m**4
        Qxs = (Q11 - Q12 - 2*Q66)*n*m**3 + (Q12 - Q22 + 2*Q66)*n**3*m
        Qys = (Q11 - Q12 - 2*Q66)*m*n**3 + (Q12 - Q22 + 2*Q66)*m**3*n
        Qss = (Q11 + Q22 - 2*Q12 - 2*Q66)*m**2*n**2 + Q66*(m**4 + n**4)

        Q_bar = [
            [float(Qxx), float(Qxy), float(Qxs)],
            [float(Qxy), float(Qyy), float(Qys)],
            [float(Qxs), float(Qys), float(Qss)]
        ]

        Q_overall_lst.append(Q_bar)

    return Q_overall_lst


def ABD_Calc(Q_overall_lst, z_lst):

    A11_final_lst = []
    A12_final_lst = []
    A13_final_lst = []
    A21_final_lst = []
    A22_final_lst = []
    A23_final_lst = []
    A31_final_lst = []
    A32_final_lst = []
    A33_final_lst = []

    B11_final_lst = []
    B12_final_lst = []
    B13_final_lst = []
    B21_final_lst = []
    B22_final_lst = []
    B23_final_lst = []
    B31_final_lst = []
    B32_final_lst = []
    B33_final_lst = []

    D11_final_lst = []
    D12_final_lst = []
    D13_final_lst = []
    D21_final_lst = []
    D22_final_lst = []
    D23_final_lst = []
    D31_final_lst = []
    D32_final_lst = []
    D33_final_lst = []

    for i in range(len(Q_overall_lst)):


        Q11 = Q_overall_lst[i][0][0]
        Q22 = Q_overall_lst[i][1][1]
        Q12 = Q_overall_lst[i][0][1]
        Q16 = Q_overall_lst[i][0][2]
        Q26 = Q_overall_lst[i][1][2]
        Q21 = Q12
        Q61 = Q16
        Q62 = Q26
        Q66 = Q_overall_lst[i][2][2]


        A11_local = Q11 * ( z_lst[i] - z_lst[i+1] )
        A12_local = Q12 * ( z_lst[i] - z_lst[i+1] )
        A13_local = Q16 * ( z_lst[i] - z_lst[i+1] )
        A21_local = Q21 * ( z_lst[i] - z_lst[i+1] )
        A22_local = Q22 * ( z_lst[i] - z_lst[i+1] )
        A23_local = Q26 * ( z_lst[i] - z_lst[i+1] )
        A31_local = Q61 * ( z_lst[i] - z_lst[i+1] )
        A32_local = Q62 * ( z_lst[i] - z_lst[i+1] )
        A33_local = Q66 * ( z_lst[i] - z_lst[i+1] )

        B11_local = 1 / 2 * Q11 * (z_lst[i] ** 2 - z_lst[i + 1] ** 2)
        B12_local = 1 / 2 * Q12 * (z_lst[i] ** 2 - z_lst[i + 1] ** 2)
        B13_local = 1 / 2 * Q16 * (z_lst[i] ** 2 - z_lst[i + 1] ** 2)
        B21_local = 1 / 2 * Q21 * (z_lst[i] ** 2 - z_lst[i + 1] ** 2)
        B22_local = 1 / 2 * Q22 * (z_lst[i] ** 2 - z_lst[i + 1] ** 2)
        B23_local = 1 / 2 * Q26 * (z_lst[i] ** 2 - z_lst[i + 1] ** 2)
        B31_local = 1 / 2 * Q61 * (z_lst[i] ** 2 - z_lst[i + 1] ** 2)
        B32_local = 1 / 2 * Q62 * (z_lst[i] ** 2 - z_lst[i + 1] ** 2)
        B33_local = 1 / 2 * Q66 * (z_lst[i] ** 2 - z_lst[i + 1] ** 2)

        D11_local = 1 / 3 * Q11 * (z_lst[i] ** 3 - z_lst[i + 1] ** 3)
        D12_local = 1 / 3 * Q12 * (z_lst[i] ** 3 - z_lst[i + 1] ** 3)
        D13_local = 1 / 3 * Q16 * (z_lst[i] ** 3 - z_lst[i + 1] ** 3)
        D21_local = 1 / 3 * Q21 * (z_lst[i] ** 3 - z_lst[i + 1] ** 3)
        D22_local = 1 / 3 * Q22 * (z_lst[i] ** 3 - z_lst[i + 1] ** 3)
        D23_local = 1 / 3 * Q26 * (z_lst[i] ** 3 - z_lst[i + 1] ** 3)
        D31_local = 1 / 3 * Q61 * (z_lst[i] ** 3 - z_lst[i + 1] ** 3)
        D32_local = 1 / 3 * Q62 * (z_lst[i] ** 3 - z_lst[i + 1] ** 3)
        D33_local = 1 / 3 * Q66 * (z_lst[i] ** 3 - z_lst[i + 1] ** 3)

        A11_final_lst.append(A11_local)
        A12_final_lst.append(A12_local)
        A13_final_lst.append(A13_local)
        A21_final_lst.append(A21_local)
        A22_final_lst.append(A22_local)
        A23_final_lst.append(A23_local)
        A31_final_lst.append(A31_local)
        A32_final_lst.append(A32_local)
        A33_final_lst.append(A33_local)

        B11_final_lst.append(B11_local)
        B12_final_lst.append(B12_local)
        B13_final_lst.append(B13_local)
        B21_final_lst.append(B21_local)
        B22_final_lst.append(B22_local)
        B23_final_lst.append(B23_local)
        B31_final_lst.append(B31_local)
        B32_final_lst.append(B32_local)
        B33_final_lst.append(B33_local)

        D11_final_lst.append(D11_local)
        D12_final_lst.append(D12_local)
        D13_final_lst.append(D13_local)
        D21_final_lst.append(D21_local)
        D22_final_lst.append(D22_local)
        D23_final_lst.append(D23_local)
        D31_final_lst.append(D31_local)
        D32_final_lst.append(D32_local)
        D33_final_lst.append(D33_local)


    A11_final = sum(A11_final_lst)
    A12_final = sum(A12_final_lst)
    A13_final = sum(A13_final_lst)
    A21_final = sum(A21_final_lst)
    A22_final = sum(A22_final_lst)
    A23_final = sum(A23_final_lst)
    A31_final = sum(A31_final_lst)
    A32_final = sum(A32_final_lst)
    A33_final = sum(A33_final_lst)
    
    B11_final = sum(B11_final_lst)
    B12_final = sum(B12_final_lst)
    B13_final = sum(B13_final_lst)
    B21_final = sum(B21_final_lst)
    B22_final = sum(B22_final_lst)
    B23_final = sum(B23_final_lst)
    B31_final = sum(B31_final_lst)
    B32_final = sum(B32_final_lst)
    B33_final = sum(B33_final_lst)

    D11_final = sum(D11_final_lst)
    D12_final = sum(D12_final_lst)
    D13_final = sum(D13_final_lst)
    D21_final = sum(D21_final_lst)
    D22_final = sum(D22_final_lst)
    D23_final = sum(D23_final_lst)
    D31_final = sum(D31_final_lst)
    D32_final = sum(D32_final_lst)
    D33_final = sum(D33_final_lst)


    ABD = [[A11_final, A12_final, A13_final, B11_final, B12_final, B13_final],
           [A21_final, A22_final, A23_final, B21_final, B22_final, B23_final],
           [A31_final, A32_final, A33_final, B31_final, B32_final, B33_final],
           [B11_final, B12_final, B13_final, D11_final, D12_final, D13_final],
           [B21_final, B22_final, B23_final, D21_final, D22_final, D23_final],
           [B31_final, B32_final, B33_final, D31_final, D32_final, D33_final]]


    return np.array(ABD)



def Equvalent_properties(ABD, z_lst):


    t_total = z_lst[0] - z_lst[-1]


    A = ABD[:3, :3]
    Ex = (A[0, 0] * A[1, 1] - A[0, 1] ** 2) / (t_total * A[1, 1])
    Ey = (A[0, 0] * A[1, 1] - A[0, 1] ** 2) / (t_total * A[0, 0])
    vxy = A[0, 1] / A[1, 1]
    vyx = A[0, 1] / A[0, 0]
    Gxy = ABD[2, 2] / t_total
    
    D = ABD[3:6, 3:6]

    Ex_f = 12 * (D[0,0]*D[1,1] - D[0,1]**2) / (t_total**3 * D[1,1])
    Ey_f = 12 * (D[0,0]*D[1,1] - D[0,1]**2) / (t_total**3 * D[0,0])
    vxy_f = D[0,1] / D[1,1]
    vyx_f = D[0,1] / D[0,0]
    Gxy_f = 12 * D[2,2] / t_total**3
    

    return Ex, Ey, vxy, vyx, Gxy, Ex_f, Ey_f, vxy_f, vyx_f, Gxy_f




def Applied_Loading(Nx, Ny, Nxy, Mx, My, Mxy):  # applied loading in the form of Nx, Ny, Nxy, Mx, My, Mxy

    load = np.array([Nx, Ny, Nxy, Mx, My, Mxy])   

    return load





def Strain_ply_calculation(strain_global, z_lst):
    
    strain = strain_global[:3]
    curvature = strain_global[3:]

    strain_global_lst = []

    for i in range(len(z_lst)-1):

        strain_global_local = strain + (z_lst[i] + z_lst[i+1]) / 2 * curvature
        strain_global_lst.append(strain_global_local)

    return np.array(strain_global_lst)




       
    

    


