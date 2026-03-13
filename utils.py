def sin(angle): #in radians
    return math.sin(angle)

def cos(angle): #in radians
    return math.cos(angle)

def stressTOstrain_angle_transformation(angle):
    m = cos(angle)
    n = sin(angle)
    return np.array([[m**2, n**2, 2*m*n],[n**2, m**2, -2*m*n],[-m*n, m*n, m**2 - n**2]])

def strainTOstress_angle_transformation(angle):
    m = cos(angle)
    n = sin(angle)
    return np.array([[m**2, n**2, m*n],[n**2, m**2, -m*n],[-2*m*n, 2*m*n, m**2 - n**2]])
