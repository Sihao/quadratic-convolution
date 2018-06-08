from gradients import *

from numpy import tensordot as tdot

"""Responses for different types of models"""

# Calculates responses for softplus model
def respSP(S,P):
    # Inputs:
    #   S - A stimulus
    #   P - Model parameters
    # Output:
    #   r - Response of model for given stimulus

    # Extract parameters
    a1,v1,J,a2,v2,d = P

    # Model nonlinearities
    f1,f2 = logistic,softPlus

    ndim = v1.ndim

    # Calculate first layer responses
    r1 = f1(a1+tdot(v1,S,2*(list(range(ndim)),))+(tdot(J,S,2*(list(range(ndim)),))*S).sum(tuple(range(ndim))))

    # Calculate second layer responses
    r2 = f2(a2+(r1*v2).sum())

    return d*r2

# Calculates response for linear softplus model
def respLinearSP(S,P):
    # Inputs:
    #   S - A stimulus
    #   P - Parameters
    # Output:
    #   r - Response of model for give stimulus

    # Extract parameters
    a1,v1,a2,v2,d = P

    # Model nonlinearities
    f1,f2 = logistic,softPlus

    ndim = v1.ndim

    # Calculate first layer responses
    r1 = f1(a1+tdot(v1,S,2*(list(range(ndim)),)))

    # Calculate second layer responses
    r2 = f2(a2+(r1*v2).sum())

    return d*r2

# Calculates responses for logistic model
def respLog2(S,P):
    # Inputs:
    #   S - A stimulus
    #   P - Model parameters
    # Output:
    #   r - Response of model for given stimulus

    # Extract parameters
    a1,v1,J,a2,v2,d = P

    # Model nonlinearities
    f1,f2 = logistic,logistic

    ndim = v1.ndim

    # Calculate first layer responses
    r1 = f1(a1+tdot(v1,S,2*(list(range(ndim)),))+(tdot(J,S,2*(list(range(ndim)),))*S).sum(tuple(range(ndim))))

    # Calculate second layer responses
    r2 = f2(a2+(r1*v2).sum())

    return d*r2

# Calculates responses for linear logistic model
def respLinearLog2(S,P):
    # Inputs:
    #   S - A stimulus
    #   P - Model parameters
    # Output:
    #   r - Response of model for given stimulus

    # Extract parameters
    a1,v1,a2,v2,d = P

    # Model nonlinearities
    f1,f2 = logistic,logistic

    ndim = v1.ndim

    # Calculate first layer responses
    r1 = f1(a1+tdot(v1,S,2*(list(range(ndim)),)))

    # Calculate second layer responses
    r2 = f2(a2+(r1*v2).sum())

    return d*r2

# Calculate responses of many stimuli
def Resp(S,P,func):
    r = array([func(s,P) for s in S])
    rs = r.shape[:1]+r.shape[2:]
    return r.reshape(rs)