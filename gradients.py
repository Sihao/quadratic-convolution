from math_utils import *
from numpy import tensordot as tdot
from Params import Params


"""
Gradients for the different models
"""

# Calculate gradient for softplus model
def gradSP(Y, # Observed response
           S, # Stimulus
           P  # Parameters
           ):

    # Extract parameters
    a1,v1,J1,a2,v2,d = P

    # Model nonlinearities
    f1,f2 = logistic,softPlus

    # Derivatives of model nonlinearities and cost function
    df1,df2,dfe = dlog,dSP,dllike

    ndim = v1.ndim

    x1 = a1+tdot(v1,S,2*(list(range(ndim)),))+(tdot(J1,S,2*(list(range(ndim)),))*S).sum(tuple(range(ndim)))
    r1 = f1(x1)
    dr1 = df1(x1)
    x2 = a2+(r1*v2).sum()
    r2 = f2(x2)
    dr2 = df2(x2)

    dy = d*dfe(Y,d*r2)
    dd = dy*r2/d

    da2 = dy*dr2
    dv2 = dy*dr2*r1

    da1 = dy*dr2*(dr1*v2).sum()
    dv1 = dy*dr2*tdot(dr1*S,v2,(list(range(-ndim,0)),list(range(ndim))))
    dJ1 = dy*dr2*tdot(dr1*S*S.reshape(S.shape[:ndim]+ndim*(1,)+S.shape[ndim:]),v2,(list(range(-ndim,0)),list(range(ndim))))

    return Params([da1,dv1,dJ1,da2,dv2,dd])

# Calculate gradient for linear softplus model
def gradLinearSP(Y, # Observed response
                 S, # Stimulus
                 P  # Parameters
                 ):

    # Extract parameters
    a1,v1,a2,v2,d = P

    # Model nonlinearities
    f1,f2 = logistic,softPlus

    # Derivative of model nonlinearities and cost function
    df1,df2,dfe = dlog,dSP,dllike

    ndim = v1.ndim

    x1 = a1+tdot(v1,S,2*(list(range(ndim)),))
    r1 = f1(x1)
    dr1 = df1(x1)
    x2 = a2+(r1*v2).sum()
    r2 = f2(x2)
    dr2 = df2(x2)

    dy = d*dfe(Y,d*r2)
    dd = dy*r2/d

    da2 = dy*dr2
    dv2 = dy*dr2*r1

    da1 = dy*dr2*(dr1*v2).sum()
    dv1 = dy*dr2*tdot(dr1*S,v2,(list(range(-ndim,0)),list(range(ndim))))

    return Params([da1,dv1,da2,dv2,dd])

# Calculate gradient for logistic model
def gradLog2(Y, # Observed response
             S, # Stimulus
             P  # Parameters
             ):

    # Extract parameters
    a1,v1,J1,a2,v2,d = P

    # Model nonlinearities
    f1,f2 = logistic,logistic

    # Derivative of model nonlinearities and cost function
    df1,df2,dfe = dlog,dlog,dllike

    ndim = v1.ndim

    x1 = a1+tdot(v1,S,2*(list(range(ndim)),))+(tdot(J1,S,2*(list(range(ndim)),))*S).sum(tuple(range(ndim)))
    r1 = f1(x1)
    dr1 = df1(x1)
    x2 = a2+(r1*v2).sum()
    r2 = f2(x2)
    dr2 = df2(x2)

    dy = d*dfe(Y,d*r2)
    dd = dy*r2/d

    da2 = dy*dr2
    dv2 = dy*dr2*r1

    da1 = dy*dr2*(dr1*v2).sum()
    dv1 = dy*dr2*tdot(dr1*S,v2,(list(range(-ndim,0)),list(range(ndim))))
    dJ1 = dy*dr2*tdot(dr1*S*S.reshape(S.shape[:ndim]+ndim*(1,)+S.shape[ndim:]),v2,(list(range(-ndim,0)),list(range(ndim))))

    return Params([da1,dv1,dJ1,da2,dv2,dd])

# Calculate gradient for linear logistic model
def gradLinearLog2(Y, # Observed response
                   S, # Stimulus
                   P  # Parameters
                   ):

    # Extract parameters
    a1,v1,a2,v2,d = P

    # Model nonlinearities
    f1,f2 = logistic,logistic

    # Derivative of model nonlinearities and cost function
    df1,df2,dfe = dlog,dSP,dllike

    ndim = v1.ndim

    x1 = a1+tdot(v1,S,2*(list(range(ndim)),))
    r1 = f1(x1)
    dr1 = df1(x1)
    x2 = a2+(r1*v2).sum()
    r2 = f2(x2)
    dr2 = df2(x2)

    dy = d*dfe(Y,d*r2)
    dd = dy*r2/d

    da2 = dy*dr2
    dv2 = dy*dr2*r1

    da1 = dy*dr2*(dr1*v2).sum()
    dv1 = dy*dr2*tdot(dr1*S,v2,(list(range(-ndim,0)),list(range(ndim))))

    return Params([da1,dv1,da2,dv2,dd])