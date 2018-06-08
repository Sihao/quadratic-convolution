from numpy import array, log, exp, spacing

# Small number
eps = spacing(1.)


# Soft-rectifier (log(1+exp(x)))
def softPlus(x):

    # Create copy of x
    r = array(x)

    # If exp(r) would overflow, treat softPlus(r) as linear
    if r.ndim:
        r[r<700] = log(1+exp(r[r<700]))
    else:
        if r < 700:
            r = log(1+exp(r))

    return r


# Derivative of softplus rectifier
def dSP(x):

    return logistic(x)

# Array version of logistic function
def logistic(x):

    return 1/(1+exp(-x))

# Derivative of logistic function
def dlog(x):

    x = logistic(x)

    return x*(1-x)


# Poisson log likelihood
# Returns difference between likelihood of predictions and observations in order
# to make value positive
def llike(Y,R):
    return (Y*log(Y+eps)-Y-Y*log(R+eps)+R).mean()

# Derivative of poisson log-likelihood
def dllike(Y,R):
    return Y/(R+eps)-1