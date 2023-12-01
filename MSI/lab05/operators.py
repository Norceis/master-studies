import numpy as np

def productN(args, op):
    return np.product(args, axis=0)

def zadeh_s_norm(args, op):
    return np.max(args, axis=0)

def probabilistic_s_norm(args, op):
    return np.array(args[0] + args[1] - (args[0] * args[1]))

def calculate_norms(args, op, norms):
    first_term = norms[0](np.array([args[0], args[1]]), op)
    return norms[1](np.array([first_term, args[2]]), op)

