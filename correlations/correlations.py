import numpy as np

def haaland(Re, dh, eps):
    term = (eps/dh/3.7)**1.11 + 6.9/Re
    f = (-1.8*np.log10(term))**(-2)
    return f


def gnielinski(Re, Pr, f):
    Nu = ((f/8)*(Re - 1000)*Pr) / \
         (1 + 12.7*np.sqrt(f/8)*(Pr**(2/3) - 1))
    return Nu