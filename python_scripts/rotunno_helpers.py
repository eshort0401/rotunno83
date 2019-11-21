# Utilities
import sys
from tqdm import tqdm

# Performance
from numba import jit, prange

# Analysis
import numpy as np
import xarray as xr

@jit(parallel=True)
def integrate_case_two(xi,zeta,tau,k,xi0,beta,Atilde):
    psi = np.zeros((tau.size, zeta.size, xi.size))
    u = np.zeros((tau.size, zeta.size, xi.size))
    w = np.zeros((tau.size, zeta.size, xi.size))
    psi_integrand = np.zeros(k.size)
    u_integrand = np.zeros(k.size)
    w_integrand = np.zeros(k.size)
    # Perform numerical integration
    for i in tqdm(prange(xi.size), file=sys.stdout, position=0, leave=True):
        for j in range(zeta.size):
            for l in range(tau.size):
                psi_integrand = (np.cos(k*xi[i])
                                *np.exp(-xi0*k)
                                *(np.sin(k*zeta[j]+tau[l])
                                  -np.exp(-zeta[j])*np.sin(tau[l]))
                                /(1+k**2))
                u_integrand = (np.cos(k*xi[i])
                               *np.exp(-xi0*k)
                               *(k*np.cos(k*zeta[j]+tau[l])
                                 +np.exp(-zeta[j])*np.sin(tau[l]))
                               /(1+k**2))
                w_integrand = (k*np.sin(k*xi[i])
                               *np.exp(-xi0*k)
                               *(np.sin(k*zeta[j]+tau[l])
                                 -np.exp(-zeta[j])*np.sin(tau[l]))
                               /(1+k**2))

                psi[l,j,i] = np.trapz(psi_integrand,k)
                u[l,j,i] = np.trapz(u_integrand,k)
                w[l,j,i] = np.trapz(w_integrand,k)
    # Scale
    psi = -(1/np.pi)*beta*Atilde*psi
    u = -(1/np.pi)*beta*Atilde*u
    w = -(1/np.pi)*beta*Atilde*w
    return psi, u, w
