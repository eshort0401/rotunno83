# Utilities
import sys
from tqdm import tqdm

# Performance
from numba import jit, prange

# Analysis
import numpy as np
import xarray as xr

# @jit(parallel=True)
def integrate_case_one(xi,zeta,tau,xip,zetap,xi0,beta,Atilde):

    psi = np.zeros((zeta.size, xi.size), dtype=np.complex64)
    u = np.zeros((zeta.size, xi.size), dtype=np.complex64)
    w = np.zeros((zeta.size, xi.size), dtype=np.complex64)

    psi_intgd_dxp_dzp = np.zeros(zetap.size, dtype=np.complex64)
    psi_intgd_dxp = np.zeros(xip.size, dtype=np.complex64)

    u_intgd_dxp_dzp = np.zeros(zetap.size, dtype=np.complex64)
    u_intgd_dxp = np.zeros(xip.size, dtype=np.complex64)

    w_intgd_dxp_dzp = np.zeros(zetap.size, dtype=np.complex64)
    w_intgd_dxp = np.zeros(xip.size, dtype=np.complex64)

    # Perform numerical integrations filtering out unwanted cases
    cond_zetap = (zetap==0)
    for j in tqdm(prange(zeta.size), file=sys.stdout, position=0, leave=True):
        cond_j = (j==0)*np.ones(zetap.size)
        for i in range(xi.size):
            for k in range(xip.size):

                # Create filters
                cond_1 = np.logical_and(cond_j, cond_zetap)
                cond_xi_xip = (xi[i]==xip[k])*np.ones(zetap.size)
                cond_xi_or_zetap = np.logical_or(cond_j, cond_zetap)
                cond_2 = np.logical_and(cond_xi_xip, cond_xi_or_zetap)
                cond_3 = (zeta[j]==zetap)

                cond_psi = (cond_1+cond_2+cond_3).astype(np.bool)
                cond_uw = (cond_xi_xip+cond_3).astype(np.bool)

                psi_intgd_dxp_dzp[~cond_psi] = calc_psi_ig(xi[i],xip[k],
                                                          zeta[j],
                                                          zetap[~cond_psi],
                                                          xi0)
                u_intgd_dxp_dzp[~cond_uw] = calc_u_ig(xi[i],xip[k],zeta[j],
                                                     zetap[~cond_uw])
                u_intgd_dxp_dzp *= calc_exp_term(xip[k],zetap,xi0)
                w_intgd_dxp_dzp[~cond_uw] = calc_w_ig(xi[i],xip[k],zeta[j],
                                                     zetap[~cond_uw])
                w_intgd_dxp_dzp *= calc_exp_term(xip[k],zetap,xi0)

                psi_intgd_dxp[k] = np.trapz(psi_intgd_dxp_dzp,zetap)
                u_intgd_dxp[k] = np.trapz(u_intgd_dxp_dzp,zetap)
                w_intgd_dxp[k] = np.trapz(w_intgd_dxp_dzp,zetap)

            psi[j,i] = np.trapz(psi_intgd_dxp, xip)
            u[j,i] = np.trapz(u_intgd_dxp, xip)
            w[j,i] = np.trapz(w_intgd_dxp, xip)

    [psi, u, w] = [np.tile(var,(tau.size,1,1)) for var in [psi, u, w]]
    tau_ar = np.tile(np.expand_dims(np.expand_dims(tau,-1),-1),
                     (zeta.size,xi.size))
    [psi, u, w] = [var*np.sin(tau_ar) for var in [psi, u, w]]

    # Scale
    # import pdb; pdb.set_trace()

    psi = -(beta*xi0*Atilde/(4*np.pi))*np.real(psi)
    u = -(beta*xi0*Atilde/(4*np.pi))*np.real(u)
    w = -(beta*xi0*Atilde/(4*np.pi))*np.real(w)

    return psi, u, w

# Term in integrand of psi
@jit(parallel=True, nopython=True)
def calc_psi_ig(xi,xip,zeta,zetap,xi0):
    # Note we are still not suppressing the singularity, but given behaviour of
    # log does not appear to matter. Better solution desirable!
    psi_ig = (np.log(((xi-xip)**2 + (zeta-zetap)**2) /
              ((xi-xip)**2 + (zeta+zetap)**2))
              *(np.exp(-zetap)/(xip**2+xi0**2)))
    return psi_ig

@jit(parallel=True, nopython=True)
def calc_w_ig(xi,xip,zeta,zetap):
    w_ig = -(2*(xi-xip)*(-(zeta-zetap)**2+(zeta+zetap)**2)
             /(((xi-xip)**2+(zeta-zetap)**2)
               *((xi-xip)**2+(zeta+zetap)**2)))
    return w_ig

def calc_u_ig(xi,xip,zeta,zetap):
    u_ig = (2*((zeta-zetap)*((xi-xip)**2
               +(zeta+zetap)**2)
               -(zeta+zetap)*((xi-xip)**2
               +(zeta-zetap)**2))
            /(((xi-xip)**2+(zeta-zetap)**2)
              *((xi-xip)**2+(zeta+zetap)**2)))
    return u_ig

# Term in integrand for psi, psiXi and psiZeta
@jit(parallel=True, nopython=True)
def calc_exp_term(xip,zetap,xi0):
    return np.exp(-zetap)/(xip**2+xi0**2)

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
    psi = -beta*Atilde*psi
    u = -beta*Atilde*u
    w = -beta*Atilde*w

    return psi, u, w
