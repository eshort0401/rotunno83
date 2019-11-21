# Utilities
from tqdm import tqdm
import sys

# Performance
from numba import jit, prange

# Analysis
import numpy as np
import xarray as xr

from scipy.special import exp1

@jit(parallel=True)
def integrate_qian(xi,zeta,tau,s,alpha,U,L):

    psi = np.zeros((3, tau.size, zeta.size, xi.size), dtype=np.complex64)
    u = np.zeros((3, tau.size, zeta.size, xi.size), dtype=np.complex64)
    w = np.zeros((3, tau.size, zeta.size, xi.size), dtype=np.complex64)

    # Define alternative domains
    theta = calc_theta(s,alpha=alpha)

    k_3 = calc_k_3(theta,U)
    k0_3 = calc_k_3(0,U)

    k_2 = calc_k_2(theta,U)
    k0_2 = calc_k_2(0,U)

    k_1=np.concatenate([k_2[-1::-1], np.array([1/U]), k_3])

    # Perform numerical integration zeta>0
    for j in tqdm(prange(1,zeta.size), file=sys.stdout):
        for i in range(xi.size):
            for l in range(tau.size):

                # Calc psi1
                psi1_ig = calc_psi1(xi[i],zeta[j],tau[l],k_1,U,L)
                psi[0][l,j,i] = np.trapz(psi1_ig, k_1)

                # Calc psi2
                psi2a_ig = calc_psi2a(xi[i],zeta[j],tau[l],s,alpha,U,L)
                psi2b_ig = calc_psi2b(xi[i],zeta[j],tau[l],s,alpha,U,L)

                psi2ab = np.trapz(psi2a_ig+psi2b_ig, s)

                psi2c = (-1/(2*U)*np.exp(1j*zeta[j]/U)
                         *calc_C2(xi[i],tau[l],k0_2,U,L)
                         *calc_ep0(-1j*zeta[j]/U))

                psi[1][l,j,i] = psi2ab+psi2c

                # Calc psi3
                psi3a_ig = calc_psi3a(xi[i],zeta[j],tau[l],s,alpha,U,L)
                psi3b_ig = calc_psi3b(xi[i],zeta[j],tau[l],s,alpha,U,L)

                psi3ab = np.trapz(psi3a_ig+psi3b_ig, s)

                psi3c = -U/2*calc_C3(xi[i],tau[l],k0_3,U,L)
                psi3c = (-U/2*calc_C3(xi[i],tau[l],k0_3,U,L)
                         *calc_ep0(1j*zeta[j]/U))

                psi[2][l,j,i] = psi3ab+psi3c

                # Calc u1
                u1_ig = calc_u1(xi[i],zeta[j],tau[l],k_1,U,L)
                u[0][l,j,i] = np.trapz(u1_ig, k_1)

                # Calc u2
                u2a_ig = calc_u2a(xi[i],zeta[j],tau[l],s,alpha,U,L)
                u2b_ig = calc_u2b(xi[i],zeta[j],tau[l],s,alpha,U,L)

                u2ab = np.trapz(u2a_ig+u2b_ig, s)

                u2b_ig = calc_u2b(xi[i],zeta[j],tau[l],s,alpha,U,L)
                u2c = (1j*np.exp(1j*zeta[j]/U)/(2*U**2)
                       *calc_C2(xi[i],tau[l],k0_2,U,L)
                       *(-calc_ep0(-1j*zeta[j]/U)
                         +calc_exp1(1j*zeta[j]/U)))

                u[1][l,j,i] = u2ab+u2c

                # Calc u3
                u3a_ig = calc_u3a(xi[i],zeta[j],tau[l],s,alpha,U,L)
                u3b_ig = calc_u3b(xi[i],zeta[j],tau[l],s,alpha,U,L)

                u3ab = np.trapz(u3a_ig+u3b_ig, s)

                u3c = (-1j/2*calc_C3(xi[i],tau[l],k0_3,U,L)
                       *calc_exp1(-1j*zeta[j]/U))

                u[2][l,j,i] = u3ab+u3c

                # Calc w1
                w1_ig = psi1_ig*1j*k_1
                w[0][l,j,i] = np.trapz(w1_ig,k_1)

                # Calc w2
                w2b_ig = calc_w2b(xi[i],zeta[j],tau[l],s,alpha,U,L)
                w2ab = np.trapz(1j*k_2*psi2a_ig+w2b_ig,s)

                w2c = (-1/(2*U)*np.exp(1j*zeta[j]/U)
                         *calc_C2x(xi[i],tau[l],k0_2,U,L)
                         *calc_ep0(-1j*zeta[j]/U))

                w[1][l,j,i] = w2ab+w2c

                # Calc w3
                w3b_ig = calc_w3b(xi[i],zeta[j],tau[l],s,alpha,U,L)
                w3ab = np.trapz(1j*k_3*psi3a_ig+w3b_ig,s)

                w3c = (-U/2*calc_C3x(xi[i],tau[l],k0_3,U,L)
                       *calc_ep0(1j*zeta[j]/U))

                w[2][l,j,i] = w3ab+w3c

    # Use a finite difference to get u at z=0.
    u[:,:,0,:] = psi[:,:,1,:]/zeta[1]

    psi = (1/np.pi)*np.real(psi)
    u = (1/np.pi)*np.real(u)
    w = -(1/np.pi)*np.real(w)
    return psi, u, w

@jit(parallel=True)
def integrate_qian_U0(xi,zeta,tau,s,alpha,L):

    psi = np.zeros((2, tau.size, zeta.size, xi.size), dtype=np.complex64)
    u = np.zeros((2, tau.size, zeta.size, xi.size), dtype=np.complex64)
    w = np.zeros((2, tau.size, zeta.size, xi.size), dtype=np.complex64)

    # Define alternative domains
    theta = calc_theta(s,alpha=alpha)

    # Get wavenumbers greater than 2*pi
    k_3 = calc_k_3(theta,1/(2*np.pi))
    k0_3 = calc_k_3(0,1/(2*np.pi))

    # Get wavenumers less than 2*pi
    k_2 = calc_k_2(theta,1/(2*np.pi))
    k0_2 = calc_k_2(0,1/(2*np.pi))

    # Create wavenumber domain
    k_1=np.concatenate([k_2[-1::-1], np.array([1/1/(2*np.pi)]), k_3])

    # Perform numerical integration zeta>0
    for j in tqdm(prange(0,zeta.size), file=sys.stdout):
        for i in range(xi.size):
            for l in range(tau.size):

                # Calc psi
                psi1_ig = calc_psi1(xi[i],zeta[j],tau[l],k_1,0,L)
                psi[0][l,j,i] = np.trapz(psi1_ig, k_1)
                psi2_ig = calc_psi2_U0(xi[i],zeta[j],tau[l],k_1,L)
                psi[1][l,j,i] = np.trapz(psi2_ig, k_1)

                # Calc u
                u1_ig = calc_u1(xi[i],zeta[j],tau[l],k_1,0,L)
                u[0][l,j,i] = np.trapz(u1_ig, k_1)
                u2_ig = calc_u2_U0(xi[i],zeta[j],tau[l],k_1,L)
                u[1][l,j,i] = np.trapz(u2_ig, k_1)

                # Calc w
                w[0][l,j,i] = np.trapz(psi1_ig*1j*k_1,k_1)
                w[1][l,j,i] = np.trapz(psi2_ig*1j*k_1,k_1)

    psi = (1/np.pi)*np.real(psi)
    u = (1/np.pi)*np.real(u)
    w = -(1/np.pi)*np.real(w)
    return psi, u, w

# psi functions
@jit(parallel=True)
def calc_psi1(xi,zeta,tau,k,U,L):
    m=k/(1+k*U)
    psi1 = (-(1/2)*np.exp(-k*L)/(k**2+(1+k*U)**2)
            *(np.exp(1j*m*zeta)-np.exp(-zeta))
            *np.exp(1j*(k*xi+tau)))
    return psi1

@jit(parallel=True)
def calc_psi2_U0(xi,zeta,tau,k,L):
    m=k
    psi2 = (-(1/2)*np.exp(-k*L)/(k**2+1)
            *(np.exp(-1j*m*zeta)-np.exp(-zeta))
            *np.exp(1j*(k*xi-tau)))
    return psi2

@jit(parallel=True)
def calc_psi2a(xi,zeta,tau,s,alpha,U,L):
    theta = calc_theta(s,alpha)
    k = calc_k_2(theta,U)
    # Note only difference between u2a is sign
    psi2a = (1/(2*U)*calc_C2(xi,tau,k,U,L)*np.exp(-zeta)*np.cos(theta)
             *alpha*s**(alpha-1)*np.pi/2)
    return psi2a

@jit(parallel=True)
def calc_psi2b(xi,zeta,tau,s,alpha,U,L):
    theta = calc_theta(s, alpha=alpha)
    k = calc_k_2(theta,U)
    k0 = calc_k_2(0,U)
    psi2b = (-1/(2*U)*np.exp(1j*zeta/U)
             *(calc_C2(xi,tau,k,U,L)-calc_C2(xi,tau,k0,U,L))
             *np.exp(-1j*zeta/(U*np.sin(theta)))*np.cos(theta)
             *alpha*s**(alpha-1)*np.pi/2)
    return psi2b

@jit(parallel=True)
def calc_psi3a(xi,zeta,tau,s,alpha,U,L):
    theta = calc_theta(s, alpha)
    k = calc_k_3(theta,U)
    psi3a = (U/2*calc_C3(xi,tau,k,U,L)*np.exp(-zeta)*np.cos(theta)
             *alpha*s**(alpha-1)*np.pi/2)
    return psi3a

@jit(parallel=True)
def calc_psi3b(xi,zeta,tau,s,alpha,U,L):
    theta = calc_theta(s, alpha=alpha)
    k = calc_k_3(theta,U)
    k0 = calc_k_3(0,U)
    psi3b = (-U/2*(calc_C3(xi,tau,k,U,L)-calc_C3(xi,tau,k0,U,L))
             *np.exp(1j*zeta/(U*np.sin(theta)))*np.cos(theta)
             *alpha*s**(alpha-1)*np.pi/2)
    return psi3b

# w functions
@jit(parallel=True)
def calc_w2b(xi,zeta,tau,s,alpha,U,L):
    theta = calc_theta(s, alpha=alpha)
    k = calc_k_2(theta,U)
    k0 = calc_k_2(0,U)
    w2b = (-1/(2*U)*np.exp(1j*zeta/U)
             *(calc_C2x(xi,tau,k,U,L)-calc_C2x(xi,tau,k0,U,L))
             *np.exp(-1j*zeta/(U*np.sin(theta)))*np.cos(theta)
             *alpha*s**(alpha-1)*np.pi/2)
    return w2b

@jit(parallel=True)
def calc_w3b(xi,zeta,tau,s,alpha,U,L):
    theta = calc_theta(s, alpha=alpha)
    k = calc_k_3(theta,U)
    k0 = calc_k_3(0,U)
    w3b = (-U/2*(calc_C3x(xi,tau,k,U,L)-calc_C3x(xi,tau,k0,U,L))
             *np.exp(1j*zeta/(U*np.sin(theta)))*np.cos(theta)
             *alpha*s**(alpha-1)*np.pi/2)
    return w3b

# u functions
@jit(parallel=True)
def calc_u1(xi,zeta,tau,k,U,L):
    m=k/(1+k*U)
    u1 = (-(1/2)*np.exp(-k*L)/(k**2+(1+k*U)**2)
          *(1j*m*np.exp(1j*m*zeta)+np.exp(-zeta))
          *np.exp(1j*(k*xi+tau)))
    return u1

@jit(parallel=True)
def calc_u2_U0(xi,zeta,tau,k,L):
    m=k
    u2 = (-(1/2)*np.exp(-k*L)/(k**2+1)
          *(-1j*m*np.exp(-1j*m*zeta)+np.exp(-zeta))
          *np.exp(1j*(k*xi-tau)))
    return u2

@jit(parallel=True)
def calc_u2a(xi,zeta,tau,s,alpha,U,L):
    theta = calc_theta(s,alpha=alpha)
    k = calc_k_2(theta,U)
    u2a = (-1/(2*U)*calc_C2(xi,tau,k,U,L)*np.exp(-zeta)*np.cos(theta)
           *alpha*s**(alpha-1)*np.pi/2)
    return u2a

@jit(parallel=True)
def calc_u2b(xi,zeta,tau,s,alpha,U,L):
    theta = calc_theta(s, alpha=alpha)
    k = calc_k_2(theta,U)
    k_0 = calc_k_2(0,U)
    u2b = (-1j/(2*U**2)*np.exp(1j*zeta/U)
           *(calc_C2(xi,tau,k,U,L)-calc_C2(xi,tau,k_0,U,L))
           *np.exp(-1j*zeta/(np.sin(theta)*U))*np.cos(theta)
           *alpha*s**(alpha-1)*np.pi/2
           +1j/(2*U**2)*np.exp(1j*zeta/U)
           *(calc_C2(xi,tau,k,U,L)-calc_C2(xi,tau,k_0,U,L))
           *np.exp(-1j*zeta/(np.sin(theta)*U))*np.cos(theta)/np.sin(theta)
           *alpha*s**(alpha-1)*np.pi/2)
    return u2b

@jit(parallel=True)
def calc_u2b_z0(xi,tau,s,alpha,U,L):
    theta = calc_theta(s, alpha=alpha)
    k = calc_k_2(theta,U)
    k_0 = calc_k_2(0,U)
    (-1j/(2*U**2)*calc_C2(xi,tau,k,U,L)*np.cos(theta)
     *alpha*s**(alpha-1)*np.pi/2
     +1j/(2*U**2)*calc_C2(xi,tau,k,U,L)*np.cos(theta)/np.sin(theta)
     *alpha*s**(alpha-1)*np.pi/2)

@jit(parallel=True)
def calc_u3a(xi,zeta,tau,s,alpha,U,L):
    theta = calc_theta(s, alpha)
    k = calc_k_3(theta,U)
    u3a = (-U/2*calc_C3(xi,tau,k,U,L)
           *np.exp(-zeta)*np.cos(theta)*alpha*s**(alpha-1)*np.pi/2)
    return u3a

@jit(parallel=True)
def calc_u3b(xi,zeta,tau,s,alpha,U,L):
    theta = calc_theta(s, alpha=alpha)
    k = calc_k_3(theta,U)
    k_0 = calc_k_3(0,U)
    u3b = (-1j/2*(calc_C3(xi,tau,k,U,L)-calc_C3(xi,tau,k_0,U,L))
           *np.exp(1j*zeta/(U*np.sin(theta)))/np.tan(theta)
           *alpha*s**(alpha-1)*np.pi/2)
    return u3b

# Miscellaneous functions
@jit(parallel=True)
def calc_C2(xi,tau,k,U,L):
    C2 = np.exp(-k*L)*np.exp(1j*(k*xi-tau))/(k**2+(U*k-1)**2)
    return C2

@jit(parallel=True)
def calc_C3(xi,tau,k,U,L):
    C3 = k**2*np.exp(-k*L)*np.exp(1j*(k*xi-tau))/(k**2+(U*k-1)**2)
    return C3

@jit(parallel=True)
def calc_C2x(xi,tau,k,U,L):
    C2x = 1j*k*np.exp(-k*L)*np.exp(1j*(k*xi-tau))/(k**2+(U*k-1)**2)
    return C2x

@jit(parallel=True)
def calc_C2_z0(xi,tau,k,U,L):
    C2x = 1j*k*np.exp(-k*L)*np.exp(1j*(k*xi-tau))/(k**2+(U*k-1)**2)
    return C2_z0

@jit(parallel=True)
def calc_C3x(xi,tau,k,U,L):
    C3x = 1j*k**3*np.exp(-k*L)*np.exp(1j*(k*xi-tau))/(k**2+(U*k-1)**2)
    return C3x

@jit(parallel=True)
def calc_theta(s,alpha=3):
    theta = (np.pi/2)*s**alpha
    return theta

@jit(parallel=True)
def calc_k_3(theta,U):
    k = 1/(U*(1-np.sin(theta)))
    return k

@jit(parallel=True)
def calc_k_2(theta,U):
    k = (1-np.sin(theta))/U
    return k

@jit(parallel=True)
def calc_exp1(z,n=100):
#     s = np.float64(0)
#     for i in np.arange(1,n):
#         i_f = np.float64(i)
#         s += (-z)**i_f/(i_f*np.math.factorial(i_f))
#     return -np.euler_gamma-np.log(z)-s
    return exp1(z)

@jit(parallel=True)
def calc_ep0(z):
    # Note this formula only applies if Re(z)>=0!
    return np.exp(z) + z*calc_exp1(-z)

@jit(parallel=True)
def calc_ep1(z):
    return (np.exp(z) + z*calc_ep0(z))/2
