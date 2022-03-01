# Utilities
# from tqdm import tqdm

# Performance
from numba import jit, prange

# Analysis
import numpy as np


@jit(parallel=True, nopython=True)
def integrate_piecewise_N(
        x, z, t, s, alpha, L, R, H1, zN, heat_right=True):

    psi = np.zeros((2, t.size, z.size, x.size), dtype=np.complex64)
    # u = np.zeros((2, t.size, z.size, x.size), dtype=np.complex64)
    # w = np.zeros((2, t.size, z.size, x.size), dtype=np.complex64)
    # bq = np.zeros((t.size, z.size, x.size), dtype=np.complex64)
    # bw = np.zeros((2, t.size, z.size, x.size), dtype=np.complex64)

    # Define alternative domains
    theta = calc_theta(s, alpha=alpha)

    # Get wavenumbers greater than 2*pi
    k_3 = calc_k_3(theta, 1/(2*np.pi))

    # Get wavenumers less than 2*pi/
    k_2 = calc_k_2(theta, 1/(2*np.pi))

    # Create wavenumber domain
    k_1 = np.concatenate((k_2[-1::-1], np.array([1/1/(2*np.pi)]), k_3))

    # Perform numerical integration 0<z<=H1
    for j in prange(1, zN):
        # print('Loop ' + j + ' of ' + z.size + '.')
        for i in range(x.size):
            for l in range(t.size):

                # Calc psi
                psi1_ig = calc_psi1_1(x[i], z[j], t[l], k_1, L, R)
                psi[0][l, j, i] = np.trapz(psi1_ig, k_1)
                psi2_ig = calc_psi2_1(x[i], z[j], t[l], k_1, L, R)
                psi[1][l, j, i] = np.trapz(psi2_ig, k_1)

                # # Calc u
                # u1_ig = calc_u1_1(x[i], z[j], t[l], k_1, L)
                # u[0][l, j, i] = np.trapz(u1_ig, k_1)
                # u2_ig = calc_u2_1(x[i], z[j], t[l], k_1, L)
                # u[1][l, j, i] = np.trapz(u2_ig, k_1)
                #
                # # Calc w
                # w[0][l, j, i] = np.trapz(psi1_ig*1j*k_1, k_1)
                # w[1][l, j, i] = np.trapz(psi2_ig*1j*k_1, k_1)
                #
                # # Calc bw
                # bw[0][l, j, i] = np.trapz(psi1_ig*k_1, k_1)
                # bw[1][l, j, i] = np.trapz(-psi2_ig*k_1, k_1)

    # Perform numerical integration H1<z
    for j in prange(zN, z.size):
        # print('Loop ' + j + ' of ' + z.size + '.')
        for i in range(x.size):
            for l in range(t.size):

                # Calc psi
                psi1_ig = calc_psi1_2_alt(x[i], z[j], t[l], k_1, L, R, H1)
                psi[0][l, j, i] = np.trapz(psi1_ig, k_1)
                psi2_ig = calc_psi2_2_alt(x[i], z[j], t[l], k_1, L, R, H1)
                psi[1][l, j, i] = np.trapz(psi2_ig, k_1)

                # Calc u
                # u1_ig = calc_u1_2_alt(x[i],z[j],t[l],k_1,L,R,H1)
                # u[0][l,j,i] = np.trapz(u1_ig, k_1)
                # u2_ig = calc_u2_2_alt(x[i],z[j],t[l],k_1,L,R,H1)
                # u[1][l,j,i] = np.trapz(u2_ig, k_1)
                #
                # # Calc w
                # w[0][l,j,i] = np.trapz(psi1_ig*1j*k_1,k_1)
                # w[1][l,j,i] = np.trapz(psi2_ig*1j*k_1,k_1)
                #
                # # Calc bw
                # bw[0][l,j,i] = np.trapz(psi1_ig*k_1, k_1)
                # bw[1][l,j,i] = np.trapz(-psi2_ig*k_1, k_1)

    psi = (1/np.pi)*np.real(psi)
    # u = (1/np.pi)*np.real(u)
    # w = -(1/np.pi)*np.real(w)
    # bq = np.real(bq)
    # bw = (1/np.pi)*np.real(bw)

    # return psi, u, w, bq, bw
    return psi


# psi functions
@jit(nopython=True)
def calc_psi1_1(x, z, t, k, L, R):
    psi1_1 = (
        -(1/2)*np.exp(-k*L)/(k**2+1)
        * (np.exp(1j*k*z) - np.exp(-z))
        * np.exp(1j*(k*x+t)))
    return psi1_1


@jit(nopython=True)
def calc_psi2_1(x, z, t, k, L, R):
    psi2_1 = (
        -(1/2)*np.exp(-k*L)/(k**2+1)
        * (np.exp(-1j*k*z)-np.exp(-z))
        * np.exp(1j*(k*x-t)))
    return psi2_1


@jit(nopython=True)
def calc_psi1_2(x, z, t, k, L, R, H1):
    psi1_2 = (
        -1/k/(k**2+1)*np.exp(-L*k)/2
        * (
            (-np.exp(-z)*(np.sin(k*z)+k*np.cos(k*z))+k)
            * np.exp(1j*k/R*z)*np.exp(1j*H1*(k-k/R))
            - (-1j*k-1)*np.exp((1j*k-1)*z)*np.sin(k*H1)
            * np.exp(1j*k/R*H1)*np.exp(-1j*k/R*z))*np.exp(1j*(k*x+t)))
    return psi1_2


@jit(nopython=True)
def calc_psi2_2(x, z, t, k, L, R, H1):
    psi2_2 = (
        -1/k/(k**2+1)*np.exp(-L*k)/2
        * (
            (-np.exp(-z)*(np.sin(k*z)+k*np.cos(k*z))+k)
            * np.exp(-1j*k/R*z)*np.exp(-1j*H1*(k-k/R))
            - (1j*k-1)*np.exp((-1j*k-1)*z)*np.sin(k*H1)
            * np.exp(-1j*k/R*H1)*np.exp(1j*k/R*z))*np.exp(1j*(k*x-t)))
    return psi2_2


@jit(nopython=True)
def calc_psi1_2_alt(x, z, t, k, L, R, H1):
    psi1_2 = (
        -np.exp(-k*L)/(k**2+1)/2*np.exp(1j*k*H1*(1-1/R))
        * np.exp(1j*(k*x+k/R*z+t)))
    return psi1_2


@jit(nopython=True)
def calc_psi2_2_alt(x, z, t, k, L, R, H1):
    psi2_2 = (
        -np.exp(-k*L)/(k**2+1)/2*np.exp(-1j*k*H1*(1-1/R))
        * np.exp(1j*(k*x-k/R*z-t)))
    return psi2_2

#
# # u functions
# @jit(nopython=True)
# def calc_u1_1(x,z,t,k,L):
#     u1_1 = (-(1/2)*np.exp(-k*L)/(k**2+1)
#             *(1j*k*np.exp(1j*k*z)+np.exp(-z))
#             *np.exp(1j*(k*x+t)))
#     return u1_1
#
# @jit(nopython=True)
# def calc_u2_1(x,z,t,k,L):
#     u2_1 = (-(1/2)*np.exp(-k*L)/(k**2+1)
#             *(-1j*k*np.exp(-1j*k*z)+np.exp(-z))
#             *np.exp(1j*(k*x-t)))
#     return u2_1
#
# # u functions
# # @jit(nopython=True)
# def calc_u1_2(x,z,t,k,L,R,H1):
#     u1_2 = ((-R*(k**2 + 1)
#              *np.exp((R*z + 3.0*1j*k*z + 1.0*1j*k*H1*(R - 1))/R)
#              *np.sin(k*z)
#              -R*(1.0*1j*k - 1)*(1.0*1j*k + 1)
#              *np.exp((R*z*(1.0*1j*k - 1)
#                   + 2*R*z + 1.0*1j*k*z
#                   + 1.0*1j*k*H1)/R)
#              *np.sin(k*H1) + 1j*k*(1.0*1j*k + 1)
#              *np.exp((R*z*(1.0*1j*k - 1) + 2*R*z + 1.0*1j*k*z
#                    + 1.0*1j*k*H1)/R)
#              *np.sin(k*H1)
#              -1.0*1j*k*(k*np.exp(z)-k*np.cos(k*z)-np.sin(k*z))
#              *np.exp((R*z + 3.0*1j*k*z + 1.0*1j*k*H1*(R - 1))/R))
#             *np.exp((-L*R*k - 2*R*z - 2.0*1j*k*z)/R)
#             /(2*R*k*(k**2 + 1))
#             *np.exp(1j*(k*x+t)))
#     return u1_2
#
# # @jit(nopython=True)
# def calc_u2_2(x,z,t,k,L,R,H1):
#     u2_2 = ((-R*(k**2 + 1)
#              *np.exp((2*R*z*(1.0*1j*k + 1) + R*z + 1.0*1j*k*z
#                       + 1.0*1j*k*H1*(R - 1) + 2.0*1j*k*H1)/R)
#              *np.sin(k*z)
#              -R*(1.0*1j*k - 1)*(1.0*1j*k + 1)
#              *np.exp((R*z*(1.0*1j*k + 1) + 2*R*z + 3.0*1j*k*z
#                       +2.0*1j*k*H1*(R - 1) + 1.0*1j*k*H1)/R)
#              *np.sin(k*H1)
#              +1j*k*(1.0*1j*k - 1)
#              *np.exp((R*z*(1.0*1j*k + 1) + 2*R*z
#                       +3.0*1j*k*z+2.0*1j*k*H1*(R - 1)+1.0*1j*k*H1)/R)
#              *np.sin(k*H1)
#              +1j*k*(k*np.exp(z) - k*np.cos(k*z) - np.sin(k*z))
#              *np.exp((2*R*z*(1.0*1j*k + 1) + R*z + 1.0*1j*k*z
#                       +1.0*1j*k*H1*(R - 1) + 2.0*1j*k*H1)/R))
#             *np.exp((-L*R*k - 2*R*z*(1.0*1j*k + 1) - 2*R*z - 2.0*1j*k*z
#                      -2.0*1j*k*H1*(R - 1) - 2.0*1j*k*H1)/R)
#             /(2*R*k*(k**2 + 1))
#             *np.exp(1j*(k*x-t)))
#     return u2_2
#
# @jit(nopython=True)
# def calc_u1_2_alt(x,z,t,k,L,R,H1):
#     psi1_2 = (-np.exp(-k*L)/(k**2+1)/2*np.exp(1j*k*H1*(1-1/R))
#               *1j*k/R*np.exp(1j*(k*x+k/R*z+t)))
#     return psi1_2
#
# @jit(nopython=True)
# def calc_u2_2_alt(x,z,t,k,L,R,H1):
#     psi2_2 = (-np.exp(-k*L)/(k**2+1)/2*np.exp(-1j*k*H1*(1-1/R))
#               *(-1j*k/R)*np.exp(1j*(k*x-k/R*z-t)))
#     return psi2_2


@jit(nopython=True)
def calc_theta(s, alpha=3):
    theta = (np.pi/2)*s**alpha
    return theta


@jit(nopython=True)
def calc_k_3(theta, U):
    k = 1/(U*(1-np.sin(theta)))
    return k


@jit(nopython=True)
def calc_k_2(theta, U):
    k = (1-np.sin(theta))/U
    return k
