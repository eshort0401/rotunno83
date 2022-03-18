# Utilities
# from tqdm import tqdm

# Performance
from numba import jit, prange

# Analysis
import numpy as np


# @jit(parallel=True, nopython=False)
def integrate_piecewise_N(
        x, z, t, s, alpha, L, N, H1, zN, A):

    psi = np.zeros((2, 2, t.size, z.size, x.size), dtype=np.complex64)
    u = np.zeros((2, 2, t.size, z.size, x.size), dtype=np.complex64)
    w = np.zeros((2, 2, t.size, z.size, x.size), dtype=np.complex64)
    # bq = np.zeros((t.size, z.size, x.size), dtype=np.complex64)
    # bw = np.zeros((2, t.size, z.size, x.size), dtype=np.complex64)

    # Define alternative domains
    theta = calc_theta(s, alpha=alpha)

    # Get wavenumbers greater than 2*pi
    k_3 = calc_k_3(theta, 1/(2*np.pi))

    # Get wavenumbers less than 2*pi/
    k_2 = calc_k_2(theta, 1/(2*np.pi))

    # Create wavenumber domain
    k = np.concatenate((k_2[-1::-1], np.array([2*np.pi]), k_3))

    print('Beginning integration.')
    print('Integrating lower sub-domain.')

    # Perform numerical integration 0<z<=H1
    for j in range(0, zN):

        psi_base_1_lf, psi_base_1_hf = calc_psi_base_lower(
            z[j], k, L, N, H1, A)
        psi_base_2_lf, psi_base_2_hf = calc_psi_base_lower(
            z[j], -k, L, N, H1, A)
        u_base_1_lf, u_base_1_hf = calc_u_base_lower(
            z[j], k, L, N, H1, A)
        u_base_2_lf, u_base_2_hf = calc_u_base_lower(
            z[j], -k, L, N, H1, A)
        base_fns = [
            psi_base_1_lf, psi_base_1_hf, psi_base_2_lf, psi_base_2_hf,
            u_base_1_lf, u_base_1_hf, u_base_2_lf, u_base_2_hf]
        base_fns = [fn*np.exp(-L*k)/(2*A**2) for fn in base_fns]

        psi, u, w = loop(
            psi, u, w, k, j,
            psi_base_1_lf, psi_base_1_hf, psi_base_2_lf, psi_base_2_hf,
            u_base_1_lf, u_base_1_hf, u_base_2_lf, u_base_2_hf, x, t)

    print('Integrating upper sub-domain.')
    # Perform numerical integration H1<z
    for j in range(zN, z.size):

        psi_base_1_lf, psi_base_1_hf = calc_psi_base_upper(
            z[j], k, L, N, H1, A)
        psi_base_2_lf, psi_base_2_hf = calc_psi_base_upper(
            z[j], -k, L, N, H1, A)
        u_base_1_lf, u_base_1_hf = calc_u_base_upper(
            z[j], k, L, N, H1, A)
        u_base_2_lf, u_base_2_hf = calc_u_base_upper(
            z[j], -k, L, N, H1, A)

        base_fns = [
            psi_base_1_lf, psi_base_1_hf, psi_base_2_lf, psi_base_2_hf,
            u_base_1_lf, u_base_1_hf, u_base_2_lf, u_base_2_hf]
        base_fns = [fn*np.exp(-L*k)/(2*A**2) for fn in base_fns]

        psi, u, w = loop(
            psi, u, w, k, j,
            psi_base_1_lf, psi_base_1_hf, psi_base_2_lf, psi_base_2_hf,
            u_base_1_lf, u_base_1_hf, u_base_2_lf, u_base_2_hf, x, t)

    psi = (1/np.pi)*np.real(psi)
    u = (1/np.pi)*np.real(u)
    w = -(1/np.pi)*np.real(w)
    # bq = np.real(bq)
    # bw = (1/np.pi)*np.real(bw)

    # return psi, u, w, bq, bw
    return psi, u, w


@jit(nopython=True)
def loop(
        psi, u, w, k, j,
        psi_base_1_lf, psi_base_1_hf, psi_base_2_lf, psi_base_2_hf,
        u_base_1_lf, u_base_1_hf, u_base_2_lf, u_base_2_hf, x, t):

    pos_fns = [psi_base_1_lf, psi_base_1_hf, u_base_1_lf, u_base_1_hf]
    neg_fns = [psi_base_2_lf, psi_base_2_hf, u_base_2_lf, u_base_2_hf]

    for i in range(x.size):
        for l in range(t.size):

            [psi1_ig_lf, psi1_ig_hf, u1_ig_lf, u1_ig_hf] = [
                fn*np.exp(1j*(k*x[i]+t[l])) for fn in pos_fns]
            [psi2_ig_lf, psi2_ig_hf, u2_ig_lf, u2_ig_hf] = [
                fn*np.exp(1j*(k*x[i]-t[l])) for fn in neg_fns]

            psi[0][0][l, j, i] = np.trapz(psi1_ig_lf, k)
            psi[1][0][l, j, i] = np.trapz(psi2_ig_lf, k)
            psi[0][1][l, j, i] = np.trapz(psi1_ig_hf, k)
            psi[1][1][l, j, i] = np.trapz(psi2_ig_hf, k)

            # Calc u
            u[0][0][l, j, i] = np.trapz(u1_ig_lf, k)
            u[1][0][l, j, i] = np.trapz(u2_ig_lf, k)
            u[0][1][l, j, i] = np.trapz(u1_ig_hf, k)
            u[1][1][l, j, i] = np.trapz(u2_ig_hf, k)

            # Calc w
            w[0][0][l, j, i] = np.trapz(psi1_ig_lf*1j*k, k)
            w[1][0][l, j, i] = np.trapz(psi2_ig_lf*1j*k, k)
            w[0][1][l, j, i] = np.trapz(psi1_ig_hf*1j*k, k)
            w[1][1][l, j, i] = np.trapz(psi2_ig_hf*1j*k, k)
    return psi, u, w


@jit(nopython=True)
def calc_psi_base_lower(z, k, L, N, H1, A):
    m = k/A
    f1 = (N+1)*np.exp(-1j*m*H1)
    f2 = (1-N)*np.exp(1j*m*H1)
    g2 = np.cos(m*H1)-1j*N*np.sin(m*H1)
    # P = np.exp(1j*m*H1*N)*g2
    P = f1+f2

    term_1 = -1/(m*P)*(-np.exp(-z)*(np.sin(m*z)+m*np.cos(m*z))+m)/(m**2+1)
    term_1 = term_1*(f1*np.exp(1j*m*z)+f2*np.exp(-1j*m*z))

    term_2 = (
        - 1/(m*P)*f1/(1j*m-1)*(np.exp((1j*m-1)*H1)-np.exp((1j*m-1)*z))
        - 1/(m*P)*f2/(-1j*m-1)*(np.exp((-1j*m-1)*H1)-np.exp((-1j*m-1)*z)))
    term_2 = term_2 * np.sin(m*z)

    term_3 = -1/m*1/g2*(-np.exp(-H1)/(1j*m*N-1))
    term_3 = term_3*np.sin(m*z)

    return (term_1+term_2), term_3
    # return term_3


# positive t mode, upper subdomain
@jit(nopython=True)
def calc_psi_base_upper(z, k, L, N, H1, A):

    m = k/A
    f1 = (N+1)/2*np.exp(1j*m*H1*(N-1))
    f2 = (1-N)/2*np.exp(1j*m*H1*(N+1))
    P = f1 + f2
    g1 = np.cos(m*H1)+1j*N*np.sin(m*H1)
    g2 = np.cos(m*H1)-1j*N*np.sin(m*H1)

    term_1 = -1/(m*P)*np.exp(1j*m*N*z)
    term_1 = term_1*(-np.exp(-H1)*(np.sin(m*H1)+m*np.cos(m*H1))+m)/(m**2+1)

    term_2_a = 1/(-1j*m*N-1)*(np.exp((-1j*m*N-1)*z)-np.exp((-1j*m*N-1)*H1))
    term_2_b = - (g1/g2)*np.exp(-2*1j*m*N*H1)/(1j*m*N-1)
    term_2_b = term_2_b*(np.exp((1j*m*N-1)*z)-np.exp((1j*m*N-1)*H1))
    term_2 = (term_2_a+term_2_b)*np.exp(1j*m*N*z)/(2*1j*m*N)

    term_3 = 1/(2*1j*N)*(g1*np.exp(1j*m*N*(z-H1))-g2*np.exp(-1j*m*N*(z-H1)))
    term_3 = -term_3/(m*g2)/(1j*m*N-1)*(-np.exp((1j*m*N-1)*z-1j*m*N*H1))

    return term_1, (term_2+term_3)
    # return term_2+term_3


@jit(nopython=True)
def calc_u_base_lower(z, k, L, N, H1, A):
    m = k/A
    f1 = (N+1)/2*np.exp(1j*m*H1*(N-1))
    f2 = (1-N)/2*np.exp(1j*m*H1*(N+1))
    g2 = np.cos(m*H1)-1j*N*np.sin(m*H1)
    P = np.exp(1j*m*H1*N)*g2

    # Simplified
    term_1_a = -(-np.exp(-z)*(np.sin(m*z)+m*np.cos(m*z))+m)/(m**2+1)
    term_1_a = term_1_a*1j*(f1*np.exp(1j*m*z)-f2*np.exp(-1j*m*z))
    term_1_b = -1/m*np.sin(m*z)*np.exp(-z)
    term_1_b = term_1_b*(f1*np.exp(1j*m*z)+f2*np.exp(-1j*m*z))
    term_1 = (term_1_a + term_1_b)/P

    term_2_a = (
        - 1/m*(f1/P)*(-np.exp((1j*m-1)*z))
        - 1/m*(f2/P)*(-np.exp((-1j*m-1)*z)))
    term_2_a = term_2_a * np.sin(m*z)
    term_2_b = (
        - 1/m*(f1/P)/(1j*m-1)*(np.exp((1j*m-1)*H1)-np.exp((1j*m-1)*z))
        - 1/m*(f2/P)/(-1j*m-1)*(np.exp((-1j*m-1)*H1)-np.exp((-1j*m-1)*z)))
    term_2_b = term_2_b*m*np.cos(m*z)
    term_2 = term_2_a+term_2_b

    term_3 = -1/m*1/(np.cos(m*H1)-1j*N*np.sin(m*H1))*(-np.exp(-H1)/(1j*m*N-1))
    term_3 = term_3*m*np.cos(m*z)

    return (term_1+term_2), term_3


# positive t mode, upper subdomain
@jit(nopython=True)
def calc_u_base_upper(z, k, L, N, H1, A):
    m = k/A
    g1 = np.cos(m*H1)+1j*N*np.sin(m*H1)
    g2 = np.cos(m*H1)-1j*N*np.sin(m*H1)
    P = np.exp(1j*m*H1*N)*g2

    term_1 = -1/(m*P)*1j*m*N*np.exp(1j*m*N*z)
    term_1 = term_1*(-np.exp(-H1)*(np.sin(m*H1)+m*np.cos(m*H1))+m)/(m**2+1)

    term_2_a = 1/(-1j*m*N-1)*(-np.exp(-z)-np.exp(1j*m*N*(z-H1)-H1)*(1j*m*N))

    term_2_b = - (g1/g2)*np.exp(-2*1j*m*N*H1)/(1j*m*N-1)
    term_2_b = term_2_b*(
        np.exp((2*1j*m*N-1)*z)*(2*1j*m*N-1)
        - np.exp(1j*m*N*(z+H1)-H1)*(1j*m*N))
    term_2 = (term_2_a+term_2_b)/(2*1j*m*N)

    term_3_a = 1/(2*1j*N)*(
        g1*np.exp(1j*m*N*(z-H1))*1j*m*N
        - g2*np.exp(-1j*m*N*(z-H1))*(-1j*m*N))
    term_3_a = -term_3_a/(m*g2)/(1j*m*N-1)*(-np.exp((1j*m*N-1)*z-1j*m*N*H1))

    term_3_b = 1/(2*1j*N)*(g1*np.exp(1j*m*N*(z-H1))-g2*np.exp(-1j*m*N*(z-H1)))
    term_3_b = -term_3_b/(m*g2)/(1j*m*N-1)*(-np.exp((1j*m*N-1)*z-1j*m*N*H1))
    term_3_b = term_3_b*(1j*m*N-1)
    term_3 = term_3_a + term_3_b

    return term_1, (term_2+term_3)


@jit(nopython=True)
def calc_theta(s, alpha=2):
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
