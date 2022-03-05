# Utilities
# from tqdm import tqdm

# Performance
from numba import jit, prange

# Analysis
import numpy as np


@jit(parallel=True, nopython=True)
def integrate_piecewise_N(
        x, z, t, s, alpha, L, N, H1, zN, A, lim=False):

    psi = np.zeros((2, t.size, z.size, x.size), dtype=np.complex64)
    u = np.zeros((2, t.size, z.size, x.size), dtype=np.complex64)
    w = np.zeros((2, t.size, z.size, x.size), dtype=np.complex64)
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
    for j in prange(0, zN):
        # print('Integrating height level ' + j + ' of ' + z.size + '.')
        for i in range(x.size):
            for l in range(t.size):

                # Calc psi
                psi1_ig, psi2_ig = calc_psi_lower(
                    x[i], z[j], t[l], k, L, N, H1, A, lim)
                psi[0][l, j, i] = np.trapz(psi1_ig, k)
                psi[1][l, j, i] = np.trapz(psi2_ig, k)

                # # Calc u
                u1_ig, u2_ig = calc_u_lower(
                    x[i], z[j], t[l], k, L, N, H1, A, lim)
                u[0][l, j, i] = np.trapz(u1_ig, k)
                u[1][l, j, i] = np.trapz(u2_ig, k)

                # Calc w
                w[0][l, j, i] = np.trapz(psi1_ig*1j*k, k)
                w[1][l, j, i] = np.trapz(psi2_ig*1j*k, k)

                # # Calc bw
                # bw[0][l, j, i] = np.trapz(psi1_ig*k_1, k_1)
                # bw[1][l, j, i] = np.trapz(-psi2_ig*k_1, k_1)

    print('Integrating upper sub-domain.')
    if lim is False:
        print('Integrating upper sub-domain.')
        # Perform numerical integration H1<z
        for j in prange(zN, z.size):
            # print('Integrating height level ' + j + ' of ' + z.size + '.')
            for i in range(x.size):
                for l in range(t.size):

                    # Calc psi
                    psi1_ig, psi2_ig = calc_psi_upper(
                        x[i], z[j], t[l], k, L, N, H1, A)
                    psi[0][l, j, i] = np.trapz(psi1_ig, k)
                    psi[1][l, j, i] = np.trapz(psi2_ig, k)

                    # Calc u
                    u1_ig, u2_ig = calc_u_upper(
                        x[i], z[j], t[l], k, L, N, H1, A)
                    u[0][l, j, i] = np.trapz(u1_ig, k)
                    u[1][l, j, i] = np.trapz(u2_ig, k)

                    # Calc w
                    w[0][l, j, i] = np.trapz(psi1_ig*1j*k, k)
                    w[1][l, j, i] = np.trapz(psi2_ig*1j*k, k)
                    #
                    # # Calc bw
                    # bw[0][l,j,i] = np.trapz(psi1_ig*k_1, k_1)
                    # bw[1][l,j,i] = np.trapz(-psi2_ig*k_1, k_1)

    psi = (1/np.pi)*np.real(psi)
    u = (1/np.pi)*np.real(u)
    w = -(1/np.pi)*np.real(w)
    # bq = np.real(bq)
    # bw = (1/np.pi)*np.real(bw)

    # return psi, u, w, bq, bw
    return psi, u, w


@jit(nopython=True)
def calc_psi_terms_lower(x, z, t, k, L, N, H1, A):
    m = k/A
    f1 = (N+1)/2*np.exp(1j*m*H1*(N-1))
    f2 = (1-N)/2*np.exp(1j*m*H1*(N+1))
    g2 = np.cos(m*H1)-1j*N*np.sin(m*H1)
    P = np.exp(1j*m*H1*N)*g2

    term_1 = -1/(m*P)*(-np.exp(-z)*(np.sin(m*z)+m*np.cos(m*z))+m)/(m**2+1)
    term_1 = term_1*(f1*np.exp(1j*m*z)+f2*np.exp(-1j*m*z))

    term_2 = (
        - 1/(m*P)*f1/(1j*m-1)*(np.exp((1j*m-1)*H1)-np.exp((1j*m-1)*z))
        - 1/(m*P)*f2/(-1j*m-1)*(np.exp((-1j*m-1)*H1)-np.exp((-1j*m-1)*z)))
    term_2 = term_2 * np.sin(m*z)

    term_3 = -1/m*1/g2*(-np.exp(-H1)/(1j*m*N-1))
    term_3 = term_3*np.sin(m*z)

    return term_1, term_2, term_3


# Calculate psi lower for N infinite
@jit(nopython=True)
def calc_psi_terms_lower_lim(x, z, t, k, L, N, H1, A):
    m = k/A

    # Note numerical issues for when denom is zero... Need to simplify?
    # Tested, and no obvious way to simplify as we have a sin(m*H1) term
    # in denominator, which messes up the numerical integral.
    c1 = np.exp(-1j*m*H1)/(np.exp(-1j*m*H1)-np.exp(1j*m*H1))
    c2 = -np.exp(1j*m*H1)/(np.exp(-1j*m*H1)-np.exp(1j*m*H1))

    term_1 = -1/m*(-np.exp(-z)*(np.sin(m*z)+m*np.cos(m*z))+m)/(m**2+1)
    term_1 = term_1*(c1*np.exp(1j*m*z)+c2*np.exp(-1j*m*z))

    term_2 = (
        - 1/m*c1/(1j*m-1)*(np.exp((1j*m-1)*H1)-np.exp((1j*m-1)*z))
        - 1/m*c2/(-1j*m-1)*(np.exp((-1j*m-1)*H1)-np.exp((-1j*m-1)*z)))
    term_2 = term_2 * np.sin(m*z)

    return term_1, term_2


# positive t mode, upper subdomain
@jit(nopython=True)
def calc_psi_terms_upper(x, z, t, k, L, N, H1, A):

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

    return term_1, term_2, term_3

    # Simplified equations
    # m = k/A
    # g1 = np.cos(m*H1)+1j*N*np.sin(m*H1)
    # g2 = np.cos(m*H1)-1j*N*np.sin(m*H1)
    # P = np.exp(1j*m*H1*N)*g2
    #
    # term_1 = -1/(m*P)*np.exp(1j*m*N*z)
    # term_1 = term_1*(-np.exp(-H1)*(np.sin(m*H1)+m*np.cos(m*H1))+m)/(m**2+1)
    #
    # term_2_a = 1/(-1j*m*N-1)*(np.exp((-1j*m*N-1)*z)-np.exp((-1j*m*N-1)*H1))
    # term_2_b = - (g1/g2)*np.exp(-2*1j*m*N*H1)/(1j*m*N-1)
    # term_2_b = term_2_b*(np.exp((1j*m*N-1)*z)-np.exp((1j*m*N-1)*H1))
    # term_2 = (term_2_a+term_2_b)*np.exp(1j*m*N*z)/(2*1j*m*N)
    #
    # term_3 = 1/(2*1j*N)*((g1/g2)*np.exp(1j*m*N*(z-H1))-np.exp(-1j*m*N*(z-H1)))
    # term_3 = -term_3/m/(1j*m*N-1)*(-np.exp((1j*m*N*(z-H1)-z)))

    return term_1, term_2, term_3


@jit(nopython=True)
def calc_u_terms_lower_lim(x, z, t, k, L, N, H1, A):
    m = k/A
    c1 = np.exp(-1j*m*H1)/(np.exp(-1j*m*H1)-np.exp(1j*m*H1))
    c2 = -np.exp(1j*m*H1)/(np.exp(-1j*m*H1)-np.exp(1j*m*H1))

    # Simplified
    term_1_a = -(-np.exp(-z)*(np.sin(m*z)+m*np.cos(m*z))+m)/(m**2+1)
    term_1_a = term_1_a*1j*(c1*np.exp(1j*m*z)-c2*np.exp(-1j*m*z))
    term_1_b = -1/m*np.sin(m*z)*np.exp(-z)
    term_1_b = term_1_b*(c1*np.exp(1j*m*z)+c2*np.exp(-1j*m*z))
    term_1 = term_1_a + term_1_b

    term_2_a = (
        - 1/m*c1*(-np.exp((1j*m-1)*z))
        - 1/m*c2*(-np.exp((-1j*m-1)*z)))
    term_2_a = term_2_a * np.sin(m*z)
    term_2_b = (
        - 1/m*c1/(1j*m-1)*(np.exp((1j*m-1)*H1)-np.exp((1j*m-1)*z))
        - 1/m*c2/(-1j*m-1)*(np.exp((-1j*m-1)*H1)-np.exp((-1j*m-1)*z)))
    term_2_b = term_2_b*m*np.cos(m*z)
    term_2 = term_2_a+term_2_b

    # term_1_a = -1/(m*P)*(-np.exp(-z)*(np.sin(m*z)+m*np.cos(m*z))+m)/(m**2+1)
    # term_1_a = term_1_a*1j*m*(f1*np.exp(1j*m*z)-f2*np.exp(-1j*m*z))
    # term_1_b = -1/(m*P)*np.sin(m*z)*np.exp(-z)
    # term_1_b = term_1_b*(f1*np.exp(1j*m*z)+f2*np.exp(-1j*m*z))
    # term_1 = term_1_a + term_1_b
    #
    # term_2_a = (
    #     - 1/(m*P)*f1*(-np.exp((1j*m-1)*z))
    #     - 1/(m*P)*f2*(-np.exp((-1j*m-1)*z)))
    # term_2_a = term_2_a * np.sin(m*z)
    # term_2_b = (
    #     - 1/(m*P)*f1/(1j*m-1)*(np.exp((1j*m-1)*H1)-np.exp((1j*m-1)*z))
    #     - 1/(m*P)*f2/(-1j*m-1)*(np.exp((-1j*m-1)*H1)-np.exp((-1j*m-1)*z)))
    # term_2_b = term_2_b*m*np.cos(m*z)
    # term_2 = term_2_a+term_2_b
    #
    # term_3 = -1/m*1/(np.cos(m*H1)-1j*N*np.sin(m*H1))*(-np.exp(-H1)/(1j*m*N-1))
    # term_3 = term_3*m*np.cos(m*z)

    # REFERENCE
    # term_1 = -1/(m*P)*(-np.exp(-z)*(np.sin(m*z)+m*np.cos(m*z))+m)/(m**2+1)
    # term_1 = term_1*(f1*np.exp(1j*m*z)+f2*np.exp(-1j*m*z))
    #
    # term_2 = (
    #     - 1/(m*P)*f1/(1j*m-1)*(np.exp((1j*m-1)*H1)-np.exp((1j*m-1)*z))
    #     - 1/(m*P)*f2/(-1j*m-1)*(np.exp((-1j*m-1)*H1)-np.exp((-1j*m-1)*z)))
    # term_2 = term_2 * np.sin(m*z)
    #
    # term_3 = -1/m*1/(np.cos(m*H1)-1j*N*np.sin(m*H1))*(-np.exp(-H1)/(1j*m*N-1))
    # term_3 = term_3*np.sin(m*z)

    return term_1, term_2


@jit(nopython=True)
def calc_u_terms_lower(x, z, t, k, L, N, H1, A):
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

    # term_1_a = -1/(m*P)*(-np.exp(-z)*(np.sin(m*z)+m*np.cos(m*z))+m)/(m**2+1)
    # term_1_a = term_1_a*1j*m*(f1*np.exp(1j*m*z)-f2*np.exp(-1j*m*z))
    # term_1_b = -1/(m*P)*np.sin(m*z)*np.exp(-z)
    # term_1_b = term_1_b*(f1*np.exp(1j*m*z)+f2*np.exp(-1j*m*z))
    # term_1 = term_1_a + term_1_b
    #
    # term_2_a = (
    #     - 1/(m*P)*f1*(-np.exp((1j*m-1)*z))
    #     - 1/(m*P)*f2*(-np.exp((-1j*m-1)*z)))
    # term_2_a = term_2_a * np.sin(m*z)
    # term_2_b = (
    #     - 1/(m*P)*f1/(1j*m-1)*(np.exp((1j*m-1)*H1)-np.exp((1j*m-1)*z))
    #     - 1/(m*P)*f2/(-1j*m-1)*(np.exp((-1j*m-1)*H1)-np.exp((-1j*m-1)*z)))
    # term_2_b = term_2_b*m*np.cos(m*z)
    # term_2 = term_2_a+term_2_b
    #
    # term_3 = -1/m*1/(np.cos(m*H1)-1j*N*np.sin(m*H1))*(-np.exp(-H1)/(1j*m*N-1))
    # term_3 = term_3*m*np.cos(m*z)

    # REFERENCE
    # term_1 = -1/(m*P)*(-np.exp(-z)*(np.sin(m*z)+m*np.cos(m*z))+m)/(m**2+1)
    # term_1 = term_1*(f1*np.exp(1j*m*z)+f2*np.exp(-1j*m*z))
    #
    # term_2 = (
    #     - 1/(m*P)*f1/(1j*m-1)*(np.exp((1j*m-1)*H1)-np.exp((1j*m-1)*z))
    #     - 1/(m*P)*f2/(-1j*m-1)*(np.exp((-1j*m-1)*H1)-np.exp((-1j*m-1)*z)))
    # term_2 = term_2 * np.sin(m*z)
    #
    # term_3 = -1/m*1/(np.cos(m*H1)-1j*N*np.sin(m*H1))*(-np.exp(-H1)/(1j*m*N-1))
    # term_3 = term_3*np.sin(m*z)

    return term_1, term_2, term_3


# positive t mode, upper subdomain
@jit(nopython=True)
def calc_u_terms_upper(x, z, t, k, L, N, H1, A):
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

    # Simplified equations
    # term_1 = -1/P*1j*N*np.exp(1j*m*N*z)
    # term_1 = term_1*(-np.exp(-H1)*(np.sin(m*H1)+m*np.cos(m*H1))+m)/(m**2+1)
    #
    # np.exp(1j*m*N*z)
    #
    # term_2_a = (1j*m*N-1)/(m**2*N**2+1)/(2*1j*m*N)*(
    #     -np.exp((-1j*m*N-1)*z)-np.exp((-1j*m*N-1)*H1)*(1j*m*N))
    #
    # term_2_b = - (g1/g2)*np.exp(-2*1j*m*N*H1)
    # term_2_b = term_2_b*(-1j*m*N-1)/(m**2*N**2+1)/(2*1j*m*N)
    # term_2_b = term_2_b*(
    #     np.exp((1j*m*N-1)*z)*(2*1j*m*N-1)
    #     - np.exp((1j*m*N-1)*H1)*(1j*m*N))
    # term_2 = (term_2_a+term_2_b)*np.exp(1j*m*N*z)
    #
    # term_3_a = 1/2*((g1/g2)*np.exp(1j*m*N*(z-H1)) + np.exp(-1j*m*N*(z-H1)))
    # term_3_a = -term_3_a/(1j*m*N-1)
    #
    # term_3_b = -1/(2*1j*m*N)*(
    #     (g1/g2)*np.exp(1j*m*N*(z-H1))-np.exp(-1j*m*N*(z-H1)))
    # term_3 = (term_3_a + term_3_b)*(-np.exp(1j*m*N*(z-H1)-z))
    #
    # term_3 = 0
    # term_2 = 0

    return term_1, term_2, term_3


# psi functions
@jit(nopython=True)
def calc_psi_lower(x, z, t, k, L, N, H1, A, lim=False):

    if lim:
        term_1, term_2 = calc_psi_terms_lower_lim(x, z, t, k, L, N, H1, A)
        base_1 = term_1+term_2
        term_1, term_2 = calc_psi_terms_lower_lim(x, z, t, -k, L, N, H1, A)
        base_2 = term_1+term_2
    else:
        term_1, term_2, term_3 = calc_psi_terms_lower(x, z, t, k, L, N, H1, A)
        base_1 = term_1+term_2+term_3
        term_1, term_2, term_3 = calc_psi_terms_lower(x, z, t, -k, L, N, H1, A)
        base_2 = term_1+term_2+term_3

    psi1 = np.exp(-L*k)/(2*A**2)*(base_1)*np.exp(1j*(k*x+t))
    psi2 = np.exp(-L*k)/(2*A**2)*(base_2)*np.exp(1j*(k*x-t))
    return psi1, psi2


@jit(nopython=True)
def calc_psi_upper(x, z, t, k, L, N, H1, A):
    term_1, term_2, term_3 = calc_psi_terms_upper(x, z, t, k, L, N, H1, A)
    psi1 = np.exp(-L*k)/(2*A**2)*(term_1+term_2+term_3)*np.exp(1j*(k*x+t))
    term_1, term_2, term_3 = calc_psi_terms_upper(x, z, t, -k, L, N, H1, A)
    psi2 = np.exp(-L*k)/(2*A**2)*(term_1+term_2+term_3)*np.exp(1j*(k*x-t))
    return psi1, psi2


# u functions
@jit(nopython=True)
def calc_u_lower(x, z, t, k, L, N, H1, A, lim=False):

    if lim:
        term_1, term_2 = calc_u_terms_lower_lim(x, z, t, k, L, N, H1, A)
        base_1 = term_1+term_2
        term_1, term_2 = calc_u_terms_lower_lim(x, z, t, -k, L, N, H1, A)
        base_2 = term_1+term_2
    else:
        term_1, term_2, term_3 = calc_u_terms_lower(x, z, t, k, L, N, H1, A)
        base_1 = term_1+term_2+term_3
        term_1, term_2, term_3 = calc_u_terms_lower(x, z, t, -k, L, N, H1, A)
        base_2 = term_1+term_2+term_3

    u1 = np.exp(-L*k)/(2*A**2)*(base_1)*np.exp(1j*(k*x+t))
    u2 = np.exp(-L*k)/(2*A**2)*(base_2)*np.exp(1j*(k*x-t))

    return u1, u2


@jit(nopython=True)
def calc_u_upper(x, z, t, k, L, N, H1, A):
    term_1, term_2, term_3 = calc_u_terms_upper(x, z, t, k, L, N, H1, A)
    u1 = np.exp(-L*k)/(2*A**2)*(term_1+term_2+term_3)*np.exp(1j*(k*x+t))
    term_1, term_2, term_3 = calc_u_terms_upper(x, z, t, -k, L, N, H1, A)
    u2 = np.exp(-L*k)/(2*A**2)*(term_1+term_2+term_3)*np.exp(1j*(k*x-t))
    return u1, u2


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
