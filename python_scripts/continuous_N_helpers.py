# Utilities
# from tqdm import tqdm

# Performance
from numba import jit, prange

# Analysis
import numpy as np
import mpmath

pcfd = np.frompyfunc(mpmath.pcfd, 2, 1)
besselj = np.frompyfunc(mpmath.besselj, 2, 1)


# @jit(parallel=True, nopython=True)
def integrate_continuous_N(
        x, z, t, s, alpha, L, N, H1, H2, zN, zN_scaled, A0):

    psi = np.zeros((2, t.size, z.size, x.size), dtype=np.complex64)
    u = np.zeros((2, t.size, z.size, x.size), dtype=np.complex64)
    w = np.zeros((2, t.size, z.size, x.size), dtype=np.complex64)

    zN_scaled = int(2+np.ceil(zN*(H2-H1)/H1))

    # Define alternative domains
    theta_p = calc_theta_p(s, alpha=alpha)

    # Get wavenumbers greater than 2*pi
    k_3 = calc_k_3(theta_p, 1/(2*np.pi))

    # Get wavenumbers less than 2*pi/
    k_2 = calc_k_2(theta_p, 1/(2*np.pi))

    # Create wavenumber domain
    k = np.concatenate((k_2[-1::-1], np.array([2*np.pi]), k_3))

    print('Calculating Coefficients')
    X_p, Y_p, beta_p, X_n, Y_n, beta_n, Da_2, Db_2 = calc_coefficients(
        k, N, H1, H2, A0)

    # T = 1/2*((np.sin(m*H1)-gamma)*(Da_2/Da_1)+gamma*(Db_2/Db_1))
    # S = 1/(2*1j*m*N)*(
    #     (np.sin(m*H1)-gamma)*(Da_2/Da_1)*(-alpha_2+dA_2/A_2)
    #     + gamma*(Db_2/Db_1)*(alpha_2+dA_2/A_2))

    print('Beginning integration.')
    print('Integrating lower sub-domain.')

    # Perform numerical integration 0<z<=H1
    for j in range(0, zN):
        psi_base_1 = calc_psi_base_lower(
            z[j], k, N, H1, A0, X_p, Y_p)*np.exp(-L*k)/(2*A0**2)
        psi_base_2 = calc_psi_base_lower(
            z[j], k, N, H1, A0, X_n, Y_n)*np.exp(-L*k)/(2*A0**2)
        u_base_1 = calc_u_base_lower(
            z[j], k, N, H1, A0, X_p, Y_p)*np.exp(-L*k)/(2*A0**2)
        u_base_2 = calc_u_base_lower(
            z[j], k, N, H1, A0, X_n, Y_n)*np.exp(-L*k)/(2*A0**2)
        psi, u, w = loop(
            psi, u, w, k, j, psi_base_1, psi_base_2,
            u_base_1, u_base_2, x, t)

    print('Integrating middle sub-domain')
    for j in prange(zN, zN+zN_scaled):
        psi_base_1 = calc_psi_base_middle(
            z[j], k, N, H1, H2, A0,
            X_p, Y_p, beta_p, Da_2, Db_2)*np.exp(-L*k)/(2*A0**2)
        psi_base_2 = calc_psi_base_middle(
            z[j], k, N, H1, H2, A0,
            X_n, Y_n, beta_n, Da_2, Db_2)*np.exp(-L*k)/(2*A0**2)
        u_base_1 = calc_u_base_middle(
            z[j], k, N, H1, H2, A0,
            X_p, Y_p, beta_p, Da_2, Db_2)*np.exp(-L*k)/(2*A0**2)
        u_base_2 = calc_u_base_middle(
            z[j], k, N, H1, H2, A0,
            X_n, Y_n, beta_n, Da_2, Db_2)*np.exp(-L*k)/(2*A0**2)

        psi, u, w = loop(
            psi, u, w, k, j, psi_base_1, psi_base_2,
            u_base_1, u_base_2, x, t)

    #
    print('Integrating upper sub-domain.')
    # Perform numerical integration H1<z
    for j in prange(zN+zN_scaled, z.size):
        psi_base_1 = calc_psi_base_upper(
            z[j], k, N, H1, H2, A0, X_p, Y_p)*np.exp(-L*k)/(2*A0**2)
        psi_base_2 = calc_psi_base_upper(
            z[j], k, N, H1, H2, A0,
            X_n, Y_n, mode='neg')*np.exp(-L*k)/(2*A0**2)
        u_base_1 = calc_u_base_upper(
            z[j], k, N, H1, H2, A0, X_p, Y_p)*np.exp(-L*k)/(2*A0**2)
        u_base_2 = calc_u_base_upper(
            z[j], k, N, H1, H2, A0,
            X_n, Y_n, mode='neg')*np.exp(-L*k)/(2*A0**2)
        psi, u, w = loop(
            psi, u, w, k, j, psi_base_1, psi_base_2,
            u_base_1, u_base_2, x, t)

    psi = (1/np.pi)*np.real(psi)
    u = (1/np.pi)*np.real(u)
    w = -(1/np.pi)*np.real(w)

    return psi, u, w


def calc_coefficients(k, N, H1, H2, A0):
    G = (N-1)/(H2-H1)
    theta_1, A_1 = calc_theta_A(H1, k, N, H1, G, A0)
    dA_1 = calc_dA(H1, k, N, H1, G, A0)
    alpha_1 = calc_alpha(H1, k, N, H1, G, A0)

    theta_2, A_2 = calc_theta_A(H2, k, N, H1, G, A0)
    dA_2 = calc_dA(H2, k, N, H1, G, A0)
    alpha_2 = calc_alpha(H2, k, N, H1, G, A0)

    m = k/A0
    # gamma = (m*np.cos(m*H1)+np.sin(m*H1)*(dA_1/A_1+alpha_1))/(2*alpha_1)

    f_1 = calc_f(H1, k, N, H1, G, A0)
    f_2 = calc_f(H2, k, N, H1, G, A0)

    # pcfd = np.frompyfunc(mpmath.pcfd, 2, 1)
    Da_1 = pcfd(-1/2, (1+1j)*f_1).astype(complex)
    Da_2 = pcfd(-1/2, (1+1j)*f_2).astype(complex)
    Db_1 = pcfd(-1/2, (1-1j)*f_1).astype(complex)
    Db_2 = pcfd(-1/2, (1-1j)*f_2).astype(complex)

    beta_p = (1j*m*N+alpha_2+dA_2/A_2)/(2*alpha_2)

    X_p = 1/2*((1-beta_p)*(Da_1/Da_2)+beta_p*(Db_1/Db_2))
    Y_p = 1j/(2*m)*(
        (1-beta_p)*(Da_1/Da_2)*(-alpha_1+dA_1/A_1)
        + beta_p*(Db_1/Db_2)*(alpha_1+dA_1/A_1))

    beta_n = (-1j*m*N+alpha_2+dA_2/A_2)/(2*alpha_2)

    X_n = 1/2*((1-beta_n)*(Da_1/Da_2)+beta_n*(Db_1/Db_2))
    Y_n = 1j/(2*m)*(
        (1-beta_n)*(Da_1/Da_2)*(-alpha_1+dA_1/A_1)
        + beta_n*(Db_1/Db_2)*(alpha_1+dA_1/A_1))

    return X_p, Y_p, beta_p, X_n, Y_n, beta_n, Da_2, Db_2


@jit(parallel=True, nopython=True)
def loop(psi, u, w, k, j, psi_base_1, psi_base_2, u_base_1, u_base_2, x, t):

    for i in prange(x.size):
        for l in range(t.size):

            psi1_ig = psi_base_1*np.exp(1j*(k*x[i]+t[l]))
            psi2_ig = psi_base_2*np.exp(1j*(k*x[i]-t[l]))
            psi[0][l, j, i] = np.trapz(psi1_ig, k)
            psi[1][l, j, i] = np.trapz(psi2_ig, k)

            # Calc u
            u1_ig = u_base_1*np.exp(1j*(k*x[i]+t[l]))
            u2_ig = u_base_2*np.exp(1j*(k*x[i]-t[l]))
            u[0][l, j, i] = np.trapz(u1_ig, k)
            u[1][l, j, i] = np.trapz(u2_ig, k)

            # Calc w
            w[0][l, j, i] = np.trapz(psi1_ig*1j*k, k)
            w[1][l, j, i] = np.trapz(psi2_ig*1j*k, k)

    return psi, u, w


# @jit(nopython=True)
def calc_f(z, k, N, H1, G, A0):
    m = k/A0
    f = np.sqrt(m/G)*(1+G*(z-H1))
    return f


# @jit(nopython=False)
def calc_alpha(z, k, N, H1, G, A0):
    m = k/A0
    dtheta = calc_dtheta(z, k, N, H1, G, A0)
    alpha = 1j*m*(1+G*(z-H1)) - 1j*dtheta
    return alpha.astype(complex)


# @jit(nopython=False)
def calc_theta_A(z, k, N, H1, G, A0):
    f = calc_f(z, k, N, H1, G, A0)
    # besselj = np.frompyfunc(mpmath.besselj, 2, 1)
    re = (
        besselj(-1/4, f**2/2)*np.cos(f**2/2)
        - besselj(1/4, f**2/2)*np.cos(f**2/2+np.pi/4))
    im = (
        besselj(-1/4, f**2/2)*np.sin(f**2/2)
        - besselj(1/4, f**2/2)*np.sin(f**2/2+np.pi/4))
    theta = np.arctan((im/re).astype(float))
    A = np.sqrt(np.pi*f/2)*np.sqrt(re**2+im**2).astype(float)

    return theta, A


# @jit(nopython=False)
def calc_dtheta(z, k, N, H1, G, A0):
    m = k/A0
    f = calc_f(z, k, N, H1, G, A0)
    # besselj = np.frompyfunc(mpmath.besselj, 2, 1)

    num = f**2*besselj(-1/4, f**2/2)**2
    num = num - np.sqrt(2)*f**2*besselj(-1/4, f**2/2)*besselj(1/4, f**2/2)
    num = num + np.sqrt(2)/2*f**2*besselj(-1/4, f**2/2)*besselj(5/4, f**2/2)
    num = num + f**2*besselj(1/4, f**2/2)**2
    num = num - np.sqrt(2)/2*f**2*besselj(1/4, f**2/2)*besselj(3/4, f**2/2)
    num = num - np.sqrt(2)/2*besselj(-1/4, f**2/2)*besselj(1/4, f**2/2)

    den = besselj(-1/4, f**2/2)**2
    den = den - np.sqrt(2)*besselj(-1/4, f**2/2)*besselj(1/4, f**2/2)
    den = den + besselj(1/4, f**2/2)**2
    den = f*den

    dtheta = num/den
    dtheta = dtheta*np.sqrt(m*G)

    return dtheta.astype(float)


# @jit(nopython=False)
def calc_dA(z, k, N, H1, G, A0):
    m = k/A0
    f = calc_f(z, k, N, H1, G, A0)
    # besselj = np.frompyfunc(mpmath.besselj, 2, 1)

    num = -f**2*besselj(-1/4, f**2/2)*besselj(3/4, f**2/2)
    num = num + 1/2*np.sqrt(2)*f**2*besselj(-1/4, f**2/2)*besselj(5/4, f**2/2)
    num = num + 1/2*np.sqrt(2)*f**2*besselj(1/4, f**2/2)*besselj(3/4, f**2/2)
    num = num - f**2*besselj(1/4, f**2/2)*besselj(5/4, f**2/2)
    num = num - 1/2*np.sqrt(2)*besselj(-1/4, f**2/2)*besselj(1/4, f**2/2)
    num = num + besselj(1/4, f**2/2)**2
    num = np.sqrt(np.pi/2)*num

    den = besselj(-1/4, f**2/2)**2
    den = den - np.sqrt(2)*besselj(-1/4, f**2/2)*besselj(1/4, f**2/2)
    den = den + besselj(1/4, f**2/2)**2
    den = f*den

    dA = num/den**(1/2)
    dA = dA*np.sqrt(m*G)

    return dA.astype(float)


# @jit(nopython=True)
def calc_psi_base_lower(z, k, N, H1, A0, X, Y):
    m = k/A0

    P = (X-Y)*np.exp(-1j*m*H1)+(X+Y)*np.exp(1j*m*H1)
    B1 = (X-Y)*np.exp(-1j*m*H1)/P
    B2 = (X+Y)*np.exp(1j*m*H1)/P

    term_1 = -1/m*(-np.exp(-z)*(np.sin(m*z)+m*np.cos(m*z))+m)/(m**2+1)
    term_1 = term_1*(B1*np.exp(1j*m*z)+B2*np.exp(-1j*m*z))

    term_2 = (
        B1/(1j*m-1)*(np.exp((1j*m-1)*H1)-np.exp((1j*m-1)*z))
        + B2/(-1j*m-1)*(np.exp((-1j*m-1)*H1)-np.exp((-1j*m-1)*z)))
    term_2 = -1/m*np.sin(m*z)*term_2

    # term_3 = -1/m*1/g2*(-np.exp(-H1)/(1j*m*N-1))
    # term_3 = term_3*np.sin(m*z)

    return term_1+term_2


# @jit(nopython=True)
def calc_psi_base_middle(
        z, k, N, H1, H2, A0, X, Y, beta, Da_2, Db_2):
    m = k/A0
    G = (N-1)/(H2-H1)

    P = (X-Y)*np.exp(-1j*m*H1)+(X+Y)*np.exp(1j*m*H1)
    B1 = (1-beta)/P
    B2 = beta/P

    f = calc_f(z, k, N, H1, G, A0)

    # pcfd = np.frompyfunc(mpmath.pcfd, 2, 1)
    Da_z = pcfd(-1/2, (1+1j)*f).astype(complex)
    Db_z = pcfd(-1/2, (1-1j)*f).astype(complex)

    term_1 = -1/m*(-np.exp(-H1)*(np.sin(m*H1)+m*np.cos(m*H1))+m)/(m**2+1)
    term_1 = term_1*(B1*Da_z/Da_2+B2*Db_z/Db_2)

    # term_2 = (
    #     B1/(1j*m-1)*(np.exp((1j*m-1)*H1)-np.exp((1j*m-1)*z))
    #     + B2/(-1j*m-1)*(np.exp((-1j*m-1)*H1)-np.exp((-1j*m-1)*z)))
    # term_2 = -1/m*np.sin(m*z)*term_2

    # term_3 = -1/m*1/g2*(-np.exp(-H1)/(1j*m*N-1))
    # term_3 = term_3*np.sin(m*z)

    return term_1


# positive t mode, upper subdomain
# @jit(nopython=True)
def calc_psi_base_upper(z, k, N, H1, H2, A0, X, Y, mode='pos'):

    m = k/A0

    P = (X-Y)*np.exp(-1j*m*H1)+(X+Y)*np.exp(1j*m*H1)

    if mode == 'pos':
        term_1 = -1/(m*P)*np.exp(1j*m*N*(z-H2))
    else:
        term_1 = -1/(m*P)*np.exp(-1j*m*N*(z-H2))
    term_1 = term_1*(-np.exp(-H1)*(np.sin(m*H1)+m*np.cos(m*H1))+m)/(m**2+1)
    #
    # term_2_a = 1/(-1j*m*N-1)*(np.exp((-1j*m*N-1)*z)-np.exp((-1j*m*N-1)*H1))
    # term_2_b = - (g1/g2)*np.exp(-2*1j*m*N*H1)/(1j*m*N-1)
    # term_2_b = term_2_b*(np.exp((1j*m*N-1)*z)-np.exp((1j*m*N-1)*H1))
    # term_2 = (term_2_a+term_2_b)*np.exp(1j*m*N*z)/(2*1j*m*N)
    #
    # term_3 = 1/(2*1j*N)*(g1*np.exp(1j*m*N*(z-H1))-g2*np.exp(-1j*m*N*(z-H1)))
    # term_3 = -term_3/(m*g2)/(1j*m*N-1)*(-np.exp((1j*m*N-1)*z-1j*m*N*H1))

    return term_1 #term_2+term_3


# @jit(nopython=True)
def calc_u_base_lower(z, k, N, H1, A0, X, Y):
    m = k/A0

    P = (X-Y)*np.exp(-1j*m*H1)+(X+Y)*np.exp(1j*m*H1)
    B1 = (X-Y)*np.exp(-1j*m*H1)/P
    B2 = (X+Y)*np.exp(1j*m*H1)/P

    term_1_a = -1/m*np.sin(m*z)*np.exp(-z)
    term_1_a = term_1_a*(B1*np.exp(1j*m*z)+B2*np.exp(-1j*m*z))

    term_1_b = -1/m*(-np.exp(-z)*(np.sin(m*z)+m*np.cos(m*z))+m)/(m**2+1)
    term_1_b = term_1_b*((1j*m)*B1*np.exp(1j*m*z)+B2*(-1j*m)*np.exp(-1j*m*z))

    term_1 = term_1_a+term_1_b

    term_2_a = (
        B1*(-np.exp((1j*m-1)*z))
        + B2*(-np.exp((-1j*m-1)*z)))
    term_2_a = -1/m*np.sin(m*z)*term_2_a

    term_2_b = (
        B1/(1j*m-1)*(np.exp((1j*m-1)*H1)-np.exp((1j*m-1)*z))
        + B2/(-1j*m-1)*(np.exp((-1j*m-1)*H1)-np.exp((-1j*m-1)*z)))
    term_2_b = -np.cos(m*z)*term_2_b

    term_2 = term_2_a+term_2_b

    # term_2 = (
    #     B1/(1j*m-1)*(np.exp((1j*m-1)*H1)-np.exp((1j*m-1)*z))
    #     + B2/(-1j*m-1)*(np.exp((-1j*m-1)*H1)-np.exp((-1j*m-1)*z)))
    # term_2 = -1/m*np.sin(m*z)*term_2

    return term_1+term_2


# @jit(nopython=True)
def calc_u_base_middle(
        z, k, N, H1, H2, A0, X, Y, beta, Da_2, Db_2):
    m = k/A0
    G = (N-1)/(H2-H1)

    P = (X-Y)*np.exp(-1j*m*H1)+(X+Y)*np.exp(1j*m*H1)
    B1 = (1-beta)/P
    B2 = beta/P

    f = calc_f(z, k, N, H1, G, A0)

    # pcfd = np.frompyfunc(mpmath.pcfd, 2, 1)
    dDa_z = 1/2*(1+1j)*f*pcfd(-1/2, (1+1j)*f)
    dDa_z = dDa_z-pcfd(1/2, (1+1j)*f)
    dDa_z = dDa_z*(1+1j)*np.sqrt(m*G)
    dDa_z = dDa_z.astype(complex)

    dDb_z = 1/2*(1-1j)*f*pcfd(-1/2, (1-1j)*f)
    dDb_z = dDb_z-pcfd(1/2, (1-1j)*f)
    dDb_z = dDb_z*(1-1j)*np.sqrt(m*G)
    dDb_z = dDb_z.astype(complex)

    term_1 = -1/m*(-np.exp(-H1)*(np.sin(m*H1)+m*np.cos(m*H1))+m)/(m**2+1)
    term_1 = term_1*(B1*dDa_z/Da_2+B2*dDb_z/Db_2)

    # term_2 = (
    #     B1/(1j*m-1)*(np.exp((1j*m-1)*H1)-np.exp((1j*m-1)*z))
    #     + B2/(-1j*m-1)*(np.exp((-1j*m-1)*H1)-np.exp((-1j*m-1)*z)))
    # term_2 = -1/m*np.sin(m*z)*term_2

    # term_3 = -1/m*1/g2*(-np.exp(-H1)/(1j*m*N-1))
    # term_3 = term_3*np.sin(m*z)

    return term_1


# positive t mode, upper subdomain
# @jit(nopython=True)
def calc_u_base_upper(z, k, N, H1, H2, A0, X, Y, mode='pos'):

    m = k/A0

    P = (X-Y)*np.exp(-1j*m*H1)+(X+Y)*np.exp(1j*m*H1)

    if mode == 'pos':
        term_1 = -1j*N/P*np.exp(1j*m*N*(z-H2))
    else:
        term_1 = 1j*N/P*np.exp(-1j*m*N*(z-H2))
    term_1 = term_1*(-np.exp(-H1)*(np.sin(m*H1)+m*np.cos(m*H1))+m)/(m**2+1)
    #
    # term_2_a = 1/(-1j*m*N-1)*(np.exp((-1j*m*N-1)*z)-np.exp((-1j*m*N-1)*H1))
    # term_2_b = - (g1/g2)*np.exp(-2*1j*m*N*H1)/(1j*m*N-1)
    # term_2_b = term_2_b*(np.exp((1j*m*N-1)*z)-np.exp((1j*m*N-1)*H1))
    # term_2 = (term_2_a+term_2_b)*np.exp(1j*m*N*z)/(2*1j*m*N)
    #
    # term_3 = 1/(2*1j*N)*(g1*np.exp(1j*m*N*(z-H1))-g2*np.exp(-1j*m*N*(z-H1)))
    # term_3 = -term_3/(m*g2)/(1j*m*N-1)*(-np.exp((1j*m*N-1)*z-1j*m*N*H1))

    return term_1  # term_2+term_3


@jit(nopython=True)
def calc_theta_p(s, alpha=2):
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
