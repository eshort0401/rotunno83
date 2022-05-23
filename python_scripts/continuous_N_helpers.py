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

    psi = np.zeros((2, 3, t.size, z.size, x.size), dtype=np.complex64)
    u = np.zeros((2, 3, t.size, z.size, x.size), dtype=np.complex64)
    w = np.zeros((2, 3, t.size, z.size, x.size), dtype=np.complex64)

    # Define alternative domains
    theta_p = calc_theta_p(s, alpha=alpha)

    # Get wavenumbers greater than 2*pi
    k_3 = calc_k_3(theta_p, 1/(2*np.pi))

    # Get wavenumbers less than 2*pi/
    k_2 = calc_k_2(theta_p, 1/(2*np.pi))

    # Create wavenumber domain
    k = np.concatenate((k_2[-1::-1], np.array([2*np.pi]), k_3))

    print('Pre-calculating Coefficients')

    G = (N-1)/(H2-H1)
    theta_1, A_1 = calc_theta_A(H1, k, N, H1, G, A0)
    dA_1 = calc_dA(H1, k, N, H1, G, A0)
    alpha_1 = calc_alpha(H1, k, N, H1, G, A0)

    theta_2, A_2 = calc_theta_A(H2, k, N, H1, G, A0)
    dA_2 = calc_dA(H2, k, N, H1, G, A0)
    alpha_2 = calc_alpha(H2, k, N, H1, G, A0)

    m = k/A0

    f_1 = calc_f(H1, k, N, H1, G, A0)
    f_2 = calc_f(H2, k, N, H1, G, A0)

    Da_1 = np.array(pcfd(-1/2, (1+1j)*f_1)).astype(complex)
    Da_2 = np.array(pcfd(-1/2, (1+1j)*f_2)).astype(complex)
    Db_1 = np.array(pcfd(-1/2, (1-1j)*f_1)).astype(complex)
    Db_2 = np.array(pcfd(-1/2, (1-1j)*f_2)).astype(complex)

    gamma = (m*np.cos(m*H1)+np.sin(m*H1)*(-dA_1/A_1+alpha_1))/(2*alpha_1)

    beta_p = (1j*m*N+alpha_2-dA_2/A_2)/(2*alpha_2)

    X_p = 1/2*((1-beta_p)*(Da_1/Da_2)+beta_p*(Db_1/Db_2))
    Y_p = 1j/(2*m)*(
        (1-beta_p)*(Da_1/Da_2)*(-alpha_1+dA_1/A_1)
        + beta_p*(Db_1/Db_2)*(alpha_1+dA_1/A_1))

    mu_p = gamma*(1-beta_p)*Db_2*Da_1-(np.sin(m*H1)-gamma)*beta_p*Da_2*Db_1

    T = 1/2*((np.sin(m*H1)-gamma)*(Da_2/Da_1)+gamma*(Db_2/Db_1))
    S = 1/(2*1j*m*N)*(
        (np.sin(m*H1)-gamma)*(Da_2/Da_1)*(-alpha_2+dA_2/A_2)
        + gamma*(Db_2/Db_1)*(alpha_2+dA_2/A_2))

    beta_n = (-1j*m*N+alpha_2-dA_2/A_2)/(2*alpha_2)

    X_n = 1/2*((1-beta_n)*(Da_1/Da_2)+beta_n*(Db_1/Db_2))
    Y_n = 1j/(2*m)*(
        (1-beta_n)*(Da_1/Da_2)*(-alpha_1+dA_1/A_1)
        + beta_n*(Db_1/Db_2)*(alpha_1+dA_1/A_1))

    mu_n = gamma*(1-beta_n)*Db_2*Da_1-(np.sin(m*H1)-gamma)*beta_n*Da_2*Db_1

    print('Pre-calculating middle sub-domain forcing integrals.')
    Ia, Ib, low_a, low_b, high_a, high_b = calc_mid_forcing_integrands(
        H1, H2, k, N, L, z, zN, zN_scaled, A0)

    print('Beginning integration.')
    print('Integrating lower sub-domain.')

    # Calculate the B coefficient from middle subdomain
    B_p_low = -Da_2*Db_1*Da_1*beta_p/(2*mu_p)*high_a[0, :]
    B_p_low = B_p_low-Db_2*Db_1*Da_1*(1-beta_p)/(2*mu_p)*high_b[0, :]

    B_n_low = -Da_2*Db_1*Da_1*beta_n/(2*mu_n)*high_a[0, :]
    B_n_low = B_n_low-Db_2*Db_1*Da_1*(1-beta_n)/(2*mu_n)*high_b[0, :]

    # Perform numerical integration 0<z<=H1
    for j in range(0, zN):

        psi_base_1_lf, psi_base_1_mf, psi_base_1_hf = calc_psi_base_lower(
            z[j], k, N, H1, H2, A0, X_p, Y_p, S, T, B_p_low)
        psi_base_2_lf, psi_base_2_mf, psi_base_2_hf = calc_psi_base_lower(
            z[j], k, N, H1, H2, A0, X_n, Y_n, S, T, B_n_low, mode='neg')
        u_base_1_lf, u_base_1_mf, u_base_1_hf = calc_u_base_lower(
            z[j], k, N, H1, H2, A0, X_p, Y_p, S, T, B_p_low)
        u_base_2_lf, u_base_2_mf, u_base_2_hf = calc_u_base_lower(
            z[j], k, N, H1, H2, A0, X_n, Y_n, S, T,
            B_n_low, mode='neg')

        base_fns = [
            psi_base_1_lf, psi_base_1_mf, psi_base_1_hf,
            psi_base_2_lf, psi_base_2_mf, psi_base_2_hf,
            u_base_1_lf, u_base_1_mf, u_base_1_hf,
            u_base_2_lf, u_base_2_mf, u_base_2_hf]
        base_fns = [fn*np.exp(-L*k)/(2*A0**2) for fn in base_fns]

        psi, u, w = loop(
            psi, u, w, k, j,
            base_fns[0], base_fns[1], base_fns[2],
            base_fns[3], base_fns[4], base_fns[5],
            base_fns[6], base_fns[7], base_fns[8],
            base_fns[9], base_fns[10], base_fns[11],
            x, t)

    # import pdb; pdb.set_trace()

    print('Integrating middle sub-domain')
    for j in range(zN, zN+zN_scaled-1):

        D_p_j_a = -Da_1*Da_2*Db_2*gamma/(2*mu_p)*low_a[j-zN+1, :]
        D_p_j_b = -Db_1*Da_2*Db_2*(np.sin(m*H1)-gamma)/(2*mu_p)
        D_p_j_b = D_p_j_b*low_b[j-zN+1, :]
        D_p_j = D_p_j_a + D_p_j_b

        dD_p_j_a = -Da_1*Da_2*Db_2*gamma/(2*mu_p)*Ia[:, j-zN+1]
        dD_p_j_b = -Db_1*Da_2*Db_2*(np.sin(m*H1)-gamma)/(2*mu_p)
        dD_p_j_b = dD_p_j_b*Ib[:, j-zN+1]
        dD_p_j = dD_p_j_a + dD_p_j_b

        B_p_j = -Da_2*Db_1*Da_1*beta_p/(2*mu_p)*high_a[j-zN+1, :]
        B_p_j = B_p_j-Db_2*Db_1*Da_1*(1-beta_p)/(2*mu_p)*high_b[j-zN+1, :]

        dB_p_j = Da_2*Db_1*Da_1*beta_p/(2*mu_p)*Ia[:, j-zN+1]
        dB_p_j = dB_p_j+Db_2*Db_1*Da_1*(1-beta_p)/(2*mu_p)*Ib[:, j-zN+1]

        D_n_j_a = -Da_1*Da_2*Db_2*gamma/(2*mu_n)*low_a[j-zN+1, :]
        D_n_j_b = -Db_1*Da_2*Db_2*(np.sin(m*H1)-gamma)/(2*mu_n)
        D_n_j_b = D_n_j_b*low_b[j-zN+1, :]
        D_n_j = D_n_j_a + D_n_j_b

        dD_n_j_a = -Da_1*Da_2*Db_2*gamma/(2*mu_n)*Ia[:, j-zN+1]
        dD_n_j_b = -Db_1*Da_2*Db_2*(np.sin(m*H1)-gamma)/(2*mu_n)
        dD_n_j_b = dD_n_j_b*Ib[:, j-zN+1]
        dD_n_j = dD_n_j_a + dD_n_j_b

        B_n_j = -Da_2*Db_1*Da_1*beta_n/(2*mu_n)*high_a[j-zN+1, :]
        B_n_j = B_n_j-Db_2*Db_1*Da_1*(1-beta_n)/(2*mu_n)*high_b[j-zN+1, :]

        dB_n_j = Da_2*Db_1*Da_1*beta_n/(2*mu_n)*Ia[:, j-zN+1]
        dB_n_j = dB_n_j+Db_2*Db_1*Da_1*(1-beta_n)/(2*mu_n)*Ib[:, j-zN+1]

        psi_base_1_lf, psi_base_1_mf, psi_base_1_hf = calc_psi_base_middle(
            z[j], k, N, H1, H2, A0, X_p, Y_p, beta_p,
            gamma, B_p_j, D_p_j, S, T, Da_1, Db_1, Da_2, Db_2)
        psi_base_2_lf, psi_base_2_mf, psi_base_2_hf = calc_psi_base_middle(
            z[j], k, N, H1, H2, A0, X_n, Y_n, beta_n,
            gamma, B_n_j, D_n_j, S, T, Da_1, Db_1, Da_2, Db_2, mode='neg')
        u_base_1_lf, u_base_1_mf, u_base_1_hf = calc_u_base_middle(
            z[j], k, N, H1, H2, A0, X_p, Y_p, beta_p,
            gamma, B_p_j, dB_p_j, D_p_j, dD_p_j, S, T, Da_1, Db_1, Da_2, Db_2)
        u_base_2_lf, u_base_2_mf, u_base_2_hf = calc_u_base_middle(
            z[j], k, N, H1, H2, A0, X_n, Y_n, beta_n,
            gamma, B_n_j, dB_n_j, D_n_j, dD_n_j,
            S, T, Da_1, Db_1, Da_2, Db_2, mode='neg')

        base_fns = [
            psi_base_1_lf, psi_base_1_mf, psi_base_1_hf,
            psi_base_2_lf, psi_base_2_mf, psi_base_2_hf,
            u_base_1_lf, u_base_1_mf, u_base_1_hf,
            u_base_2_lf, u_base_2_mf, u_base_2_hf]
        base_fns = [fn*np.exp(-L*k)/(2*A0**2) for fn in base_fns]

        psi, u, w = loop(
            psi, u, w, k, j,
            base_fns[0], base_fns[1], base_fns[2],
            base_fns[3], base_fns[4], base_fns[5],
            base_fns[6], base_fns[7], base_fns[8],
            base_fns[9], base_fns[10], base_fns[11],
            x, t)

    print('Integrating upper sub-domain.')
    # Perform numerical integration H1<z

    D_p_high_a = -Da_1*Da_2*Db_2*gamma/(2*mu_p)*low_a[-1, :]
    D_p_high_b = -Db_1*Da_2*Db_2*(np.sin(m*H1)-gamma)/(2*mu_p)
    D_p_high_b = D_p_high_b*low_b[-1, :]
    D_p_high = D_p_high_a + D_p_high_b

    D_n_high_a = -Da_1*Da_2*Db_2*gamma/(2*mu_n)*low_a[-1, :]
    D_n_high_b = -Db_1*Da_2*Db_2*(np.sin(m*H1)-gamma)/(2*mu_n)
    D_n_high_b = D_n_high_b*low_b[-1, :]
    D_n_high = D_n_high_a + D_n_high_b

    for j in range(zN+zN_scaled-1, z.size):
        psi_base_1_lf, psi_base_1_mf, psi_base_1_hf = calc_psi_base_upper(
            z[j], k, N, H1, H2, A0, X_p, Y_p, S, T,
            beta_p, gamma, D_p_high)
        psi_base_2_lf, psi_base_2_mf, psi_base_2_hf = calc_psi_base_upper(
            z[j], k, N, H1, H2, A0, X_n, Y_n, S, T,
            beta_n, gamma, D_n_high, mode='neg')
        u_base_1_lf, u_base_1_mf, u_base_1_hf = calc_u_base_upper(
            z[j], k, N, H1, H2, A0, X_p, Y_p, S, T,
            beta_p, gamma, D_p_high)
        u_base_2_lf, u_base_2_mf, u_base_2_hf = calc_u_base_upper(
            z[j], k, N, H1, H2, A0, X_n, Y_n, S, T,
            beta_n, gamma, D_n_high, mode='neg')

        base_fns = [
            psi_base_1_lf, psi_base_1_mf, psi_base_1_hf,
            psi_base_2_lf, psi_base_2_mf, psi_base_2_hf,
            u_base_1_lf, u_base_1_mf, u_base_1_hf,
            u_base_2_lf, u_base_2_mf, u_base_2_hf]
        base_fns = [fn*np.exp(-L*k)/(2*A0**2) for fn in base_fns]

        psi, u, w = loop(
            psi, u, w, k, j,
            base_fns[0], base_fns[1], base_fns[2],
            base_fns[3], base_fns[4], base_fns[5],
            base_fns[6], base_fns[7], base_fns[8],
            base_fns[9], base_fns[10], base_fns[11],
            x, t)

    psi = (1/np.pi)*np.real(psi)
    u = (1/np.pi)*np.real(u)
    w = -(1/np.pi)*np.real(w)

    return psi, u, w


@jit(parallel=True, nopython=True)
def loop(
        psi, u, w, k, j,
        psi_base_1_lf, psi_base_1_mf, psi_base_1_hf,
        psi_base_2_lf, psi_base_2_mf, psi_base_2_hf,
        u_base_1_lf, u_base_1_mf, u_base_1_hf,
        u_base_2_lf, u_base_2_mf, u_base_2_hf, x, t):

    pos_fns = [
        psi_base_1_lf, psi_base_1_mf, psi_base_1_hf,
        u_base_1_lf, u_base_1_mf, u_base_1_hf]
    neg_fns = [
        psi_base_2_lf, psi_base_2_mf, psi_base_2_hf,
        u_base_2_lf, u_base_2_mf, u_base_2_hf]

    for i in range(x.size):
        for l in range(t.size):

            [
                psi1_ig_lf, psi1_ig_mf, psi1_ig_hf,
                u1_ig_lf, u1_ig_mf, u1_ig_hf] = [
                fn*np.exp(1j*(k*x[i]+t[l])) for fn in pos_fns]
            [
                psi2_ig_lf, psi2_ig_mf, psi2_ig_hf,
                u2_ig_lf, u2_ig_mf, u2_ig_hf] = [
                fn*np.exp(1j*(k*x[i]-t[l])) for fn in neg_fns]

            psi[0][0][l, j, i] = np.trapz(psi1_ig_lf, k)
            psi[0][1][l, j, i] = np.trapz(psi1_ig_mf, k)
            psi[0][2][l, j, i] = np.trapz(psi1_ig_hf, k)

            psi[1][0][l, j, i] = np.trapz(psi2_ig_lf, k)
            psi[1][1][l, j, i] = np.trapz(psi2_ig_mf, k)
            psi[1][2][l, j, i] = np.trapz(psi2_ig_hf, k)

            # Calc u
            u[0][0][l, j, i] = np.trapz(u1_ig_lf, k)
            u[0][1][l, j, i] = np.trapz(u1_ig_mf, k)
            u[0][2][l, j, i] = np.trapz(u1_ig_hf, k)

            u[1][0][l, j, i] = np.trapz(u2_ig_lf, k)
            u[1][1][l, j, i] = np.trapz(u2_ig_mf, k)
            u[1][2][l, j, i] = np.trapz(u2_ig_hf, k)

            # Calc w
            w[0][0][l, j, i] = np.trapz(psi1_ig_lf*1j*k, k)
            w[0][1][l, j, i] = np.trapz(psi1_ig_mf*1j*k, k)
            w[0][2][l, j, i] = np.trapz(psi1_ig_hf*1j*k, k)

            w[1][0][l, j, i] = np.trapz(psi2_ig_lf*1j*k, k)
            w[1][1][l, j, i] = np.trapz(psi2_ig_mf*1j*k, k)
            w[1][2][l, j, i] = np.trapz(psi2_ig_hf*1j*k, k)

    return psi, u, w


def calc_mid_forcing_integrands(
        H1, H2, k, N, L, z, zN, zN_scaled, A0, grain=3):

    G = (N-1)/(H2-H1)

    z_mid = z[zN-1:zN+zN_scaled]
    zp_mid_N = (len(z_mid)-1)*grain+1
    zp_mid = np.linspace(z_mid[0], z_mid[-1], zp_mid_N)

    Zp, K = np.meshgrid(zp_mid, k)

    f = calc_f(Zp, K, N, H1, G, A0)
    alpha = calc_alpha(Zp, K, N, H1, G, A0)

    Ia = np.array(np.exp(-Zp)/(pcfd(-1/2, (1+1j)*f)*alpha)).astype(complex)
    Ib = np.array(np.exp(-Zp)/(pcfd(-1/2, (1-1j)*f)*alpha)).astype(complex)

    low_a = np.zeros([len(z_mid), len(k)]).astype(complex)
    low_b = np.zeros([len(z_mid), len(k)]).astype(complex)
    high_a = np.zeros([len(z_mid), len(k)]).astype(complex)
    high_b = np.zeros([len(z_mid), len(k)]).astype(complex)

    for i in range(0, len(z_mid)):
        ind = i*grain
        low_a[i] = np.trapz(Ia[:, :ind+1], zp_mid[:ind+1], axis=1)
        low_b[i] = np.trapz(Ib[:, :ind+1], zp_mid[:ind+1], axis=1)
        high_a[i] = np.trapz(Ia[:, ind:], zp_mid[ind:], axis=1)
        high_b[i] = np.trapz(Ib[:, ind:], zp_mid[ind:], axis=1)

    return Ia[:, ::grain], Ib[:, ::grain], low_a, low_b, high_a, high_b


# @jit(nopython=True)
def calc_f(z, k, N, H1, G, A0):
    m = k/A0
    f = np.sqrt(m/np.abs(G))*(1+G*(z-H1))
    return f


# @jit(nopython=False)
def calc_alpha(z, k, N, H1, G, A0):
    m = k/A0
    dtheta = calc_dtheta(z, k, N, H1, G, A0)
    if N > 1:
        alpha = 1j*m*(1+G*(z-H1)) - 1j*dtheta
    else:
        alpha = -1j*m*(1+G*(z-H1)) - 1j*dtheta
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
    re = np.array(re).astype(float)
    im = np.array(im).astype(float)
    theta = np.arctan(im/re)
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
    if N > 1:
        dtheta = dtheta*np.sqrt(m*G)
    else:
        dtheta = -dtheta*np.sqrt(m*np.abs(G))

    return np.array(dtheta).astype(float)


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
    if N > 1:
        dA = dA*np.sqrt(m*G)
    else:
        dA = -dA*np.sqrt(m*np.abs(G))

    return np.array(dA).astype(float)


# @jit(nopython=True)
def calc_psi_base_lower(
        z, k, N, H1, H2, A0, X, Y, S, T, B, mode='pos'):
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

    term_3 = B*np.sin(m*z)
    if mode == 'pos':
        term_4 = 1/(2*1j*m*N*(T-S))*1/(1j*m*N-1)*(-np.exp(-H2))*np.sin(m*z)
    else:
        term_4 = -1/(2*1j*m*N*(T+S))*1/(-1j*m*N-1)
        term_4 = term_4*(-np.exp(-H2))*np.sin(m*z)

    return (term_1+term_2), term_3, term_4


# @jit(nopython=True)
def calc_psi_base_middle(
        z, k, N, H1, H2, A0, X, Y, beta, gamma, B, D, S, T,
        Da_1, Db_1, Da_2, Db_2, mode='pos'):
    m = k/A0
    G = (N-1)/(H2-H1)

    P = (X-Y)*np.exp(-1j*m*H1)+(X+Y)*np.exp(1j*m*H1)
    B1 = (1-beta)/P
    B2 = beta/P

    f = calc_f(z, k, N, H1, G, A0)

    # pcfd = np.frompyfunc(mpmath.pcfd, 2, 1)
    Da_z = np.array(pcfd(-1/2, (1+1j)*f)).astype(complex)
    Db_z = np.array(pcfd(-1/2, (1-1j)*f)).astype(complex)

    term_1 = -1/m*(-np.exp(-H1)*(np.sin(m*H1)+m*np.cos(m*H1))+m)/(m**2+1)
    term_1 = term_1*(B1*Da_z/Da_2+B2*Db_z/Db_2)

    term_2 = D*((1-beta)*Da_z/Da_2+beta*Db_z/Db_2)
    term_3 = B*((np.sin(m*H1)-gamma)*Da_z/Da_1+gamma*Db_z/Db_1)
    if mode == 'pos':
        term_4 = 1/(2*1j*m*N*(T-S))*1/(1j*m*N-1)*(-np.exp(-H2))
        term_4 = term_4*((np.sin(m*H1)-gamma)*Da_z/Da_1+gamma*Db_z/Db_1)
    else:
        term_4 = -1/(2*1j*m*N*(T+S))*1/(-1j*m*N-1)*(-np.exp(-H2))
        term_4 = term_4*((np.sin(m*H1)-gamma)*Da_z/Da_1+gamma*Db_z/Db_1)

    return term_1, (term_2+term_3), term_4


# positive t mode, upper subdomain
# @jit(nopython=True)
def calc_psi_base_upper(
        z, k, N, H1, H2, A0, X, Y, S, T, beta, gamma, D, mode='pos'):

    m = k/A0

    P = (X-Y)*np.exp(-1j*m*H1)+(X+Y)*np.exp(1j*m*H1)

    if mode == 'pos':
        term_1 = -1/(m*P)*np.exp(1j*m*N*(z-H2))
    else:
        term_1 = -1/(m*P)*np.exp(-1j*m*N*(z-H2))
    term_1 = term_1*(-np.exp(-H1)*(np.sin(m*H1)+m*np.cos(m*H1))+m)/(m**2+1)

    if mode == 'pos':
        term_2 = D*np.exp(1j*m*N*(z-H2))

        term_3_a = 1/(-1j*m*N-1)*(np.exp(-1j*m*N*(z-H2)-z)-np.exp(-H2))
        term_3_b = 1/(1j*m*N-1)*(np.exp(1j*m*N*(z-H2)-z)-np.exp(-H2))
        term_3 = term_3_a+(T+S)/(T-S)*term_3_b
        term_3 = term_3*np.exp(1j*m*N*(z-H2))/(2*1j*m*N)

        term_4 = ((T+S)*np.exp(1j*m*N*(z-H2))+(T-S)*np.exp(-1j*m*N*(z-H2)))
        term_4 = term_4*1/(2*1j*m*N*(T-S))*1/(1j*m*N-1)
        term_4 = term_4*(-np.exp(1j*m*N*(z-H2)-z))

    else:
        term_2 = D*np.exp(-1j*m*N*(z-H2))

        term_3_a = -1/(-1j*m*N-1)*(np.exp(-1j*m*N*(z-H2)-z)-np.exp(-H2))
        term_3_b = -1/(1j*m*N-1)*(np.exp(1j*m*N*(z-H2)-z)-np.exp(-H2))
        term_3 = (T-S)/(T+S)*term_3_a+term_3_b
        term_3 = term_3*np.exp(-1j*m*N*(z-H2))/(2*1j*m*N)

        term_4 = ((T+S)*np.exp(1j*m*N*(z-H2))+(T-S)*np.exp(-1j*m*N*(z-H2)))
        term_4 = -term_4*1/(2*1j*m*N*(T+S))*1/(-1j*m*N-1)
        term_4 = term_4*(-np.exp(-1j*m*N*(z-H2)-z))

    return term_1, term_2, (term_3+term_4)


# @jit(nopython=True)
def calc_u_base_lower(z, k, N, H1, H2, A0, X, Y, S, T, B, mode='pos'):
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

    term_3 = B*m*np.cos(m*z)
    if mode == 'pos':
        term_4 = 1/(2*1j*N*(T-S))*1/(1j*m*N-1)*(-np.exp(-H2))*np.cos(m*z)
    else:
        term_4 = -1/(2*1j*N*(T+S))*1/(-1j*m*N-1)
        term_4 = term_4*(-np.exp(-H2))*np.cos(m*z)

    return (term_1+term_2), term_3, term_4


# @jit(nopython=True)
def calc_u_base_middle(
        z, k, N, H1, H2, A0, X, Y, beta, gamma, B, dB, D, dD, S, T,
        Da_1, Db_1, Da_2, Db_2, mode='pos'):

    m = k/A0
    G = (N-1)/(H2-H1)

    P = (X-Y)*np.exp(-1j*m*H1)+(X+Y)*np.exp(1j*m*H1)
    B1 = (1-beta)/P
    B2 = beta/P

    f = calc_f(z, k, N, H1, G, A0)

    Da_z = np.array(pcfd(-1/2, (1+1j)*f)).astype(complex)
    Db_z = np.array(pcfd(-1/2, (1-1j)*f)).astype(complex)

    dDa_z = 1j*f*Da_z
    dDa_z = dDa_z-(1+1j)*np.array(pcfd(1/2, (1+1j)*f)).astype(complex)
    if N > 1:
        dDa_z = dDa_z*np.sqrt(m*G)
    else:
        dDa_z = -dDa_z*np.sqrt(m*np.abs(G))
    dDa_z = dDa_z.astype(complex)

    dDb_z = -1j*f*Db_z
    dDb_z = dDb_z-(1-1j)*np.array(pcfd(1/2, (1-1j)*f)).astype(complex)
    if N > 1:
        dDb_z = dDb_z*np.sqrt(m*G)
    else:
        dDb_z = -dDb_z*np.sqrt(m*np.abs(G))
    dDb_z = dDb_z.astype(complex)

    term_1 = -1/m*(-np.exp(-H1)*(np.sin(m*H1)+m*np.cos(m*H1))+m)/(m**2+1)
    term_1 = term_1*(B1*dDa_z/Da_2+B2*dDb_z/Db_2)

    term_2 = D*((1-beta)*dDa_z/Da_2+beta*dDb_z/Db_2)
    term_2 = term_2 + dD*((1-beta)*Da_z/Da_2+beta*Db_z/Db_2)
    term_3 = B*((np.sin(m*H1)-gamma)*dDa_z/Da_1+gamma*dDb_z/Db_1)
    term_3 = term_3 + dB*((np.sin(m*H1)-gamma)*Da_z/Da_1+gamma*Db_z/Db_1)
    if mode == 'pos':
        term_4 = 1/(2*1j*m*N*(T-S))*1/(1j*m*N-1)*(-np.exp(-H2))
        term_4 = term_4*((np.sin(m*H1)-gamma)*dDa_z/Da_1+gamma*dDb_z/Db_1)
    else:
        term_4 = -1/(2*1j*m*N*(T+S))*1/(-1j*m*N-1)*(-np.exp(-H2))
        term_4 = term_4*((np.sin(m*H1)-gamma)*dDa_z/Da_1+gamma*dDb_z/Db_1)

    # m = k/A0
    # G = (N-1)/(H2-H1)
    #
    # P = (X-Y)*np.exp(-1j*m*H1)+(X+Y)*np.exp(1j*m*H1)
    # B1 = (1-beta)/P
    # B2 = beta/P
    #
    # f = calc_f(z, k, N, H1, G, A0)
    #
    # # pcfd = np.frompyfunc(mpmath.pcfd, 2, 1)
    # Da_z = pcfd(-1/2, (1+1j)*f).astype(complex)
    # Db_z = pcfd(-1/2, (1-1j)*f).astype(complex)
    #
    # term_1 = -1/m*(-np.exp(-H1)*(np.sin(m*H1)+m*np.cos(m*H1))+m)/(m**2+1)
    # term_1 = term_1*(B1*Da_z/Da_2+B2*Db_z/Db_2)
    #
    # term_2 = D*((1-beta)*Da_z/Da_2+beta*Db_z/Db_2)
    # term_3 = B*((np.sin(m*H1)-gamma)*Da_z/Da_1+gamma*Db_z/Db_1)
    # if mode == 'pos':
    #     term_4 = 1/(2*1j*m*N*(T-S))*1/(1j*m*N-1)*(-np.exp(-H2))
    #     term_4 = term_4*((np.sin(m*H1)-gamma)*Da_z/Da_1+gamma*Db_z/Db_1)
    # else:
    #     term_4 = -1/(2*1j*m*N*(T+S))*1/(-1j*m*N-1)*(-np.exp(-H2))
    #     term_4 = term_4*((np.sin(m*H1)-gamma)*Da_z/Da_1+gamma*Db_z/Db_1)

    return term_1, (term_2+term_3), term_4


# positive t mode, upper subdomain
# @jit(nopython=True)
def calc_u_base_upper(
        z, k, N, H1, H2, A0, X, Y, S, T, beta, gamma, D, mode='pos'):

    m = k/A0

    P = (X-Y)*np.exp(-1j*m*H1)+(X+Y)*np.exp(1j*m*H1)

    if mode == 'pos':
        term_1 = -1j*N/P*np.exp(1j*m*N*(z-H2))
    else:
        term_1 = 1j*N/P*np.exp(-1j*m*N*(z-H2))
    term_1 = term_1*(-np.exp(-H1)*(np.sin(m*H1)+m*np.cos(m*H1))+m)/(m**2+1)

    if mode == 'pos':
        term_2 = D*1j*m*N*np.exp(1j*m*N*(z-H2))

        term_3_a = np.exp(-1j*m*N*(z-H2)-z)
        term_3_b = np.exp(1j*m*N*(z-H2)-z)
        term_3 = term_3_a+(T+S)/(T-S)*term_3_b
        term_3_1 = term_3*np.exp(1j*m*N*(z-H2))/(2*1j*m*N)

        term_3_a = 1/(-1j*m*N-1)*(np.exp(-1j*m*N*(z-H2)-z)-np.exp(-H2))
        term_3_b = 1/(1j*m*N-1)*(np.exp(1j*m*N*(z-H2)-z)-np.exp(-H2))
        term_3 = term_3_a+(T+S)/(T-S)*term_3_b
        term_3_2 = term_3*np.exp(1j*m*N*(z-H2))/2

        term_3 = term_3_1 + term_3_2

        term_4 = ((T+S)*np.exp(1j*m*N*(z-H2))-(T-S)*np.exp(-1j*m*N*(z-H2)))
        term_4 = term_4*1/(2*(T-S))*1/(1j*m*N-1)
        term_4_1 = term_4*(-np.exp(1j*m*N*(z-H2)-z))

        term_4 = ((T+S)*np.exp(1j*m*N*(z-H2))+(T-S)*np.exp(-1j*m*N*(z-H2)))
        term_4 = term_4*1/(2*1j*m*N*(T-S))
        term_4_2 = term_4*(-np.exp(1j*m*N*(z-H2)-z))

        term_4 = term_4_1 + term_4_2

    else:
        term_2 = -D*1j*m*N*np.exp(-1j*m*N*(z-H2))

        term_3_a = -np.exp(-1j*m*N*(z-H2)-z)
        term_3_b = -np.exp(1j*m*N*(z-H2)-z)
        term_3 = (T-S)/(T+S)*term_3_a+term_3_b
        term_3_1 = term_3*np.exp(-1j*m*N*(z-H2))/(2*1j*m*N)

        term_3_a = -1/(-1j*m*N-1)*(np.exp(-1j*m*N*(z-H2)-z)-np.exp(-H2))
        term_3_b = -1/(1j*m*N-1)*(np.exp(1j*m*N*(z-H2)-z)-np.exp(-H2))
        term_3 = (T-S)/(T+S)*term_3_a+term_3_b
        term_3_2 = -term_3*np.exp(-1j*m*N*(z-H2))/2

        term_3 = term_3_1 + term_3_2

        term_4 = ((T+S)*np.exp(1j*m*N*(z-H2))-(T-S)*np.exp(-1j*m*N*(z-H2)))
        term_4 = -term_4*1/(2*(T+S))*1/(-1j*m*N-1)
        term_4_1 = term_4*(-np.exp(-1j*m*N*(z-H2)-z))

        term_4 = ((T+S)*np.exp(1j*m*N*(z-H2))+(T-S)*np.exp(-1j*m*N*(z-H2)))
        term_4 = -term_4*1/(2*1j*m*N*(T+S))
        term_4_2 = term_4*(-np.exp(-1j*m*N*(z-H2)-z))

        term_4 = term_4_1 + term_4_2

    return term_1, term_2, (term_3+term_4)  # term_2+term_3


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
