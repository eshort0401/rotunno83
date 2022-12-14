# Utilities
# from tqdm import tqdm

# Performance
from numba import jit

# Analysis
import numpy as np
import mpmath

erfi = np.frompyfunc(mpmath.erfi, 1, 1)


# @jit(parallel=True, nopython=False)
def integrate_convective(
        x, z, t, s, alpha, L, D, A):

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
    print('Integrating')

    # Perform numerical integration 0<z<=H1
    for j in range(0, z.size):

        print('{} of {}'.format(j+1, z.size))

        psi_base_1 = calc_psi_base(
            z[j], k, L, D, A)
        psi_base_2 = calc_psi_base(
            z[j], -k, L, D, A)
        u_base_1 = calc_u_base(
            z[j], k, L, D, A)
        u_base_2 = calc_u_base(
            z[j], -k, L, D, A)
        base_fns = [
            psi_base_1, psi_base_2, u_base_1, u_base_2]
        base_fns = [
            fn*1j*np.sqrt(np.pi)*L*k*np.exp(-L**2*k**2/4)/(2*A**2)
            for fn in base_fns]

        psi, u, w = loop(
            psi, u, w, k, j,
            base_fns[0], base_fns[1], base_fns[2], base_fns[3], x, t)

    psi = (1/np.pi)*np.real(psi)
    u = (1/np.pi)*np.real(u)
    w = -(1/np.pi)*np.real(w)

    return psi, u, w


@jit(nopython=True)
def loop(
        psi, u, w, k, j,
        psi_base_1, psi_base_2, u_base_1, u_base_2, x, t):

    pos_fns = [psi_base_1, u_base_1]
    neg_fns = [psi_base_2, u_base_2]

    for i in range(x.size):
        for l in range(t.size):

            [psi1_ig, u1_ig] = [
                fn*np.exp(1j*(k*x[i]+t[l])) for fn in pos_fns]
            [psi2_ig, u2_ig] = [
                fn*np.exp(1j*(k*x[i]-t[l])) for fn in neg_fns]

            # calc psi
            psi[0][l, j, i] = np.trapz(psi1_ig, k)
            psi[1][l, j, i] = np.trapz(psi2_ig, k)

            # calc u
            u[0][l, j, i] = np.trapz(u1_ig, k)
            u[1][l, j, i] = np.trapz(u2_ig, k)

            # Calc w
            w[0][l, j, i] = np.trapz(psi1_ig*1j*k, k)
            w[1][l, j, i] = np.trapz(psi2_ig*1j*k, k)

    return psi, u, w


# @jit(nopython=True)
def calc_psi_base(z, k, L, D, A):
    m = k/A

    E1inf = .5*np.sqrt(np.pi)*D*np.exp(-1/4*m*(D**2*m-4*1j))

    integral = -1/m*np.exp(1j*m*z)*(E2(z, D, m)-E2(0, D, m))
    integral = integral-1/m*np.sin(m*z)*(E1inf-E1(z, D, m))

    return integral
    # return term_3


# @jit(nopython=True)
def calc_u_base(z, k, L, D, A):

    m = k/A

    E1inf = .5*np.sqrt(np.pi)*D*np.exp(-1/4*m*(D**2*m-4*1j))

    integral = -1/m*1j*m*np.exp(1j*m*z)*(E2(z, D, m)-E2(0, D, m))
    integral = integral-1/m*np.exp(1j*m*z)*np.sin(m*z)*np.exp(-(z-1)**2/D**2)
    integral = integral-np.cos(m*z)*(E1inf-E1(z, D, m))
    integral = integral+1/m*np.sin(m*z)*np.exp(1j*m*z)*np.exp(-(z-1)**2/D**2)

    return integral


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


def E1(z, D, m):
    E = -.5*1j*np.sqrt(np.pi)*D*np.exp(-1/4*m*(D**2*m-4*1j))
    E = E*erfi(D*m/2+1j*(z-1)/D)
    return E.astype(complex)


def E2(z, D, m):
    E = -.25*np.sqrt(np.pi)*D*np.exp(-1/4*m*(m*D**2+4*1j))
    E = E*(erfi(D*m/2-1j*(z-1)/D)+np.exp(2*1j*m)*erfi(D*m/2+1j*(z-1)/D))
    return E.astype(complex)
