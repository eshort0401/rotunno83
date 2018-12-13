program rotunnoCaseTwo

    use nc_tools
    use math
    implicit none

    integer :: narg, argInd
    real :: latitude, xi0, h, delTheta, theta0, theta1, tropHeight

    integer(kind=4) :: ncid, status

    ! Third dimension (time) size should be a multiple of 4.
    complex :: psi(81,41,32), u(81,41,32), v(81,41,32), w(81,41,32)
    complex :: b(81,41,32) ! Non dimensional bouyancy
    real :: xi(81), zeta(41), tau(32), k(2001)
    real :: beta, Atilde, N

    xi0 = 0.2
    latitude = 5.0
    h = 1375.0
    delTheta = 8
    theta0 = 300.0
    theta1 = 360.0
    tropHeight = 11000.0

    narg = command_argument_count()

    ! Read in command line arguments if present
    if (narg >= 1) read_cl_real_arg(1, xi0)
    if (narg >= 2) read_cl_real_arg(2, latitude)
    if (narg >= 3) read_cl_real_arg(3, h)
    if (narg >= 4) read_cl_real_arg(4, delTheta)
    if (narg >= 5) read_cl_real_arg(5, theta0)
    if (narg >= 6) read_cl_real_arg(6, theta1)
    if (narg >= 7) read_cl_real_arg(7, tropHeight)

    if (narg >= 8) then
        print *, 'Error: rotunnoCaseTwo only accepts between 0 and 7 arguments!'
        stop
    endif

    ! Constants
    real, parameter :: pi  = 4 * atan(1.0_8)
    real, parameter :: omega = 7.2921159 * (10 ** (-5))
    real, parameter :: g = 9.80665

    ! Calculate nondimensional model parameters from inputs
    N = sqrt((g/theta0) * (theta1-theta0)/tropHeight) ! Brunt Vaisala Frequency
    beta = omega**2/((omega**2-f**2)**(1/2)*N)
    Atilde = .5*delTheta*(g/(pi*300))*h**(-1)*omega**(-3)/(12*60*60)

    !$OMP PARALLEL

    call solveCaseTwo(&
        -2.0, 2.0, size(xi), &! xiMin, xiMax, xiN
        0.0, 4.0, size(zeta), &! zetaMin, zetaMax, zetaN
        0.0, 20.0, size(k), &! k
        size(tau), beta, Atilde, xi0, latitude, &! tauN, beta, Atilde, xi0, lat
        psi, u, v, w, b, xi, zeta, tau &! Outputs
    )

    !$OMP END PARALLEL

    ncid = export_nc_3d( &
        './rotunnoCaseTwo.nc',real(psi), 'psi', &
        xi, zeta, tau, &
        int(size(xi), 4), int(size(zeta), 4), int(size(tau), 4), &
        'xi','zeta','tau', &
        'dimensionless','dimensionless','dimensionless' &
    )

    ncid = add_var_nc_3d( &
        './rotunnoCaseTwo.nc',real(u), 'u', &
        int(size(xi), 4), int(size(zeta), 4), int(size(tau), 4), &
        'xi','zeta','tau' &
    )

    ncid = add_var_nc_3d( &
        './rotunnoCaseTwo.nc',real(v), 'v', &
        int(size(xi), 4), int(size(zeta), 4), int(size(tau), 4), &
        'xi','zeta','tau' &
    )

    ncid = add_var_nc_3d( &
        './rotunnoCaseTwo.nc',real(w), 'w', &
        int(size(xi), 4), int(size(zeta), 4), int(size(tau), 4), &
        'xi','zeta','tau' &
    )

    ncid = add_global_attr_nc('./rotunnoCaseTwo.nc', "beta", beta)
    ncid = add_global_attr_nc('./rotunnoCaseTwo.nc', "Atilde", Atilde)
    ncid = add_global_attr_nc('./rotunnoCaseTwo.nc', "xi0", xi0)
    ncid = add_global_attr_nc('./rotunnoCaseTwo.nc', "lat", latitude)
    ncid = add_global_attr_nc('./rotunnoCaseTwo.nc', "N", N)
    ncid = add_global_attr_nc('./rotunnoCaseTwo.nc', "theta0", theta0)
    ncid = add_global_attr_nc('./rotunnoCaseTwo.nc', "theta1", theta1)
    ncid = add_global_attr_nc('./rotunnoCaseTwo.nc', "tropHeight", tropHeight)
    ncid = add_global_attr_nc('./rotunnoCaseTwo.nc', "delTheta", delTheta)
    ncid = add_global_attr_nc('./rotunnoCaseTwo.nc', "h", h)

contains
    subroutine solveCaseTwo( &
        xiMin, xiMax, xiN, &! xi
        zetaMin, zetaMax, zetaN, &! zeta
        kMin, kMax, kN, &! k (wavenumber)
        tauN, beta, Atilde, xi0, latitude, &! other parameters
        psi, u, v, w, b, xi, zeta, tau &! outputs
     )

    ! Inputs
    real, intent(in) :: xiMin, xiMax, zetaMin, zetaMax, kMin, kMax
    integer, intent(in) :: xiN, zetaN, kN
    integer, intent(in) :: tauN ! Should be a multiple of 4
    real, intent(in) :: beta, Atilde, xi0, latitude

    ! Subroutine variables
    integer :: i, j, l, m, v0_ind, ind_i, ind_im1
    real :: dxi, dzeta, dtau
    complex :: int_fun_k(kN)
    real :: k(kN)

    ! Constants
    real, parameter :: pi  = 4 * atan(1.0_8)
    real, parameter :: N = 0.005
    real, parameter :: omega = 7.2921159 * (10 ** (-5))

    ! Outputs
    complex, intent(out) :: psi(xiN, zetaN, tauN)
    complex, intent(out) :: b(xiN, zetaN, tauN)
    complex, intent(out) :: u(xiN, zetaN, tauN)
    complex, intent(out) :: v(xiN, zetaN, tauN)
    complex, intent(out) :: w(xiN, zetaN, tauN)
    real, intent(out) :: xi(xiN), zeta(zetaN), tau(tauN)

    ! Initialise arrays
    xi=createArray(xiMin, xiMax, xiN)
    zeta=createArray(zetaMin, zetaMax, zetaN)
    k=createArray(kMin, kMax, kN)

    dtau = 2.0 * pi / (tauN - 1)
    tau=createArray(0.0, 2.0*pi - dtau, tauN)

    ! Perform numerical integration
    do i=1, xiN, 1
        do j=1, zetaN, 1
            do l=1, tauN, 1
                do m=1, kN, 1
                    int_fun_k(m) = fun(xi(i), zeta(j), tau(l), k(m), xi0)
                enddo
                psi(i,j,l) = trapz(cmplx(k), int_fun_k)
            enddo
        enddo
    enddo

    ! Scale
    psi = -beta * Atilde * psi

    ! Calculate psi partials using centred finite differencing, and
    ! forward/backward differencing at endpoints.
    dxi = (xiMax - xiMin) / (xiN - 1)
    dzeta = (zetaMax - zetaMin) / (zetaN - 1)

    ! w = -dpsi_dxi
    w(2:xiN-1,:,:) = (psi(3:,:,:) - psi(1:xiN-2,:,:)) / (2 * dxi)
    w(1,:,:) = (psi(2,:,:) - psi(1,:,:)) / dxi
    w(xiN,:,:) = (psi(xiN,:,:) - psi(xiN-1,:,:)) / dxi
    w = -w

    ! u = dpsi_dzeta
    u(:,2:zetaN-1,:) = (psi(:,3:,:) - psi(:,1:zetaN-2,:)) / (2 * dzeta)
    u(:,1,:) = (psi(:,2,:) - psi(:,1,:)) / dzeta
    u(:,zetaN,:) = (psi(:,zetaN,:) - psi(:,zetaN-1,:)) / dzeta

    ! Calculate v from dv_dtau * omega + f * u = 0
    ! v is a perturbation, so it should integrate to zero.
    ! This can be achieved by assuming v=0 when abs(u) is maximised.
    ! abs(u) is maximised/minimised at tau = pi / 2.

    v0_ind = floor(tauN / 4.0) + 1
    v(:,:,v0_ind) = 0

    do i = 1, tauN-1, 1

        ind_i = mod((v0_ind + i - 1), tauN) + 1
        ind_im1 = mod((v0_ind + (i - 1) - 1), tauN) + 1

        v(:,:,ind_i) = ( &
            v(:,:,ind_im1) - &
            (u(:,:,ind_i) + u(:,:,ind_im1)) * dtau * sin(latitude * pi / 180) &
        )
    enddo

    ! Calculate b(tilde) by numerically integrating non dimensional form of
    ! equation (4) in Rotunno (1983).
    ! Note error in Rotunno (1983): b = bTilde * h * omega ^ 2, not omega ^ 3
    ! This gives dbTilde_dtau = Qtilde - (N/omega) ^ 2 - wTilde
    ! (N/omega)^2 is like a Berger number with H = L.

    ! b(:,:,1) = - ((N/omega) ** 2)
    ! tesT

end subroutine solveCaseTwo

! Term in integrand of psi
function fun(xi,zeta,tau,k,xi0) result(f)

    real, intent(in) :: xi, zeta, tau, k, xi0
    complex :: f
    ! Note adding cmplx conversion didn't solve gfortran bug!
    f = cos(cmplx(k) * cmplx(xi)) * exp(-cmplx(xi0) * cmplx(k)) * &
        (sin(cmplx(k) * cmplx(zeta)+cmplx(tau)) - &
        exp(-cmplx(zeta))*sin(cmplx(tau)))/(1+cmplx(k**2))

end function fun

end program rotunnoCaseTwo
