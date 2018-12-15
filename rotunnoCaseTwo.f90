program rotunnoCaseTwo

    use nc_tools
    use math
    implicit none

    integer :: narg
    real :: latitude, xi0, h, delTheta, theta0, theta1, tropHeight

    integer(kind=4) :: ncid

    ! Third dimension (time) size should be a multiple of 4.
    complex :: psi(81,41,32), u(81,41,32), w(81,41,32)
    real :: xi(81), zeta(41), tau(32), k(2001)
    real :: beta, Atilde, N, f

    ! Constants
    real, parameter :: pi  = 4 * atan(1.0_16)
    real, parameter :: omega = 7.2921159 * 10.0 ** (-5)
    real, parameter :: g = 9.80665

    xi0 = 0.2
    latitude = 6.0 ! Degrees
    h = 1500.0
    delTheta = 6.0
    theta0 = 300.0
    theta1 = 360.0
    tropHeight = 12000.0

    narg = command_argument_count()

    ! Read in command line arguments if present
    if (narg >= 1) call read_cl_real_arg(1, xi0)
    if (narg >= 2) call read_cl_real_arg(2, latitude)
    if (narg >= 3) call read_cl_real_arg(3, h)
    if (narg >= 4) call read_cl_real_arg(4, delTheta)
    if (narg >= 5) call read_cl_real_arg(5, theta0)
    if (narg >= 6) call read_cl_real_arg(6, theta1)
    if (narg >= 7) call read_cl_real_arg(7, tropHeight)

    if (narg >= 8) then
        print *, 'Error: rotunnoCaseTwo only accepts between 0 and 7 arguments!'
        stop
    endif

    ! Calculate nondimensional model parameters from inputs
    f = 2.0*omega*sin(latitude * pi / 180.0)
    N = sqrt((g/theta0) * (theta1-theta0)/tropHeight) ! Brunt Vaisala Frequency
    beta = omega**2/(N*sqrt(omega**2-f**2))
    Atilde = .5*delTheta*(g/(pi*theta0))*h**(-1)*omega**(-3)/(12*60*60)

    !$OMP PARALLEL

    call solveCaseTwo(&
        -2.0, 2.0, size(xi), &! xiMin, xiMax, xiN
        0.0, 4.0, size(zeta), &! zetaMin, zetaMax, zetaN
        0.0, 20.0, size(k), &! k
        size(tau), beta, Atilde, xi0, &! tauN, beta, Atilde, xi0, lat
        psi, u, w, xi, zeta, tau &! Outputs
    )

    !$OMP END PARALLEL

    ncid = export_nc_3d( &
        './rotunnoCaseTwo.nc',real(psi), 'psi', &
        xi, zeta, tau, &
        int(size(xi), 4), int(size(zeta), 4), int(size(tau), 4), &
        'xi','zeta','tau', &
        '-','-','-' &
    )

    ncid = add_var_nc_3d( &
        './rotunnoCaseTwo.nc',real(u), 'u', &
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
        tauN, beta, Atilde, xi0, &! other parameters
        psi, u, w, xi, zeta, tau &! outputs
     )

    ! Inputs
    real, intent(in) :: xiMin, xiMax, zetaMin, zetaMax, kMin, kMax
    integer, intent(in) :: xiN, zetaN, kN
    integer, intent(in) :: tauN ! Should be a multiple of 4
    real, intent(in) :: beta, Atilde, xi0

    ! Subroutine variables
    integer :: i, j, l, m
    real :: dxi, dzeta, dtau
    complex :: int_fun_k(kN)
    real :: k(kN)

    ! Constants
    real, parameter :: pi  = 4 * atan(1.0_8)
    real, parameter :: omega = 7.2921159 * (10.0 ** (-5))

    ! Outputs
    complex, intent(out) :: psi(xiN, zetaN, tauN)
    complex, intent(out) :: u(xiN, zetaN, tauN)
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
