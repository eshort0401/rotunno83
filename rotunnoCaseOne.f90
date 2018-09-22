program rotunnoCaseOne

    use nc_tools
    use math
    implicit none

    integer(kind=4) :: ncid

    ! Third dimension (time) size should be a multiple of 4.
    complex :: psi(81,41,32), u(81,41,32), v(81,41,32), w(81,41,32)
    real :: xi(81), zeta(41), tau(32)

    !$OMP PARALLEL

    call solveCaseOne(&
        -4.0, 4.0, size(xi), &! xiMin, xiMax, xiN
        0.0, 4.0, size(zeta), &! zeta
        -6.0, 6.0, 481, &! xip
        0.0, 6.0, 241, &! zetap
        size(tau), 7.27*10.0**(-3), 10.0**3, 0.2, 80.0, &! tauN, beta, Atilde, xi0
        psi, u, v, w, xi, zeta, tau &
    )

    !$OMP END PARALLEL

    ncid = export_nc_3d( &
        './rotunnoCaseOne.nc',real(psi), 'psi', &
        xi, zeta, tau, &
        int(size(xi), 4), int(size(zeta), 4), int(size(tau), 4), &
        'xi','zeta','tau', &
        'dimensionless','dimensionless','dimensionless' &
    )

    ncid = add_var_nc_3d( &
        './rotunnoCaseOne.nc',real(u), 'u', &
        int(size(xi), 4), int(size(zeta), 4), int(size(tau), 4), &
        'xi','zeta','tau' &
    )

    ncid = add_var_nc_3d( &
        './rotunnoCaseOne.nc',real(v), 'v', &
        int(size(xi), 4), int(size(zeta), 4), int(size(tau), 4), &
        'xi','zeta','tau' &
    )

    ncid = add_var_nc_3d( &
        './rotunnoCaseOne.nc',real(w), 'w', &
        int(size(xi), 4), int(size(zeta), 4), int(size(tau), 4), &
        'xi','zeta','tau' &
    )

contains
    subroutine solveCaseOne( &
        xiMin, xiMax, xiN, &! xi
        zetaMin, zetaMax, zetaN, &! zeta
        xipMin, xipMax, xipN, &! xip
        zetapMin, zetapMax, zetapN, &! zetap
        tauN, beta, Atilde, xi0, latitude, &! other parameters
        psi, u, v, w, xi, zeta, tau &! outputs
     )

    ! Inputs
    real, intent(in) :: xiMin, xiMax, zetaMin, zetaMax
    real, intent(in) :: xipMin, xipMax, zetapMin, zetapMax
    integer, intent(in) :: xiN, zetaN, xipN, zetapN
    integer, intent(in) :: tauN ! Should be a multiple of 4
    real, intent(in) :: beta, Atilde, xi0, latitude

    ! Subroutine variables
    integer :: i, j, k, l, v0_ind, ind_i, ind_im1
    real :: xip(xipN), zetap(zetapN)
    real :: dxi, dzeta, dtau
    complex :: int_dxip_dzetap(zetapN)
    logical :: mask_int_dxip_dzetap(zetapN)
    complex :: int_dxip(xipN)
    logical :: mask_int_dxip(xipN)
    real, parameter :: pi  = 4 * atan(1.0_8)
    complex :: psi_n_tau(xiN,zetaN)

    ! Outputs
    complex, intent(out) :: psi(xiN,zetaN,tauN)
    complex, intent(out) :: u(xiN,zetaN,tauN)
    complex, intent(out) :: v(xiN,zetaN,tauN)
    complex, intent(out) :: w(xiN,zetaN,tauN)
    real, intent(out) :: xi(xiN), zeta(zetaN), tau(tauN)

    ! Initialise arrays
    xi=createArray(xiMin, xiMax, xiN)
    zeta=createArray(zetaMin, zetaMax, zetaN)
    xip=createArray(xipMin, xipMax, xipN)
    zetap=createArray(zetapMin, zetapMax, zetapN)

    dtau = 2.0 * pi / (tauN - 1)
    tau=createArray(0.0, 2.0*pi - dtau, tauN)

    ! Perform numerical integration
    do i=1, xiN, 1
        do j=1, zetaN, 1
            do k=1, xipN, 1
                do l=1, zetapN, 1
                    int_dxip_dzetap(l) = &
                        funA(xi(i),xip(k),zeta(j),zetap(l)) * &
                        funB(xip(k),zetap(l),xi0)
                enddo

                mask_int_dxip_dzetap = ( &
                    int_dxip_dzetap==int_dxip_dzetap .and. &
                    abs(int_dxip_dzetap)<huge(real(int_dxip_dzetap(1))) &
                )

                int_dxip(k)=trapz( &
                    cmplx(pack(zetap,mask_int_dxip_dzetap)), &
                    pack(int_dxip_dzetap,mask_int_dxip_dzetap) &
                    )

            enddo

        mask_int_dxip = ( &
            int_dxip==int_dxip .and. abs(int_dxip)<huge(real(int_dxip(1))) &
        )

        psi_n_tau(i,j)=trapz( &
            cmplx(pack(xip,mask_int_dxip)), &
            pack(int_dxip,mask_int_dxip) &
        )
        enddo
    enddo

    ! Add time dependence
    do k=1, tauN, 1
        psi(:,:,k) = psi_n_tau * sin(tau(k))
    enddo

    ! Scale
    psi = -(beta * xi0 * Atilde / (4*pi)) * psi

    ! Calculate psi partials using centred finite differencing, and
    ! forward/backward differencing at endpoints.
    dxi = (xiMax - xipMin) / (xiN - 1)
    dzeta = (zetaMax - zetaMin) / (zetaN - 1)

    ! w = -dpsi_dxi
    w(2:xiN-1,:,:) = (psi(3:,:,:) - psi(1:xiN-2,:,:)) / (2*dxi)
    w(1,:,:) = (psi(2,:,:) - psi(1,:,:)) / dxi
    w(xiN,:,:) = (psi(xiN,:,:) - psi(xiN-1,:,:)) / dxi
    w = -w

    ! u = dpsi_dzeta
    u(:,2:zetaN-1,:) = (psi(:,3:,:) - psi(:,1:zetaN-2,:)) / (2*dzeta)
    u(:,1,:) = (psi(:,2,:) - psi(:,1,:)) / dzeta
    u(:,zetaN,:) = (psi(:,zetaN,:) - psi(:,zetaN-1,:)) / dzeta

    ! Calculate v from dv_dtau * omega + f * u = 0
    ! v is a perturbation, so it should integrate to zero.
    ! This can be achieved by assuming v=0 when abs(u) is maximised.
    ! abs(u) is maximised minimised at tau = pi / 2.

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

    !v(:,:,1) = ( &
    !    v(:,:,tauN) - &
    !    (u(:,:,1) + u(:,:,tauN)) * dtau * sin(latitude * pi / 180) &
    !)

end subroutine solveCaseOne

! Term in integrand of psi
function funA(xi,xip,zeta,zetap) result(f)

    real, intent(in) :: xi, xip, zeta, zetap
    real :: f
    f = log(((xi-xip)**2+(zeta-zetap)**2)/((xi-xip)**2+(zeta+zetap)**2))

end function funA

! Term in integrand for psi, psiXi and psiZeta
function funB(xip,zetap,xi0) result(f)

    real, intent(in) :: xip, zetap, xi0
    real :: f
    f=exp(-zetap)/(xip**2+xi0**2)

end function funB

end program rotunnoCaseOne
