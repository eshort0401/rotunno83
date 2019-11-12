/* Standard headers */
#import <stdio.h>
#include <limits.h>
/* Standard library for complex numbers */
/* Functions like cos must be replaced by ccos etc */
#include <complex.h>

double complex psi[81][41][32], u[81][41][32], v[81][41][32], w[81][41][32];
double xi[81], zeta[41], tau[32];

void solveCaseOne(
    double xiMin, xiMax, int xiN,
    double zetaMin, zetaMax, int zetaN,
    double xipMin, xipMax, int xipN,
    double zetapMin, zetapMax, int zetapN,
    int tauN, double beta, Atilde, xi0, latitude,
    double *psi[][][], *u[][][], *v[][][], *w[][][],
    double *xi[], *zeta[], *tau[]
);

void solveCaseOne(
    double xiMin, xiMax, int xiN,
    double zetaMin, zetaMax, int zetaN,
    double xipMin, xipMax, int xipN,
    double zetapMin, zetapMax, int zetapN,
    int tauN, double beta, Atilde, xi0, latitude,
    double complex *psi[][][], *u[][][], *v[][][], *w[][][],
    double *xi[], *zeta[], *tau[]
) {

int i, j, k, l v0_ind, ind_i, ind_im1;
double xip[xipN], zetap[zetapN];
double dxi, dzeta, dtau;
double complex int_dxip_dzetap[zetapN]
double complex 

}

int main()
{



    printf("Hello, World!\n");
    return 0;
}
