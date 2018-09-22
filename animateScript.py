from rotunno import animateVelocity, animatePsi
from rotunno import read

psi, u, w, xi, zeta, tau = \
read('./output/rotunnoCaseOne_20180618_00_27_19.npz')

animateVelocity(u,w,xi,zeta,tau)
animatePsi(psi,xi,zeta,tau)