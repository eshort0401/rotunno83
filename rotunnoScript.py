from rotunno import solveCaseOne
from rotunno import solveCaseTwo
from rotunno import plotPsi
import numpy as np
import datetime

psi, u, w, xi, zeta, tau = solveCaseOne()

plotPsi(psi, xi, zeta, tau, t=3)
#plotVelocity(u,w,xi,zeta,tau,t=8)

#print(psi[:,:,1])

dt=str(datetime.datetime.now())[0:-7]
dt=dt.replace(" ", "_")
dt=dt.replace(":", "_")
dt=dt.replace("-", "")

outFile='./output/rotunno_' + dt + '.npz'

np.savez_compressed(outFile, psi=psi, u=u, w=w, xi=xi, zeta=zeta, tau=tau)
