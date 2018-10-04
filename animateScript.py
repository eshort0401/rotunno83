import xarray as xr
import matplotlib.pyplot as plt
from rotunno import animateVelocity, animatePsi, plotVelocity, plotPsi

plt.close('all')

data = xr.open_dataset('rotunnoCaseTwo.nc')

psi = data['psi'].values.T
u = data['u'].values.T
v = data['v'].values.T
w = data['w'].values.T
xi = data['xi'].values
zeta = data['zeta'].values
tau = data['tau'].values

# Calculate dimensional variables

plotVelocity(u,w,xi,zeta,tau)
plotPsi(psi,xi,zeta,tau)
