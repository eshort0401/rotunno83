import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from rotunno import animateVelocity, animatePsi, calcTheta, animateTheta, redimensionalise

plt.close('all')

ds = xr.open_dataset('../rotunno83/rotunnoCaseTwo.nc')

#psi = data['psi'].values.T
#u = data['u'].values.T
#v = data['v'].values.T
#w = data['w'].values.T
#xi = data['xi'].values
#zeta = data['zeta'].values
#tau = data['tau'].values

# Specify constants
omega = 2*np.pi/(24*3600)
g = 9.81

# Load non-dimensional model parameters
beta = ds.beta
Atilde = ds.Atilde
xi0 = ds.xi0
lat = ds.lat
f = 2.0 * omega * np.sin(np.deg2rad(lat))

# beta = omega**2/((omega**2-f**2)**(1/2)*N) so should specify before solving!

# Specify dimensional parameters
theta0 = 300.0 # Potential temperature (K) at surface
theta1 = 360.0 # Potential temperature (K) at tropopause
tropHeight = 11.0 * 10 ** 3 # Tropopause height (m)
h = (11.0 * 10 ** 3) / 8 # e-folding height for heating

d_thetaBar_d_z = (theta1-theta0)/tropHeight
d_thetaBar_d_zeta = h * d_thetaBar_d_z # Recall zeta * h = z
N = np.sqrt((g/theta0) * d_thetaBar_d_z) # Brunt Vaisala Frequency

theta, theta0, thetaBar, thetaPrime = calcTheta(
    ds, h, theta0, theta1, tropHeight
    )

ds.xi.attrs['units'] = '-'
ds.zeta.attrs['units'] = '-'
ds.tau.attrs['units'] = '-'
ds.psi.attrs['units'] = '-'
ds.u.attrs['units'] = '-'
ds.v.attrs['units'] = '-'
ds.w.attrs['units'] = '-'

# Calculate dimensional variables
ds = redimensionalise(ds, h, f, N)

# Animate
#animateVelocity(ds)
animatePsi(ds)
#animateTheta(ds, theta)
