import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from rotunno import animateVelocity, animatePsi, calcTheta, calc_v, animateTheta, animate_v, redimensionalise

plt.close('all')

ds = xr.open_dataset('../rotunno83/rotunnoCaseTwo.nc')

# Specify constants
omega = 7.2921159 * 10.0 ** (-5)
g = 9.80665

# Load non-dimensional model parameters
beta = ds.beta
Atilde = ds.Atilde
xi0 = ds.xi0
lat = ds.lat
h = ds.h
Atilde = ds.Atilde

theta0 = ds.theta0
theta1 = ds.theta1
tropHeight = ds.tropHeight
delTheta = ds.delTheta

N = ds.N
f = 2.0 * omega * np.sin(np.deg2rad(lat))

ds = calcTheta(ds)
ds = calc_v(ds)

ds.xi.attrs['units'] = '-'
ds.zeta.attrs['units'] = '-'
ds.tau.attrs['units'] = '-'
ds.psi.attrs['units'] = '-'
ds.u.attrs['units'] = '-'
ds.v.attrs['units'] = '-'
ds.w.attrs['units'] = '-'

# Calculate dimensional variables
ds = redimensionalise(ds, h, f, N)
if ds.lat < 30:
    ds.attrs['xi0'] = ds.xi0 * N * h * ((omega**2 - f**2)**(-1/2))
elif ds.lat > 30: 
    ds.attrs['xi0'] = ds.xi0 * N * h * ((f**2-omega**2)**(-1/2))
    
# Animate
#animateVelocity(ds)
#animatePsi(ds)
#animateTheta(ds)
#animate_v(ds)
