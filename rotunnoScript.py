import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from rotunno import calcTheta

plt.close('all')

data = xr.open_dataset('rotunnoCaseTwo.nc')

# Load outputs
psi = data['psi'].values.T
u = data['u'].values.T
v = data['v'].values.T
w = data['w'].values.T

# Load domain variables
xi = data['xi'].values
zeta = data['zeta'].values
tau = data['tau'].values

# Load model parameters
beta = data.beta
Atilde = data.Atilde
xi0 = data.xi0

theta, theta0, thetaBar, thetaPrime = calcTheta(
        xi, zeta, tau, beta, Atilde, xi0, w
        )

# Redimensionalise before plotting
