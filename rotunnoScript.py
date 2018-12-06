import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from rotunno import calculatePotentialTemperature

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

# Specify dimensional parameters and constants
N = 0.005
omega = 2*np.pi/(24*3600)
h = 10 ** 4 / 4
theta0 = 300
g = 9.81
thetaBar = 300 + zeta * h * 0.006

thetaBar = np.outer(thetaBar, np.ones(np.size(zeta)))

# Specify heating function array
Q = np.zeros(np.shape(psi))

for i in np.arange(0,np.size(tau)):
    Q[:,:,i] = np.outer(
        Atilde * ((np.pi / 2) + np.arctan(xi / xi0)), 
        np.exp(-zeta) 
        )
    Q[:,:,i] = Q[:,:,i] * np.sin(tau[i])
    
LHS = np.zeros((np.size(tau),np.size(tau)))
RHS = np.zeros(np.size(tau))
dtau = tau[1]-tau[0]

LHS[0, 0] = 1
RHS[0] = 0
for k in np.arange(1,np.size(tau)):
    LHS[k, np.mod(k+1,32)] = 1
    LHS[k, k] = -1

# Initialise bouyancy matrix
btilde = np.zeros(np.shape(psi)) 

# Calculate bouyancy
for i in np.arange(0,np.size(xi)):
    for j in np.arange(0, np.size(zeta)):
        for k in np.arange(1,np.size(tau)):
            RHS[k] = dtau * (Q[i,j,k] - ((N/omega) ** 2) * w[i, j, k])
            
        btilde[i, j, :] = np.linalg.solve(LHS,RHS)
        
# Convert btilde to potential temperature perturbation
thetaPrime = btilde * (omega ** 2) * h * theta0 / g
        
