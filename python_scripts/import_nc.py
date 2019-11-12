import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from rotunno import plotVelocity


plt.close('all')

data = xr.open_dataset('rotunnoCaseOne.nc')

#plotVelocity(
#    data['v'].values.T,data['w'].values.T,
#    data['xi'].values,data['zeta'].values,
#    data['tau'].values,t=8
#)

#x = data['u'].values[:,10,41]
#y = data['v'].values[:,10,41]
#
#plt.plot(x,y,'-o')

#
#plt.contourf(data['w'].values[3,:,:])
#plt.colorbar()
#plt.show()

x = data['xi'].values
y = np.arange(-4,4.1,0.1)
z = data['zeta'].values

time=5

u = data['u'][time,:,:].T.values
v = data['v'][time,:,:].T.values
w = data['w'][time,:,:].T.values

X, Y, Z = np.meshgrid(x,y,z,indexing = 'ij')

U = np.empty(X.shape)*np.nan
V = np.empty(Y.shape)*np.nan
W = np.empty(Z.shape)*np.nan

for i in np.arange(0, x.size):
    for j in np.arange(0, z.size):
        
        U[i,41,j] = u[i,j]
        V[i,41,j] = v[i,j]
        W[i,41,j] = w[i,j]

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_xlim3d(-4, 4)
ax.set_ylim3d(-4,4)
ax.set_zlim3d(0,4)

ax.set_xlabel('$x$')
ax.set_ylabel('$\eta$')
ax.set_zlabel('$\zeta$')
#ax.axis('equal')

#ax.quiver(X, Y, Z, U, V, W, length = .5)

step=5
scale=5

ax.quiver(X[::step,:,::step], 
          Y[::step,:,::step], 
          Z[::step,:,::step], 
          U[::step,:,::step]/scale, 
          V[::step,:,::step]/scale, 
          W[::step,:,::step]/scale, 
          length = .5)

