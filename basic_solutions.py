import numpy as np
import matplotlib.pyplot as plt

z = np.arange(0, 2, 1e-3)
x = np.arange(-1, 1, 1e-3)
X, Z = np.meshgrid(x, z)

N = 3
H1 = 1
k = 2*np.pi/2
m = k

R = np.cos(m*H1) + 1j*N*np.sin(m*H1)

psi = np.zeros(Z.shape).astype(complex)
psi[0:1000, :] = (N+1)/2*np.exp(1j*m*(Z[0:1000, :]-H1))
psi[0:1000, :] += (1-N)/2*np.exp(-1j*m*(Z[0:1000, :]-H1))
psi[0:1000, :] *= np.exp(1j*k*X[0:1000, :])
psi[1000:, :] = np.exp(1j*m*N*(Z[1000:, :]-H1))*np.exp(1j*k*X[1000:, :])

t = np.arange(0, np.pi/2, np.pi/8)
t_lab = ['0', 'pi/8', 'pi/4', '3*pi/8']

dl = .4
levels = np.arange(-4, 4+dl, dl)
styles = []

for i in range(len(levels)):
    if levels[i] >= 0:
        styles.append('-')
    else:
        styles.append('--')

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i in range(len(t)):
    axes.flatten()[i].plot([-1, 1], [1, 1], '--', color='grey')
    axes.flatten()[i].set_title('t = {}'.format(t_lab[i]))
    con = axes.flatten()[i].contour(
        X, Z, np.real(psi*np.exp(1j*t[i])),
        levels=levels, linestyles=styles, colors='k')
    axes.flatten()[i].set_xlim([-1, 1])
    axes.flatten()[i].set_ylim([0, 2])
    axes.flatten()[i].set_aspect('equal')
#     plt.colorbar(con, ax=axes.flatten()[i])
    
plt.savefig(
    './basic_sln1.png', 
    dpi=200, bbox_inches='tight', facecolor='white')

psi = np.zeros(Z.shape).astype(complex)
psi[0:1000, :] = np.sin(m*Z[0:1000, :])*np.exp(1j*k*X[0:1000, :])
psi[1000:, :] = np.abs(R)/N*np.sin(m*N*(Z[1000:, :]-H1)+np.angle(R))*np.exp(1j*k*X[1000:, :])

t = np.arange(0, np.pi/2, np.pi/8)
t_lab = ['0', 'pi/8', 'pi/4', '3*pi/8']

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
dl = .2
levels = np.arange(-4, 4+dl, dl)
styles = []

for i in range(len(levels)):
    if levels[i] >= 0:
        styles.append('-')
    else:
        styles.append('--')

for i in range(len(t)):
    axes.flatten()[i].plot([-1, 1], [1, 1], '--', color='grey')
    axes.flatten()[i].set_title('t = {}'.format(t_lab[i]))
    axes.flatten()[i].contour(
        X, Z, np.real(psi*np.exp(1j*t[i])),
        levels=levels, linestyles=styles, colors='k')
    axes.flatten()[i].set_xlim([-1, 1])
    axes.flatten()[i].set_ylim([0, 2])
    axes.flatten()[i].set_aspect('equal')
    
plt.savefig(
    './basic_sln2.png', 
    dpi=200, bbox_inches='tight', facecolor='white')
