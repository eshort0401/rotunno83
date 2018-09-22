# Third party libraries
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation
import numpy as np
import datetime

def solveCaseOne(
    beta=7.27*10**(-3), Atilde=10**3, xi0=0.2, xiMin=-4, xiMax=4,
    dxi=0.1, zetaMin=0, zetaMax=4, dzeta=0.1, tauN=4
    ):

    xi=np.arange(xiMin,xiMax+dxi,dxi)
    zeta=np.arange(zetaMin,zetaMax+dzeta,dzeta)
    tau=np.arange(0,1,1/tauN)*2*math.pi

    dxiP=0.2
    dzetaP=0.2

    xiP=np.arange(-4,4+dxiP,dxiP)
    zetaP=np.arange(0,4+dzetaP,dzetaP)

    intPsi=np.full((np.size(xi),np.size(zeta)),np.nan)
    psi=np.full((np.size(xi),np.size(zeta),np.size(tau)),np.nan)

    print('Numerically integrating.')

    # Term in integrand of psi
    def funA(xi,xiP,zeta,zetaP,xi0):
        f = (((xi-xiP)**2+(zeta-zetaP)**2)/((xi-xiP)**2+(zeta+zetaP)**2))
        return f

    # Term in integrand for psi, psiXi and psiZeta
    def funB(xiP,zetaP,xi0):
        f=np.exp(-zetaP)/(xiP**2+xi0**2)
        return f

    xi4D, zeta4D, xiP4D, zetaP4D = np.meshgrid(xi, zeta, xiP, zetaP,
                                               indexing='ij')

    integrandPsi=np.log(funA(xi4D,xiP4D,zeta4D,zetaP4D,xi0)) \
                 *funB(xiP4D,zetaP4D,xi0)

    print(np.any(((xi4D == xiP4D) & (zeta4D == zetaP4D)) | \
    ((zeta4D == 0) & (zetaP4D == 0) & ((xi4D-xiP4D) == 0))))

    intPsi=np.trapz(
            np.trapz(integrandPsi,x=zetaP4D,axis=3), x=xiP4D[:,:,:,0],axis=2
            )
    intPsi=intPsi.reshape((np.size(xi),np.size(zeta),1))

    tau3D=tau.reshape(1,1,np.size(tau))
    psi=-(beta*xi0*Atilde/(4*math.pi))*intPsi*np.sin(tau3D)

    # Calculate psi partials using centred finite differencing
    psiXi=np.full((np.size(xi),np.size(zeta),np.size(tau)),np.nan);
    psiXi[1:-1,:,:]=(psi[2:,:,:]-psi[0:-2,:,:])/(2*dxi)
    psiXi[0,:,:,]=(psi[1,:,:]-psi[0,:,:])/dxi
    psiXi[-1,:,:,]=(psi[-1,:,:]-psi[-2,:,:])/dxi

    psiZeta=np.full((np.size(xi),np.size(zeta),np.size(tau)),np.nan);
    psiZeta[:,1:-1,:]=(psi[:,2:,:]-psi[:,0:-2,:])/(2*dxi)
    psiZeta[:,0,:,]=(psi[:,1,:]-psi[:,0,:])/dxi
    psiZeta[:,-1,:,]=(psi[:,-1,:]-psi[:,-2,:])/dxi

    w=-psiXi
    u=psiZeta

    return psi, u, w, xi, zeta, tau

def solveCaseTwo(
    beta=7.27*10**(-3), Atilde=10**3, xi0=0.2, xiMin=-2, xiMax=2,
    dxi=0.01, zetaMin=0, zetaMax=4, dzeta=0.01, kMax=5, dk=0.01,
    tauN=32
    ):

    xi=np.arange(xiMin,xiMax+dxi,dxi)
    zeta=np.arange(zetaMin,zetaMax+dzeta,dzeta)
    tau=np.arange(0,1,1/tauN)*2*math.pi

    k=np.arange(0,kMax+dk,dk)

    psi=np.full((np.size(xi),np.size(zeta),np.size(tau)),np.nan)

    # Term in integrand of psi
    def funA(xi,zeta,tau,k,xi0):
        f = np.cos(k*xi)*np.exp(-xi0*k)/(1+k**2)
        return f

    # Term in integrand for psi, psiXi and psiZeta
    def funB(xi,zeta,tau,k,xi0):
        f=np.sin(k*zeta+tau)-np.exp(-zeta)*np.sin(tau)
        return f

    def funC(xi,zeta,tau,k,xi0):
        f = -k*np.sin(k*xi)*np.exp(-xi0*k)/(1+k**2)
        return f

    def funD(xi,zeta,tau,k,xi0):
        f=k*np.cos(k*zeta+tau)+np.exp(-zeta)*np.sin(tau)
        return f

    xi4D, zeta4D, tau4D, k4D = np.meshgrid(xi, zeta, tau, k,
                                               indexing='ij')

    integrandPsi=-beta*Atilde*(funA(xi4D,zeta4D,tau4D,k4D,xi0)) \
                 *funB(xi4D,zeta4D,tau4D,k4D,xi0)
    integrandPsiXi=-beta*Atilde*(funC(xi4D,zeta4D,tau4D,k4D,xi0)) \
                 *funB(xi4D,zeta4D,tau4D,k4D,xi0)
    integrandPsiZeta=-beta*Atilde*(funA(xi4D,zeta4D,tau4D,k4D,xi0)) \
                 *funD(xi4D,zeta4D,tau4D,k4D,xi0)

    print('Numerically integrating psi.')
    psi=np.trapz(integrandPsi,dx=dk,axis=3)
    print('Numerically integrating psiXi.')
    psiXi=np.trapz(integrandPsiXi,dx=dk,axis=3)
    print('Numerically integrating psiZeta.')
    psiZeta=np.trapz(integrandPsiZeta,dx=dk,axis=3)

    w=-psiXi
    u=psiZeta

    return psi, u, w, xi, zeta, tau

def plotPsi(psi,xi,zeta,tau,t=0):

    # Plot
    plt.rc('text', usetex=False)

    plt.ion()

    # Initialise fonts
    rcParams['font.family'] = 'serif'
    rcParams.update({'font.serif': 'Times New Roman'})
    rcParams.update({'font.size': 12})
    rcParams.update({'font.weight': 'normal'})

    print('Plotting stream function.')

    # psi plot
    fig, ax = plt.subplots()

    psiMax=np.ceil(np.amax(psi))
    psiMin=np.floor(np.amin(psi))

    levels=np.arange(psiMin,psiMax+0.5,0.5)

    [Xi,Zeta]=np.meshgrid(xi,zeta,indexing='ij')

    contourPlot=ax.contourf(Xi,Zeta,psi[:,:,t],levels,vmin=psiMin,
                             vmax=psiMax, cmap='RdBu')
    plt.title('Stream function')
    plt.xlabel('Distance [-]')
    plt.ylabel('Height [-]')
    plt.colorbar(contourPlot, ticks=np.arange(psiMin,psiMax+1,1))

    return fig, ax, contourPlot

def plotVelocity(u,w,xi,zeta,tau,t=0):

    plt.rc('text', usetex=False)

    # Initialise fonts
    rcParams['font.family'] = 'serif'
    rcParams.update({'font.serif': 'Times New Roman'})
    rcParams.update({'font.size': 12})
    rcParams.update({'font.weight': 'normal'})

    print('Plotting velocity field.')

    # Velocity plot
    fig, ax = plt.subplots()

    speed=(u**2+w**2)**(1/2)

    speedMax=np.ceil(np.amax(speed)/2)*2
    speedMin=0

    levels=np.arange(speedMin,speedMax+2,2)

    [Xi,Zeta]=np.meshgrid(xi,zeta,indexing='ij')

    contourPlot=ax.contourf(Xi,Zeta,speed[:,:,t],levels,vmin=speedMin,
                             vmax=speedMax, cmap='Reds')

    skip=int(np.floor(np.size(xi)/12))

    uQ=np.sign(u)*(np.abs(u)/speedMax)**(2/3)*speedMax
    wQ=np.sign(w)*(np.abs(w)/speedMax)**(2/3)*speedMax

    sQ=int(np.ceil(skip/2))
    speedMaxQ=np.amax((uQ[sQ::skip,sQ::skip,:]**2+ \
                       wQ[sQ::skip,sQ::skip,:]**2)**(1/2))

    dxi=xi[1]-xi[0]

    ax.quiver(Xi[sQ::skip,sQ::skip], Zeta[sQ::skip,sQ::skip],
               uQ[sQ::skip,sQ::skip,t], wQ[sQ::skip,sQ::skip,t],
               units='xy', scale=speedMaxQ/(skip*dxi))

    plt.title('Velocity')
    plt.xlabel('Distance [-]')
    plt.ylabel('Height [-]')
    plt.colorbar(contourPlot, ticks=np.arange(speedMin,speedMax+4,4))

    return fig, ax, contourPlot

def animatePsi(psi,xi,zeta,tau):

    # Plot
    plt.rc('text', usetex=False)

    plt.ioff()

    # Initialise fonts
    rcParams['font.family'] = 'serif'
    rcParams.update({'font.serif': 'Times New Roman'})
    rcParams.update({'font.size': 12})
    rcParams.update({'font.weight': 'normal'})

    print('Animating stream function.')

    # psi plot
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    psiMax=np.ceil(np.amax(psi))
    psiMin=np.floor(np.amin(psi))

    levels=np.arange(psiMin,psiMax+0.5,0.5)

    [Xi,Zeta]=np.meshgrid(xi,zeta,indexing='ij')

    contourPlot=ax.contourf(Xi,Zeta,psi[:,:,0],levels,vmin=psiMin,
                             vmax=psiMax, cmap='RdBu')
    plt.title('Stream function')
    plt.xlabel('Distance [-]')
    plt.ylabel('Height [-]')
    plt.colorbar(contourPlot, ticks=np.arange(psiMin,psiMax+1,1))

    def update(i):
        label = 'timestep {0}'.format(i)
        print(label)
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        contourPlot=ax.contourf(Xi,Zeta,psi[:,:,i],levels,vmin=psiMin,
                               vmax=psiMax, cmap='RdBu')
        return contourPlot, ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, np.size(tau)),
                         interval=200)

    dt=str(datetime.datetime.now())[0:-7]
    dt=dt.replace(" ", "_")
    dt=dt.replace(":", "_")
    dt=dt.replace("-", "")

    outFile='./figures/psi_' + dt + '.gif'

    anim.save(outFile, dpi=80, writer='imagemagick')

    plt.close("all")

    return

def animateVelocity(u,w,xi,zeta,tau):

    # Plot
    plt.rc('text', usetex=False)

    plt.ioff()

    # Initialise fonts
    rcParams['font.family'] = 'serif'
    rcParams.update({'font.serif': 'Times New Roman'})
    rcParams.update({'font.size': 12})
    rcParams.update({'font.weight': 'normal'})

    print('Animating velocity field.')

    # Velocity plot
    fig, ax = plt.subplots()

    speed=(u**2+w**2)**(1/2)

    speedMax=np.ceil(np.amax(speed)/2)*2
    speedMin=0

    levels=np.arange(speedMin,speedMax+2,2)

    [Xi,Zeta]=np.meshgrid(xi,zeta,indexing='ij')

    contourPlot=ax.contourf(Xi,Zeta,speed[:,:,0],levels,vmin=speedMin,
                             vmax=speedMax, cmap='Reds')

    skip=int(np.floor(np.size(xi)/12))

    uQ=np.sign(u)*(np.abs(u)/speedMax)**(2/3)*speedMax
    wQ=np.sign(w)*(np.abs(w)/speedMax)**(2/3)*speedMax

    sQ=int(np.ceil(skip/2))
    speedMaxQ=np.amax((uQ[sQ::skip,sQ::skip,:]**2+ \
                       wQ[sQ::skip,sQ::skip,:]**2)**(1/2))

    dxi=xi[1]-xi[0]

    ax.quiver(Xi[sQ::skip,sQ::skip], Zeta[sQ::skip,sQ::skip],
               uQ[sQ::skip,sQ::skip,0], wQ[sQ::skip,sQ::skip,0],
               units='xy', scale=speedMaxQ/(skip*dxi))

    plt.title('Velocity')
    plt.xlabel('Distance [-]')
    plt.ylabel('Height [-]')
    cbar=plt.colorbar(contourPlot, ticks=np.arange(speedMin,speedMax+4,4))
    cbar.set_label('Speed [-]')

    def update(i):
        label = 'timestep {0}'.format(i)
        print(label)
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        contourPlot=ax.contourf(Xi,Zeta,speed[:,:,i],levels,vmin=speedMin,
                             vmax=speedMax, cmap='Reds')
        ax.quiver(Xi[sQ::skip,sQ::skip], Zeta[sQ::skip,sQ::skip],
               uQ[sQ::skip,sQ::skip,i], wQ[sQ::skip,sQ::skip,i],
               units='xy', scale=speedMaxQ/(skip*dxi))

        return contourPlot, ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, np.size(tau)),
                         interval=200)
    anim.save('test2.gif', dpi=80, writer='imagemagick')

    plt.close("all")

    return

def read(filename):

    rotunno=np.load(filename)
    psi=rotunno['psi']
    u=rotunno['u']
    w=rotunno['w']
    xi=rotunno['xi']
    zeta=rotunno['zeta']
    tau=rotunno['tau']

    return psi, u, w, xi, zeta, tau
