# Third party libraries
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation
import numpy as np
import datetime

def plotPsi(psi,xi,zeta,tau,t=0):

    # Plot
    plt.rc('text', usetex=False)

    plt.ioff()

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
#    plt.colorbar(contourPlot, ticks=np.arange(psiMin,psiMax+1,1))
    plt.colorbar(contourPlot)

    dt=str(datetime.datetime.now())[0:-7]
    dt=dt.replace(" ", "_")
    dt=dt.replace(":", "_")
    dt=dt.replace("-", "")
    outFile='./figures/psi_' + dt + '.png'

    fig.savefig(outFile, dpi=80, writer='imagemagick')

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
#    plt.colorbar(contourPlot, ticks=np.arange(speedMin,speedMax+4,4))
    plt.colorbar(contourPlot)

    dt=str(datetime.datetime.now())[0:-7]
    dt=dt.replace(" ", "_")
    dt=dt.replace(":", "_")
    dt=dt.replace("-", "")
    outFile='./figures/velocity_' + dt + '.png'

    plt.savefig(outFile, dpi=80, writer='imagemagick')

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
#    plt.colorbar(contourPlot, ticks=np.arange(psiMin,psiMax+1,1))
    plt.colorbar(contourPlot)

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

    skip=int(np.floor(np.size(xi)/16))

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
    cbar=plt.colorbar(contourPlot)
#    cbar=plt.colorbar(contourPlot, ticks=np.arange(speedMin,speedMax+4,4))
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

    dt=str(datetime.datetime.now())[0:-7]
    dt=dt.replace(" ", "_")
    dt=dt.replace(":", "_")
    dt=dt.replace("-", "")

    outFile='./figures/velocity_' + dt + '.gif'

    anim.save(outFile, dpi=80, writer='imagemagick')

    plt.close("all")

    return
    
def calculatePotentialTemperature(Q,N,w,t):
    
    
    
    return
