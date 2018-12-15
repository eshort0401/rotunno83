# Third party libraries
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation
import numpy as np
import datetime

def redimensionalise(ds, h, f, N):
    
    # Specify constants
    omega = 7.2921159 * 10.0 ** (-5)
    
    ds = ds.assign_coords(zeta = ds.zeta * h)
    ds.zeta.attrs['units'] = 'm'
    ds = ds.assign_coords(tau = ds.tau / omega)
    ds.tau.attrs['units'] = 's'
    ds['u'] = ds.u * h * omega
    ds.u.attrs['units'] = 'm/s'
    ds['v'] = ds.v * h * omega
    ds.v.attrs['units'] = 'm/s'
    ds['w'] = ds.w * h * omega
    ds.w.attrs['units'] = 'm/s'
    ds['psi'] = ds.psi * h**2 * omega
    ds.psi.attrs['units'] = 'm^2/s'
    
    if ds.lat < 30:
        ds = ds.assign_coords(xi = ds.xi * N * h * ((omega**2 - f**2)**(-1/2)))
    elif ds.lat > 30: 
        ds = ds.assign_coords(xi = ds.xi * N * h * ((f**2-omega**2)**(-1/2)))
    
    ds.xi.attrs['units'] = 'm'

    return ds

def calcTheta(ds):
    
    w = ds['w'].values.T
    xi = ds['xi'].values
    zeta = ds['zeta'].values
    tau = ds['tau'].values
    xi0 = ds.xi0
    h = ds.h
    theta0 = ds.theta0
    theta1 = ds.theta1
    tropHeight = ds.tropHeight
    N = ds.N
    
    # Specify constants
    omega = 7.2921159 * 10.0 ** (-5)
    g = 9.80665
    
    thetaBar = np.zeros(np.shape(w))
    d_thetaBar_d_z = (theta1-theta0)/tropHeight
    d_thetaBar_d_zeta = h * d_thetaBar_d_z # Recall zeta * h = z
    
    for i in np.arange(0,np.size(tau)):
        thetaBar[:,:,i] = np.outer(
                np.ones(np.size(xi)), zeta * d_thetaBar_d_zeta
                )
    
    # Specify heating function array
    Q = np.zeros(np.shape(w))
    
    for i in np.arange(0,np.size(tau)):
        Q[:,:,i] = np.outer(
            ds.Atilde * ((np.pi / 2) + np.arctan(xi / xi0)), 
            np.exp(-zeta) 
            )
        Q[:,:,i] = Q[:,:,i] * np.sin(tau[i])
        
    LHS = np.zeros((np.size(tau),np.size(tau)))
    RHS = np.zeros(np.size(tau))
    dtau = tau[1]-tau[0]
    
    LHS[0, 0] = 1
    if (np.mod(np.size(tau),2) == 0):
        LHS[0,int(np.size(tau)/2)] = 1
    else:
        LHS[0,np.floor(np.size(tau)/2)] = 1/2
        LHS[0,np.floor(np.size(tau)/2)] = 1/2
    
    RHS[0] = 0
    for k in np.arange(1,np.size(tau)):
        LHS[k, np.mod(k+1,32)] = 1
        LHS[k, k] = -1
    
    # Initialise bouyancy matrix
    btilde = np.zeros(np.shape(w)) 
    
    # Calculate bouyancy
    for i in np.arange(0,np.size(xi)):
        for j in np.arange(0, np.size(zeta)):
            for k in np.arange(1,np.size(tau)):
                RHS[k] = dtau * (Q[i,j,k] - ((N/omega) ** 2) * w[i,j,k])
                
            btilde[i,j,:] = np.linalg.solve(LHS,RHS)
            
    # Convert btilde to potential temperature perturbation
    thetaPrime = btilde * (omega ** 2) * h * theta0 / g
            
    theta = theta0 + thetaBar + thetaPrime
    
    ds = ds.assign(theta=(('xi','zeta','tau'),theta))
    ds = ds.assign(thetaBar=(('xi','zeta','tau'),thetaBar))
    ds = ds.assign(thetaPrime=(('xi','zeta','tau'),thetaPrime))
    
    return ds

def calc_v(ds):
    w = ds['w'].values.T
    xi = ds['xi'].values
    zeta = ds['zeta'].values
    tau = ds['tau'].values
    lat=ds.lat
    
    # Specify constants
    omega = 7.2921159 * 10.0 ** (-5)
    f = 2 * omega * np.sin(np.deg2rad(lat))   
    
    LHS = np.zeros((np.size(tau),np.size(tau)))
    RHS = np.zeros(np.size(tau))
    dtau = tau[1]-tau[0]
    
    LHS[0, 0] = 1
    if (np.mod(np.size(tau),2) == 0):
        LHS[0,int(np.size(tau)/2)] = 1
    else:
        LHS[0,np.floor(np.size(tau)/2)] = 1/2
        LHS[0,np.floor(np.size(tau)/2)] = 1/2
        
    RHS[0] = 0
    for k in np.arange(1,np.size(tau)):
        LHS[k, np.mod(k+1,np.size(tau))] = 1
        LHS[k, k] = -1
        
    # Initialise v matrix
    v = np.zeros(np.shape(w)) 
    
    # Calculate bouyancy
    for i in np.arange(0,np.size(xi)):
        for j in np.arange(0, np.size(zeta)):
            for k in np.arange(1,np.size(tau)):
                RHS[k] = - dtau * (f/omega) * w[i,j,k]
                
            v[i,j,:] = np.linalg.solve(LHS,RHS)
            
    ds = ds.assign(v=(('xi','zeta','tau'),v))
            
    return ds
    

def plotPsi(ds,t=0):

    psi = ds['psi'].values.T
    xi = ds['xi'].values
    zeta = ds['zeta'].values
    
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
                             vmax=psiMax, cmap='RdBu_r')
    plt.title('Stream function [' + ds.psi.attrs['units'] + ']')
    plt.xlabel('Distance [' + ds.xi.attrs['units'] + ']')
    plt.ylabel('Height [' + ds.zeta.attrs['units'] + ']')
#    plt.colorbar(contourPlot, ticks=np.arange(psiMin,psiMax+1,1))
    plt.colorbar(contourPlot)

    dt=str(datetime.datetime.now())[0:-7]
    dt=dt.replace(" ", "_")
    dt=dt.replace(":", "_")
    dt=dt.replace("-", "")
    outFile='./figures/psi_' + dt + '.png'

    fig.savefig(outFile, dpi=80, writer='imagemagick')

    return fig, ax, contourPlot

def plotVelocity(ds, t=0):

    u = ds['u'].values.T
    w = ds['w'].values.T
    xi = ds['xi'].values
    zeta = ds['zeta'].values
        
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

    plt.title('Velocity [' + ds.u.attrs['units'] + ']')
    plt.xlabel('Distance [' + ds.xi.attrs['units'] + ']')
    plt.ylabel('Height [' + ds.zeta.attrs['units'] + ']')
    cbar=plt.colorbar(contourPlot)
    cbar.set_label('Speed  [' + ds.u.attrs['units'] + ']')

    dt=str(datetime.datetime.now())[0:-7]
    dt=dt.replace(" ", "_")
    dt=dt.replace(":", "_")
    dt=dt.replace("-", "")
    outFile='./figures/velocity_' + dt + '.png'

    plt.savefig(outFile, dpi=80, writer='imagemagick')

    return fig, ax, contourPlot

def animatePsi(ds):

    psi = ds['psi'].values.T
    xi = ds['xi'].values
    zeta = ds['zeta'].values
    tau = ds['tau'].values
    
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

    psiInc = np.round(np.ceil(np.max(np.abs(psi))*10)/100,1)
    psiMax = np.ceil(np.max(np.abs(psi))*10)/10
    levels=np.arange(-psiMax,psiMax+psiInc,psiInc)

    [Xi,Zeta]=np.meshgrid(xi,zeta,indexing='ij')

    contourPlot=ax.contourf(Xi,Zeta,psi[:,:,0],levels,vmin=-psiMax,
                             vmax=psiMax, cmap='RdBu_r')
    
    plt.title('Stream function [' + ds.psi.attrs['units'] + ']')
    plt.xlabel('Distance [' + ds.xi.attrs['units'] + ']')
    plt.ylabel('Height [' + ds.zeta.attrs['units'] + ']')
    cbar = plt.colorbar(contourPlot)
    cbar.set_label('[' + ds.psi.attrs['units'] + ']')

    def update(i):
        label = 'timestep {0}'.format(i)
        print(label)
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        contourPlot=ax.contourf(Xi,Zeta,psi[:,:,i],levels,vmin=-psiMax,
                               vmax=psiMax, cmap='RdBu_r')
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

def animateVelocity(ds):

    u = ds['u'].values.T
    w = ds['w'].values.T
    xi = ds['xi'].values
    zeta = ds['zeta'].values
    tau = ds['tau'].values
    
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
    speedInc = np.round(np.ceil(np.max(speed)*10)/100,1)    
    speedMax = np.ceil(np.max(speed)*10)/10
    speedMin=0
    levels=np.arange(speedMin,speedMax+speedInc,speedInc)

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

    plt.title('Velocity [' + ds.u.attrs['units'] + ']')
    plt.xlabel('Distance [' + ds.xi.attrs['units'] + ']')
    plt.ylabel('Height [' + ds.zeta.attrs['units'] + ']')
    cbar=plt.colorbar(contourPlot)
    cbar.set_label('Speed  [' + ds.u.attrs['units'] + ']')

    def update(i):
        label = 'timestep {0}'.format(i)
        print(label)
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        ax.collections = []
        
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

def animate_v(ds):

    u = ds['u'].values.T
    v = ds['v'].values
    w = ds['w'].values.T
    xi = ds['xi'].values
    zeta = ds['zeta'].values
    tau = ds['tau'].values
    
    # Plot
    plt.rc('text', usetex=False)

    plt.ioff()

    # Initialise fonts
    rcParams['font.family'] = 'serif'
    rcParams.update({'font.serif': 'Times New Roman'})
    rcParams.update({'font.size': 12})
    rcParams.update({'font.weight': 'normal'})

    print('Animating v.')

    # Velocity plot
    fig, ax = plt.subplots()

    #vInc = np.round(np.ceil(np.max(np.abs(v))*10)/100,1)
    vMax = np.ceil(np.max(np.abs(v))*10)/10
    vInc = vMax/10
    levels=np.arange(-vMax,vMax+vInc,vInc)

    [Xi,Zeta]=np.meshgrid(xi,zeta,indexing='ij')

    contourPlot=ax.contourf(Xi,Zeta,v[:,:,0],levels,vmin=-vMax,
                             vmax=vMax, cmap='RdBu_r')

    skip=int(np.floor(np.size(xi)/16))
    
    speed=(u**2+w**2)**(1/2)    
    speedMax = np.ceil(np.max(speed)*10)/10

    uQ=np.sign(u)*(np.abs(u)/speedMax)**(2/3)*speedMax
    wQ=np.sign(w)*(np.abs(w)/speedMax)**(2/3)*speedMax

    sQ=int(np.ceil(skip/2))
    speedMaxQ=np.amax((uQ[sQ::skip,sQ::skip,:]**2+ \
                       wQ[sQ::skip,sQ::skip,:]**2)**(1/2))

    dxi=xi[1]-xi[0]

    ax.quiver(Xi[sQ::skip,sQ::skip], Zeta[sQ::skip,sQ::skip],
               uQ[sQ::skip,sQ::skip,0], wQ[sQ::skip,sQ::skip,0],
               units='xy', scale=speedMaxQ/(skip*dxi))

    plt.title('v Velocity [' + ds.u.attrs['units'] + ']')
    plt.xlabel('Distance [' + ds.xi.attrs['units'] + ']')
    plt.ylabel('Height [' + ds.zeta.attrs['units'] + ']')
    cbar=plt.colorbar(contourPlot)
    cbar.set_label('v [' + ds.v.attrs['units'] + ']')

    def update(i):
        label = 'timestep {0}'.format(i)
        print(label)
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        ax.collections = []
        
        contourPlot=ax.contourf(Xi,Zeta,v[:,:,i],levels,vmin=-vMax,
                             vmax=vMax, cmap='RdBu_r')

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

    outFile='./figures/v_velocity_' + dt + '.gif'

    anim.save(outFile, dpi=80, writer='imagemagick')

    plt.close("all")

    return

def animateTheta(ds):

    xi = ds['xi'].values
    zeta = ds['zeta'].values
    tau = ds['tau'].values
    theta = ds['theta'].values
    u = ds['u'].values.T
    w = ds['w'].values.T

    # Plot
    plt.rc('text', usetex=False)

    plt.ioff()

    # Initialise fonts
    rcParams['font.family'] = 'serif'
    rcParams.update({'font.serif': 'Times New Roman'})
    rcParams.update({'font.size': 12})
    rcParams.update({'font.weight': 'normal'})

    print('Animating theta.')

    # Velocity plot
    fig, ax = plt.subplots()

    [Xi,Zeta]=np.meshgrid(xi,zeta,indexing='ij')
        
    thetaMax=np.ceil(np.max(theta)/20)*20
    thetaMin=np.floor(np.min(theta)/20)*20
    
    if thetaMax-thetaMin > 200:
        thetaInc = 40
    elif thetaMax-thetaMin > 100:
        thetaInc = 20
    else:
        thetaInc = 10

    levels=np.arange(thetaMin,thetaMax+thetaInc,thetaInc)
    
    contourPlot = ax.contourf(
        Xi, Zeta, theta[:,:,0], cmap='Reds',
        levels=levels, 
        )
    cbar=plt.colorbar(contourPlot)
    cbar.set_label('Potential Temperature [K]')
    
    ax.contour(
            Xi, Zeta, theta[:,:,0], colors='black',
            linestyles=None, levels=levels,
            linewidths=1.4
            )
    
    skip=int(np.floor(np.size(xi)/16))
    
    speed=(u**2+w**2)**(1/2)    
    speedMax = np.ceil(np.max(speed)*10)/10

    uQ=np.sign(u)*(np.abs(u)/speedMax)**(2/3)*speedMax
    wQ=np.sign(w)*(np.abs(w)/speedMax)**(2/3)*speedMax

    sQ=int(np.ceil(skip/2))
    speedMaxQ=np.amax((uQ[sQ::skip,sQ::skip,:]**2+ \
                       wQ[sQ::skip,sQ::skip,:]**2)**(1/2))

    dxi=xi[1]-xi[0]

    ax.quiver(Xi[sQ::skip,sQ::skip], Zeta[sQ::skip,sQ::skip],
               uQ[sQ::skip,sQ::skip,0], wQ[sQ::skip,sQ::skip,0],
               units='xy', scale=speedMaxQ/(skip*dxi))
    
    plt.title('Potential Temperature [K]')
    plt.xlabel('Distance [' + ds.xi.attrs['units'] + ']')
    plt.ylabel('Height [' + ds.zeta.attrs['units'] + ']')

    def update(i):
        label = 'timestep {0}'.format(i)
        print(label)
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        global contourPlot

        ax.collections = []
        
        contourPlot = ax.contourf(
            Xi, Zeta, theta[:,:,i], cmap='Reds',
            levels=levels, 
            )
        
        ax.contour(
            Xi, Zeta, theta[:,:,i], colors='black',
            linestyles=None, levels=levels,
            linewidths=1.4
            )

        ax.quiver(
            Xi[sQ::skip,sQ::skip], Zeta[sQ::skip,sQ::skip],
            uQ[sQ::skip,sQ::skip,i], wQ[sQ::skip,sQ::skip,i],
            units='xy', scale=speedMaxQ/(skip*dxi)
            )

        return contourPlot, ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, np.size(tau)),
                         interval=200)

    dt=str(datetime.datetime.now())[0:-7]
    dt=dt.replace(" ", "_")
    dt=dt.replace(":", "_")
    dt=dt.replace("-", "")

    outFile='./figures/theta_' + dt + '.gif'

    anim.save(outFile, dpi=80, writer='imagemagick')

    plt.close("all")

    return
