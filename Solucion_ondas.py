# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 23:33:03 2017

@author: martin
"""
import numpy as np
from numpy import pi 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 100      # discretización de las dimensiones X, Y
space_step = 1/N
n = 500     # discretización del tiempo
T_max = 2#np.sqrt(2) - 4/n
time_step = T_max/n 
c = 1       # velocidad de propagación

X = np.linspace(0,1,N)
Y = np.linspace(0,1,N)
T = np.arange(0,T_max, time_step)

# Definimos el array que para cada tiempo, U[:,:,t] nos da la matriz U(x,y)
# de NxN con la posición vertical para cada (x,y)
U = np.zeros((N,N,n))

rh = c*(time_step)**2 / (space_step)**2   # factor en las diferencias finitas

DU = lambda x,y: 0 # condición incial, derivada respecto al tiempo
U_0 = lambda x,y: 10*np.exp((-(x-0.5)**2 - (y-0.5)**2)/0.005) #np.sin(3*pi*x)*np.sin(2*pi*y)

# Realizamos la iteración de t_0 a t_1 a partir de las condiciones iniciales
for i in range(N):
    for j in range(N):
        U[i,j,1] = U_0(X[i],Y[j])
        U[i,j,0] = U[i,j,1] - time_step*DU(X[i],Y[j])

# Realizamos la iteración a tiempos posteriores a mediante las diferencias
# finitas
for t in range(1,n-1):
    for x in range(1,N-1):
        for y in range(1,N-1):
          U[x,y,t+1]= rh*(U[x-1,y,t] + U[x+1,y,t] + U[x,y-1,t] + U[x,y+1,t] \
          - 4*U[x,y,t]) + 2*U[x,y,t] -U[x,y,t-1]
    
# Graficamos la solución para cada t y tomamos la frame
xx, yy = np.meshgrid(X,Y)
fig = plt.figure()
ax3 = Axes3D(fig)
ax3.set_zlim3d(-0.8, 0.8)
#ax3.title('U(x,y)')
#ax3.axis
for t in range(n):
    ax3.set_zlim3d(-10.0, 10.0)
    ax3.plot_surface(xx, yy, U[:,:,t], cmap = 'jet')
    plt.savefig('D:\\martin\\Documents\\Python Scripts\\Animation\\frames\\frame_{:03d}.png'.format(t))
    #plt.pause(0.0001)
    ax3.clear()
plt.close(fig)



