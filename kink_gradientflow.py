# -*- coding: utf-8 -*-
"""
GRADIENT FLOW ALGORITHM FOR phi^4 KINKS

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


#.............PARAMETERS  ..................

n = 101 #number of points in the interval
h = 0.05 #space step
xmax=(n-1)*h;
f=np.zeros(n)
printing= 100;            # Every number of iterations  we  print the information..
plotting =  100           # Every number of iterations  we  plot the configuration..
delta=1.*pow(10,-3);      # This is the  time step. One has to play a little bit to find optimal values. If it is too high, it won't converge, and if it is too low, it will but slowly.
precision = 0.00001       # This tells us when to stop the algorithm
deltaE = 1 #
max_iters = 100000 # maximum number of iterations
iters = 0 #iteration counter
#........... FIELD CONFIGURATIONS ......................
def line(xmax):
    # Linear configuration
    for j in range(n):
        x=j*h
        f[j]=x/xmax
    return f

#Exact kink solution:
fexact=np.zeros(n)
for j in range(n):
    x=j*h;
    fexact[j]=np.tanh(x);
fexact

'''''
# Other configuration
fother=np.zeros(n)
for j in range(n):
    x=j*h;
    fother[j]=np.sinh(x);
fexact
'''''
#................. ENERGY DENSITY.................
#compute energy density at a site of a field configuration f:
def sden(i,f):
  if i==0:
      df0=(f[i+1]-f[i])/h;
  if(i==n-1):
      df0=(f[i]-f[i-1])/h;
  else:
      df0=(f[i+1]-f[i-1])/(2.0*h);
  f0=f[i];
  den0=(0.5*pow(df0,2)+0.5*pow((1.0-pow(f0,2)),2))*h;
  return den0


#compute total energy density of a field configuration f:
def tden(f):
   tden=np.zeros(n)
   for j in range(n):
           tden[j]=sden(j,f);
   return tden

#compute total  energy of a field configuration:
def energy(f):
    energy=0.0;
    for j in range (1,n) :
      energy=energy+sden(j,f);
    return energy
#............... FIELD VARIATION............
#compute the variation of the field configuration f at a site:

def var_phi(i,f):
    f0=f[i];
    df0=(f[i+1]-f[i-1])/(2.0*h);
    d2f0=(f[i+1]-2.0*f0+f[i-1])/(h*h);
    var0=-2.0*f0*(1.0-f0*f0)-d2f0;
    return(var0)


#..........  PLOTS   ..................
x = np.linspace(0,1,n)
plt.subplot(2, 1, 1)
plt.plot(x,line(xmax))
plt.title('Plots')
plt.ylabel(r'$\phi$')
plt.subplot(2, 1, 2)
plt.plot(x,tden(line(xmax)))
#plt.yscale('log')
plt.xlabel('x')
plt.ylabel('Energy density')


############ GRADIENT FLOW ALGORITHM ###############

#Set initial conditions:
f = line(xmax) #set linear configuration as initial condition
varf=np.zeros(n)
plotcounter = 0
printcounter = 0
color=iter(cm.rainbow(np.linspace(0,1,15)))
while deltaE > precision and iters < max_iters:
    energy0=energy(f); #Store current energy value in energy0 
    
    #Calculating and storing the variation at each site:
    
    for j in range (1,n-1) : #( We do not vary the field at the boundary points)
        varf[j]=var_phi(j,f)
    # Implementing the variation of the field configuration:
    for j in range (1,n-1) :
        f[j]=f[j]-delta*varf[j]; #new field configuration
    energyf=energy(f); #Store energy value of the new field configuration in energyf
    deltaE = abs(energyf - energy0) #Change in x
    iters = iters+1 #iteration count
    if (printcounter == printing):    
        print("Iteration",iters,"Energy value is",energyf) #Print iterations
        printcounter = 0
    if (plotcounter == plotting):
        c=next(color)
        x = np.linspace(0,1,n)
        plt.subplot(2, 1, 1)
        plt.plot(x,f,c=c)
        plt.subplot(2, 1, 2)
        plt.plot(x,tden(f),c=c)
        plotcounter = 0
    printcounter += 1
    plotcounter += 1
print("The minimum energy configuration corresponds to", energyf)
plt.show()





'''
plt.subplot(2, 1, 1)
plt.plot(x,line(xmax))
plt.plot(x,fexact)
plt.plot(x,f)
plt.title('Plots')
plt.ylabel('Configurations')

plt.subplot(2, 1, 2)
plt.plot(x,tden(line(xmax)))
plt.plot(x,tden(fexact))
plt.plot(x,tden(f))
plt.xlabel('x')
plt.ylabel('Energy density')
'''
