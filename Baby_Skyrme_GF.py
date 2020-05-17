"""
GRADIENT FLOW ALGORITHM FOR BABY SKYRMIONS

author: Alberto García Martín-Caro
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D

#.............PARAMETERS  ..................

n = 15 #number of grid points 
h = 0.2  #space step
xmax=(n-1)*h
ymax=(n-1)*h
printing= 5;            # Every number of iterations  we  print the information..
plotting =  100           # Every number of iterations  we  plot the configuration..
delta=5.*pow(10,-3);      # This is the  time step. One has to play a little bit to find optimal values. If it is too high, it won't converge, and if it is too low, it will but slowly.
precision = 0.00001       # This tells us when to stop the algorithm
deltaE = 1 #
max_iters = 100000 # maximum number of iterations
iters = 0 #iteration counter
mu2= 0.1 #Potential length scale

#........... FIELD CONFIGURATIONS ......................
#................ FIELD VARIABLES..................
#We have 3 field variables, a,b,c, subject to a normalization condition: a^2+b^2+c^2=1.
# Each one is a function over the 2d plane, i.e
# a=a[i,j], b=b[i,j], c=c[i,j]  (nxn arrays)

#we can define a vector field ϕ (denoted by f) whose components are a,b,c:
#f=[a,b,c], --> f[i,j]=[a[i,j],b[i,j],c[i,j]], f[0][i,j]=a[i,j] and so forth

def hedgehog(B):
#....HEDGEHOG ANSATZ CONFIGURATION
#.... B = Topological charge
    a=np.zeros((2*n,2*n))
    b=np.zeros((2*n,2*n))
    c=np.zeros((2*n,2*n))
    for j in range(0,2*n):
        y=(j-n)*h
        for i in range(0,2*n):
            x=(i-n)*h
            r2=x**2+y**2
            if j<n:
                th = -np.arccos(x/np.sqrt(r2))
            elif i==j and i==n:
                th = 0.0
            else:
                th = np.arccos(x/np.sqrt(r2))
          
            xi = np.pi/(1+r2)
            if B==0:    
                a[i,j]=2*x/(1+r2)
                b[i,j]=-2*y/(1+r2)
                c[i,j]=(r2-1)/(1+r2)
            else:
                a[i,j]=np.sin(xi)*np.sin(B*th)
            
                b[i,j]=np.sin(xi)*np.cos(B*th)
            
                c[i,j]=np.cos(xi)
    hed0=[a,b,c]
    return hed0

#Plot the ansatz configuration field
hed = hedgehog(1)
x = np.linspace(-n*h,n*h,2*n)
y = np.linspace(-n*h,n*h,2*n)
#plt.subplot(2, 1, 1)
X,Y =np.meshgrid(x,y)
plt.quiver(X, Y, hed[0], hed[1], hed[2], units='width',pivot='mid')
#plt.subplot(2, 1, 2)
#plt.contourf(X, Y, f[2])
#plt.contourf(-X, Y, f[2])
#plt.contourf(X, -Y, f[2])
#plt.contourf(-X, -Y, f[2])
plt.colorbar()

#......FINITE DIFFERENCE DERIVATIVES........
#Compute the field derivatives(in a given field configuration f) at a site:

#...First derivatives
def dxf(i,j,f):
    dxf0=[0,0,0]
    for a in range(0,3):
        if i==0:
            dxf0[a]=((f[a][i+1,j]-f[a][i,j])/h)
        if(i==2*n-1):
            dxf0[a]=((f[a][i,j]-f[a][i-1,j])/h)
        else:
            dxf0[a]=((f[a][i+1,j]-f[a][i-1,j])/(2.0*h))
    return dxf0
 
def dyf(i,j,f):
    dyf0=[0,0,0]
    for a in range(0,3):
        if j==0:
            dyf0[a]=((f[a][i,j+1]-f[a][i,j])/h)
        if(j==2*n-1):
            dyf0[a]=((f[a][i,j]-f[a][i,j-1])/h)
        else:
            dyf0[a]=((f[a][i,j+1]-f[a][i,j-1])/(2.0*h))
    return dyf0    


#...Second derivatives
def dxxf(i,j,f):
    dxxf0=[0,0,0]
    for a in range(0,3):
        if i==0:
            dxxf0[a] = (1*f[a][i,j]-2*f[a][i+1,j]+f[a][i+2,j])/(1*1.0*h**2)
        dxxf0[a] = (1*f[a][i-1,j]-2*f[a][i,j]+1*f[a][i+1,j])/(1*1.0*h**2)
    return dxxf0  

def dyyf(i,j,f):
    dyyf0=[0,0,0]
    for a in range(0,3):
        if j==0:
            dyyf0[a] = (1*f[a][i,j]-2*f[a][i,j+1]+f[a][i,j+2])/(1*1.0*h**2)
        if j==2*n-1:
            dyyf0[a] = (1*f[a][i,j-2]-2*f[a][i,j-1]+f[a][i,j])/(1*1.0*h**2)
        else: 
            dyyf0[a] = (f[a][i,j-1]-2*f[a][i,j]+f[a][i,j+1])/(1*1.0*h**2)
    return dyyf0  

def dxyf(i,j,f):
    dxyf0=[0,0,0]
    for a in range(0,3):
        dxyf0[a] = (1*f[a][i+1,j+1]-f[a][i-1,j+1]-f[a][i+1,j-1]+f[a][i-1,j-1])/(4*1.0*h**2)
    return dxyf0  
    

#................. ENERGY DENSITY.................
#compute energy density at a site of a field configuration f:
def sden(i,j,f):   
  f0=f
  dxf0=dxf(i,j,f)
  dyf0=dyf(i,j,f)
  dxf2=np.dot(dxf0,dxf0)
  dyf2=np.dot(dyf0,dyf0)
  dxfdyf= np.dot(dxf0,dyf0)
  V=mu2*(1-f0[2][i,j])
  den0=(0.5*(dxf2+dyf2)+0.25*(dxf2*dyf2-dxfdyf**2)+V);
  return den0

#compute total energy density of a field configuration f:
def tden(f):
  Z=np.zeros((2*n,2*n))
  for i in range(2*n):
    for j in range(2*n):
        Z[i,j]=sden(i,j,hed)   
  return Z

#compute total  energy of a field configuration:
def energy(f):
    energy=0.0;
    for j in range (0,2*n) :
        for i in range (0,2*n):      
            energy=energy+sden(i,j,f)*h**2
    return energy


#Surface plot 
fig = plt.figure()
ax = fig.gca(projection='3d')
x = np.linspace(-(n)*h,(n)*h,2*n)
y = np.linspace(-(n)*h,(n)*h,2*n)
#plt.subplot(2, 1, 1)
X,Y =np.meshgrid(x,y)

surf = ax.plot_surface(X, Y, tden(hed), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
#surf = ax.plot_surface(-X, Y, tden(hed), cmap=cm.coolwarm,
                       #linewidth=0, antialiased=False)
#surf = ax.plot_surface(X, -Y, tden(hed), cmap=cm.coolwarm,
                       #linewidth=0, antialiased=False)
#surf = ax.plot_surface(-X, -Y, tden(hed), cmap=cm.coolwarm,
                       #linewidth=0, antialiased=False)

#............... FIELD VARIATION............
#Compute the (vector) quantity F at each site:

def F(i,j,f):
    F0=[0,0,0]
    Nunit=[0,0,1]
    #derivatives:
    dxf0=dxf(i,j,f)
    dyf0=dyf(i,j,f)
    dxxf0=dxxf(i,j,f)
    dyyf0=dyyf(i,j,f)
    dxyf0=dxyf(i,j,f)
    #....
    for a in range(0,3):
        F0[a]=np.dot(dxf0,dyf0)*dxyf0[a]+0.5*(np.dot(dyyf0,dxf0)-np.dot(dxyf0,dyf0))*dxf0[a]+0.5*(np.dot(dxxf0,dyf0)-np.dot(dxyf0,dxf0))*dyf0[a]-(1+0.5*np.dot(dyf0,dyf0))*dxxf0[a]-(1+0.5*np.dot(dxf0,dxf0))*dyyf0[a]-mu2*Nunit[a]
    return F0

#compute F·ϕ at each site
    
def Fphi(i,j,f):
    f0=[0,0,0]
    #derivatives:
    dxf0=dxf(i,j,f)
    dyf0=dyf(i,j,f)
    dxxf0=dxxf(i,j,f)
    dyyf0=dyyf(i,j,f)
    dxyf0=dxyf(i,j,f)
    #.....
    for a in range(0,3):
        f0[a]=f[a][i,j]
    F0phi=np.dot(dxf0,dyf0)*np.dot(dxyf0,f0)-(1+0.5*np.dot(dyf0,dyf0))*np.dot(f0,dxxf0)-(1+0.5*np.dot(dxf0,dxf0))*np.dot(f0,dyyf0)-mu2*f0[2]
    return F0phi

#compute the variation of the field configuration f at a site:

def var_phi(i,j,f):
    varf=[0,0,0]
    f0=[0,0,0]
    for a in range(0,3):
        f0[a]=f[a][i,j]
    F0=F(i,j,f)
    Ff=Fphi(i,j,f)
    for a in range (0,3):
        varf[a]=-F0[a]+2*Ff*f0[a]
    return(varf)


############ GRADIENT FLOW ALGORITHM ###############

#Set initial conditions:
f = hedgehog(1) #set linear configuration as initial condition
norm=0
varf=[0,0,0]
for a in range(0,3):
    varf[a]=np.zeros((2*n,2*n))
printcounter = 0
while deltaE > precision and iters < max_iters:
    energy0=energy(f); #Store current energy value in energy0 
    
    #Calculating and storing the variation at each site:
    for i in range(1,2*n-1):
        for j in range (1,2*n-1) : 
            for a in range(0,3):  #( We do not vary the field at the boundary points)
                
                varf[a][i,j]=var_phi(i,j,f)[a]
        # Implementing the variation of the field configuration:
    for i in range(2*n):
        for j in range (2*n) :
            norm=np.sqrt((f[0][i,j]+delta*varf[0][i,j])**2+(f[1][i,j]+delta*varf[1][i,j])**2+(f[2][i,j]+delta*varf[2][i,j])**2)#normalization
            for a in range(0,3):
                f[a][i,j]=(f[a][i,j]+delta*varf[a][i,j])/norm #new field configuration (normalized)
    energyf=energy(f); #Store energy value of the new field configuration in energyf
    deltaE = abs(energyf - energy0) #Change in energy
    iters = iters+1 #iteration count
    if (printcounter == printing):    
        print("Iteration",iters,"Energy value is",energyf) #Print results each ''printing'' number of iterations
        printcounter = 0
    printcounter += 1
print("The minimum energy configuration has an energy of", energyf)


x = np.linspace(-n*h,n*h,2*n)
y = np.linspace(-n*h,n*h,2*n)
X,Y =np.meshgrid(x,y)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9.75, 3))
im= axes[1].quiver(X, Y, f[0], f[1], f[2], units='width',pivot='mid')
axes[0].quiver(X, Y, hed[0], hed[1], hed[2], units='width',pivot='mid')
axes[0].set_title('Hedgehog ansatz')
axes[1].set_title('solution')
fig.colorbar(im, ax=axes.ravel().tolist())
plt.show()
#



fr=[]
hr=[]
r=[]
for i in range(0,2*n):
    r.append((i-n)*h)
    fr.append(f[2][i,i])
    hr.append(hed[2][i,i])
fig, ax = plt.subplots()
ax.plot(r,fr,label='solution')
ax.plot(r,hr,label='hedgehog ansatz')
ax.set_ylabel('$\phi^3$', rotation=0)
ax.set_xlabel('x')
plt.legend()
plt.show()



