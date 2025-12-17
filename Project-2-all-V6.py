"""
LPHYS2163 - Atmosphere and ocean : physics and dynamics : Poject 2
@author: charlie Taylor, Chloé Coppens, Julien Monfils

remark : AI has been used to generate some of the plots and to rewrite the code in a more readable way.
"""
#-----------------------------  Question 2  -----------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

#-----Variables

L = 15000
L2 = 5000
L3 = 25000
H = 1000
dT = 5
P1 = 85000
P0 = 100000
mu = 5*10**-3
R = 287.05

#---------------------------------------------------------------
#Bullet 1

a =R*np.log(P0/P1)
b =2*(H+L)

dvdt = (a/b)*dT
dvdt_print = round(dvdt, 5)

print("Tangential acceleration (no friction) =", dvdt_print, "m/s^2")

#---------------------------------------------------------------
#Bullet 2
v = (dvdt)/mu
v_print = round(v, 3)
print("Steady state velocity (friction) =", v_print, "m/s")

#---------------------------------------------------------------
#Bullet 3 & 6
v = 0
v_f = 0
v_q = 0
v_qf = 0
dt = 0.1
t = np.arange(0,3601,dt)
vs = []
vs2 = []
vs3 = []
vs_f = []
vs_f2 = []
vs_f3 = []
vs_q = []
vs_q2 = []
vs_q3 = []
vs_qf = []
vs_qf2 = []
vs_qf3 = []



Cd = 0.003
Omega = 7.272*10**-5
lat = 51*np.pi/180
f = 2*Omega*np.sin(lat)


for i in t:
    a = R*np.log(P0/P1)
    b = 2*(H+L)
    dvdt = ((a/b)*dT)
    dvdt_f = ((a/b)*dT)-mu*v_f
    dvdt_q = ((a/b)*dT) - Cd*v_q**2
    dvdt_qf = ((a/b)*dT) - Cd*v_qf**2 - f*v_qf
    v = v + dvdt *dt
    v_f = v_f + dvdt_f * dt
    v_q = v_q + dvdt_q * dt
    v_qf = v_qf + dvdt_qf * dt
    vs.append(float(v))
    vs_f.append(float(v_f))
    vs_q.append(float(v_q))
    vs_qf.append(float(v_qf))
    # print("Non-Friction: At time:", i, "then: ", v, " m/s")
    # print("Friction: At time:", i, "then: ", v_f, " m/s")
#print("Steady state velocity (no friction) for length: ", L, "m, is: ", round(v,4), "m/s")
print()
print("Steady state velocity (friction) for length: ", L, "m, is: ", round(v_f,5), "m/s")
print("Steady state velocity (quadratic friction) for length: ", L, "m, is: ", round(v_q,5), "m/s")

v=0
v_f = 0
v_q = 0
v_qf = 0
for i in t:
    a = R*np.log(P0/P1)
    b = 2*(H+L2)
    dvdt = ((a/b)*dT)
    dvdt_f = ((a/b)*dT)-mu*v_f
    dvdt_q = ((a/b)*dT) - Cd*v_q**2
    dvdt_qf = ((a/b)*dT) - Cd*v_qf**2 - f*v_qf
    v = v + dvdt *dt
    v_f = v_f + dvdt_f * dt
    v_q = v_q + dvdt_q * dt
    v_qf = v_qf + dvdt_qf * dt
    vs2.append(float(v))
    vs_f2.append(float(v_f))
    vs_q2.append(float(v_q))
    vs_qf2.append(float(v_qf))
    # print("Non-Friction: At time:", i, "then: ", v, " m/s")
    # print("Friction: At time:", i, "then: ", v_f, " m/s")
print("Steady state velocity (friction) for length: ", L2, "m, is: ", round(v_f,5), "m/s")
print("Steady state velocity (quadratic friction) for length: ", L2, "m, is: ", round(v_q,5), "m/s")



v=0
v_f = 0 
v_q = 0  
v_qf = 0 
for i in t:
    a = R*np.log(P0/P1)
    b = 2*(H+L3)
    dvdt = ((a/b)*dT)
    dvdt_f = ((a/b)*dT)-mu*v_f
    dvdt_q = ((a/b)*dT) - Cd*v_q**2
    dvdt_qf = ((a/b)*dT) - Cd*v_qf**2 - f*v_qf
    v = v + dvdt *dt
    v_f = v_f + dvdt_f * dt
    v_q = v_q + dvdt_q * dt
    v_qf = v_qf + dvdt_qf * dt
    vs3.append(float(v))
    vs_f3.append(float(v_f))
    vs_q3.append(float(v_q))
    vs_qf3.append(float(v_qf))
    # print("Non-Friction: At time:", i, "then: ", v, " m/s")
    # print("Friction: At time:", i, "then: ", v_f, " m/s")   
print("Steady state velocity (friction) for length: ", L3, "m, is: ", round(v_f,5), "m/s")
print("Steady state velocity (quadratic friction) for length: ", L3, "m, is: ", round(v_q,5), "m/s")    
print()

acc = ((a/b)*dT)
acc_f = ((a/b)*dT) - mu*v_f

#plt.plot(t,vs2, label = 'Without friction, L=5,000m')
plt.plot(t,vs, '-.', color = 'k',label = 'Without friction, L=15,000m')

plt.plot(t,vs_f2, '--', color = 'b', label = 'Linear friction, L=5,000m')
plt.plot(t,vs_f, '--',color = 'r',label = 'Linear frictionn, L=15,000m')
plt.plot(t,vs_f3, '--',color = 'g',label = 'Linear friction, L=25,000m')
plt.plot(t,vs_q2, color = 'b', label = 'Quadratic friction, L=5,000m')
plt.plot(t,vs_q, color = 'r', label = 'Quadratic friction, L=15,000m')
plt.plot(t,vs_q3, color = 'g', label = 'Quadratic friction, L=25,000m')
plt.plot(t,vs_qf2, ':',color = 'b', label = 'Quadratic friction with Coriolis, L=5,000m')
plt.plot(t,vs_qf, ':',color = 'r', label = 'Quadratic friction with Coriolis, L=15,000m')
plt.plot(t,vs_qf3, ':',color = 'g', label = 'Quadratic friction with Coriolis, L=25,000m')


plt.ylabel("Velocity (m/s)")
plt.xlabel("Time (s)")
plt.ylim(0,5)
plt.legend(bbox_to_anchor=(1.01, 0.75), borderaxespad=0)
plt.show()
print("Acceleration after an hour without Friction:", acc, "m/s^2")
print ("Acceleration after an hour with Friction:", acc_f, "m/s^2")


#------------------------------------------------------------------
#Advanced question stedy state calculation
Cd = 0.003
Omega = 7.272*10**-5
lat = 51*(np.pi/180)

f = 2*Omega*np.sin(lat)

A = (R*np.log(P0/P1))/(2*(H+L))*dT

v_ss = (-f + np.sqrt(f**2 + 4*Cd*A))/(2*Cd)
#v_ss = round(v_ss,5)
print("Steady state velocity (quadratic & coriolis) = ", v_ss)

#-----------------------------    Task 3    -----------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

gam = 0.0065


def T(x,t,z):
    a = (4*x/20000)+(10+(4/10000))
    b = math.pi/12
    c = 3*math.pi/4
    Temp = (a)*math.sin(b*t - c)+296-gam*z
    return Temp


t = np.arange(0,25,3)
x_land = 10000
x_sea = -10000
z = np.arange (0,10001,1000)



#Colour mapping desion for both plots
cmap_land = plt.get_cmap("Reds")
cmap_sea  = plt.get_cmap("Blues")
colors_land = cmap_land(np.linspace(0.5, 1, len(z)))
colors_sea  = cmap_sea(np.linspace(0.5, 1, len(z)))


#Setting up figure (size, grid, plots, colourbards etc.)
fig = plt.figure(figsize=(9,5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.03, 0.03], wspace=0.05)
ax = fig.add_subplot(gs[0,0])       
cax_land = fig.add_subplot(gs[0,1]) 
cax_sea  = fig.add_subplot(gs[0,2]) 


#Loop that wll run the calculations
for i in z:
    T_land = []
    T_sea = []
    for j in t:
        T_land.append(T(x_land,j,i))
        T_sea.append(T(x_sea,j,i)) 
    idx = list(z).index(i)
    ax.plot(t, T_land, color=colors_land[idx])
    ax.plot(t, T_sea,'--', color=colors_sea[idx])


ax.set_ylabel("Temperature (K)")
ax.set_xlabel("Time (t)")
ax.grid()


#Colour bar creation
norm = Normalize(vmin=z.min(), vmax=z.max())
sm_land = ScalarMappable(cmap=cmap_land, norm=norm)
sm_land.set_array([])
sm_sea = ScalarMappable(cmap=cmap_sea, norm=norm)
sm_sea.set_array([])

#Set up labels and ticks for both sea abd land
cbar_land = plt.colorbar(sm_land, cax=cax_land)
cbar_land.set_ticks([])     
cbar_land.set_label("")
cbar_sea = plt.colorbar(sm_sea, cax=cax_sea)
cbar_sea.set_label("Height z (m)")       
plt.show()

#-----------------------------    Task 4    -----------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#------------Variables--------------

gam = 0.0065 # Km^-1
P_0 = 101325
g = 9.81
R = 287.05

#------------Define Functions--------------
def T(x, t, z):
    a = (4*x/20000)+(10+(4/10000))
    b = math.pi/12
    c = 3*math.pi/4
    Ts = (a)*np.sin(b*t - c) + 296
    Tval = Ts - gam*z
    return Tval

def P(x, t, z):
    a = (4*x/20000)+(10+(4/10000))
    b = math.pi/12
    c = 3*math.pi/4
    Ts = (a)*np.sin(b*t - c) + 296
    Pressure = P_0 * (1 - (gam*z / Ts))**(g/(R*gam))
    return Pressure

def rho(x, t, z):
    Tval = T(x,t,z)
    Pval = P(x,t,z)
    return Pval/ (R*Tval)


#------------Plot parameters--------------
t_hours = [3, 9, 15, 21]          # times requested (hours)
x = np.linspace(-10000, 10000, 200) # horizontal domain (sea - land)
#z = np.linspace(0, 1001, 100)       # vertical (m)
z = np.linspace(0, 10_000, 10)       # vertical (m)
X, Z = np.meshgrid(x, z)            # shapes: (nz, nx)

fig, axs = plt.subplots(4, 2, figsize=(12, 14), constrained_layout=True)


#-----------Running simulation-------------

for i, hour in enumerate(t_hours):

    T_field = T(X, hour, Z)
    P_field = P(X, hour, Z)
    rho_field = rho(X, hour, Z)
    
    rho_levels = np.linspace(np.nanmin(rho_field), np.nanmax(rho_field), 8)

    #-Temperature map & density contours
    axT = axs[i, 0]
    axT.tick_params(axis='both', labelsize=14)
    
    pcmT = axT.contourf(x, z, T_field, levels=10, cmap='plasma')
    csT = axT.contour(
        x, z, rho_field,
        levels=rho_levels,
        colors='black',
        linewidths=1
    )
    axT.clabel(csT, fmt='%.2f', fontsize=12)
    axT.set_ylabel('z (m)', fontsize=14)
    axT.set_title(f'Temperature at {hour:02d}:00', fontsize=14)

    cbarT = fig.colorbar(pcmT, ax=axT, label='Temperature (K)')
    cbarT.set_label('Temperature (K)', fontsize=14)
    cbarT.ax.tick_params(labelsize=14)

    #-Pressure map & density contours
    axP = axs[i, 1]
    axP.tick_params(axis='both', labelsize=14)

    pcmP = axP.contourf(x, z, P_field, levels=10, cmap='Blues')
    csP = axP.contour(
        x, z, rho_field,
        levels=rho_levels,
        colors='black',
        linewidths=1
    )
    
    axP.clabel(csP, fmt='%.2f', fontsize=12)
    axP.set_title(f'Pressure at {hour:02d}:00', fontsize=14)

    cbarP = fig.colorbar(pcmP, ax=axP, label='Pressure (Pa)')
    cbarP.set_label('Pressure (Pa)', fontsize=14)
    cbarP.ax.tick_params(labelsize=14)
    
    if i == 3:
        axT.set_xlabel('x (m)', fontsize=13)
        axP.set_xlabel('x (m)', fontsize=13)
    else:
        axT.set_xticklabels([])
        axP.set_xticklabels([])

plt.suptitle('Temperature (left) and Pressure (right) with density contours', fontsize=18)
plt.show()


#-----------------------------    Task 5    -----------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

dtHours = 3 #hours
tHoursDay = np.arange(0, 24, dtHours) # time array in hours
z = np.linspace(0, 10_000, 100)
# tHoursDay = np.arange(0,24.1,0.1)
print(len(tHoursDay))
accelerationArray = np.zeros((len(z), len(tHoursDay)))
for i in range(len(tHoursDay)):
    for k in range(len(z)):
        height = z[k]
        if k ==0:
            T1bar = np.mean(T(-L, tHoursDay[i], 0))
            T2bar = np.mean(T(L, tHoursDay[i], 0))
        else:
            T1bar = np.mean(T(-L, tHoursDay[i], z[:k]))
            T2bar = np.mean(T(L, tHoursDay[i], z[:k]))
        p1 = np.mean(P(x, tHoursDay[i], z[k]))
        tgtl_acc = R*np.log(P(10_000, tHoursDay[i], 0)/p1)*(T2bar - T1bar)/(2*(height+20_000))
        accelerationArray[k, i] = tgtl_acc

velocity2 = np.zeros(accelerationArray.shape)

for i in range(1, len(tHoursDay)):
    a_mid = 0.5*(accelerationArray[:, i] + accelerationArray[:,i-1])
    dtSimu = 3*60*60
    velocity2[:, i] = (velocity2[:,i-1] + dtSimu*a_mid)/(1.0 + mu*dtSimu)


plt.figure("acceleration")
plt.pcolormesh(tHoursDay, z, accelerationArray, shading="auto", cmap = 'plasma')
plt.xlabel("time")
plt.ylabel("z")
plt.colorbar(label="acceleration")
plt.title("Acceleration without friction")
plt.tight_layout()
plt.show()


plt.figure("Velocity")
plt.pcolormesh(tHoursDay, z, velocity2, shading="auto", cmap = 'plasma')
plt.xlabel("time")
plt.ylabel("z")
plt.colorbar(label="velocity")
plt.title("Velocity with friction")
plt.tight_layout()
plt.show()

plt.figure("Acceleration with friction")
plt.pcolormesh(tHoursDay, z, accelerationArray - mu*velocity2, shading="auto", cmap = 'plasma')
plt.xlabel("time")
plt.ylabel("z")
plt.colorbar(label="Acceleration")
plt.title("Acceleration with friction")
plt.tight_layout()
plt.show()


#-----------------------------    Task 7    -----------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
print("Barotropic case")
#------------Variables--------------

gam = 0.0065 # Km^-1
P_0 = 101325
g = 9.81
R = 287.05

#------------Define Functions--------------
def T(x, t, z):
    # a = (4*x/20000)+(10+(4/10000))
    # b = math.pi/12
    # c = 3*math.pi/4
    Ts = 296
    Tval = Ts - gam*z
    return Tval

def P(x, t, z):
    # a = (4*x/20000)+(10+(4/10000))
    # b = math.pi/12
    # c = 3*math.pi/4
    Ts =  296
    # keep original functional form (vectorized)
    Pressure = P_0 * (1 - (gam*z / Ts))**(g/(R*gam))
    return Pressure

def rho(x, t, z):
    Tval = T(x,t,z)
    Pval = P(x,t,z)
    return Pval/ (R*Tval)


#------------Plot parameters--------------
t_hours = [3, 9, 15, 21]          # times requested (hours)
x = np.linspace(-10000, 10000, 200) # horizontal domain (sea - land)
z = np.linspace(0, 10001, 10)       # vertical (m)
X, Z = np.meshgrid(x, z)            # shapes: (nz, nx)

fig, axs = plt.subplots(4, 2, figsize=(12, 14), constrained_layout=True)


#-----------Running simulation-------------

for i, hour in enumerate(t_hours):

    T_field = T(X, hour, Z)
    P_field = P(X, hour, Z)
    rho_field = rho(X, hour, Z)
    
    rho_levels = np.linspace(np.nanmin(rho_field), np.nanmax(rho_field), 8)

    #-Temperature map & density contours
    axT = axs[i, 0]
    axT.tick_params(axis='both', labelsize=14)
    
    pcmT = axT.contourf(x, z, T_field, levels=10, cmap='plasma')
    csT = axT.contour(
        x, z, rho_field,
        levels=rho_levels,
        colors='black',
        linewidths=1
    )
    axT.clabel(csT, fmt='%.2f', fontsize=12)
    axT.set_ylabel('z (m)', fontsize=14)
    axT.set_title(f'Temperature at {hour:02d}:00', fontsize=14)

    cbarT = fig.colorbar(pcmT, ax=axT, label='Temperature (K)')
    cbarT.set_label('Temperature (K)', fontsize=14)
    cbarT.ax.tick_params(labelsize=14)

    #-Pressure map & density contours
    axP = axs[i, 1]
    axP.tick_params(axis='both', labelsize=14)

    pcmP = axP.contourf(x, z, P_field, levels=10, cmap='Blues')
    csP = axP.contour(
        x, z, rho_field,
        levels=rho_levels,
        colors='black',
        linewidths=1
    )
    
    axP.clabel(csP, fmt='%.2f', fontsize=12)
    axP.set_title(f'Pressure at {hour:02d}:00', fontsize=14)

    cbarP = fig.colorbar(pcmP, ax=axP, label='Pressure (Pa)')
    cbarP.set_label('Pressure (Pa)', fontsize=14)
    cbarP.ax.tick_params(labelsize=14)
    
    if i == 3:
        axT.set_xlabel('x (m)', fontsize=13)
        axP.set_xlabel('x (m)', fontsize=13)
    else:
        axT.set_xticklabels([])
        axP.set_xticklabels([])

plt.suptitle('Temperature (left) and Pressure (right) with density contours', fontsize=18)
plt.show()



# p = np.arange(P0, P1, 50)   # pressure levels (Pa)

# t = np.arange(0, 3601, dt)


# V = np.tile(vs_f, (len(z), 1))

# plt.figure(figsize=(10, 5))

# pcm = plt.contourf(
#     t/3600,     # time in hours
#     z,          # height (m)
#     V,
#     levels=10,
#     cmap='plasma'
# )

# plt.xlabel("Time (hours)")
# plt.ylabel("Pressure (Pa)")
# plt.gca().invert_yaxis()
# plt.title("Figure 3: Tangential velocity evolution (with friction)")
# plt.colorbar(pcm, label="Tangential velocity (m/s)")
# plt.show()


