import numpy as np
import matplotlib.pyplot as plt


G = 6.67430e-11
c = 299792458
mass_sun = 1.989e30

def calculate_swarzschild_radius(mass):
    return(2*G*mass) / (c**2)

def calc_time_dilation(r, rs):
    with np.errstate(invalid= 'ignore', divide = 'ignore'):
        factor = np.sqrt(1 - (rs/r))
    return factor

bh_mass = 10 *mass_sun
rs_meters = calculate_swarzschild_radius(bh_mass)
rs_km = rs_meters /1000

print("black hole is 10 solar masses")
print(f"Swarzschild radius(EVENT HORIZION): {rs_km:.2f} km\n")

r_distances = np.linspace(1.01 * rs_meters, 5 * rs_meters, 500)
dilationfactor = calc_time_dilation(r_distances, rs_meters)

plt.figure(figsize=(9,5))
plt.plot(r_distances / 1000, dilationfactor, color = 'purple', lw=2.5, label = 'Time Flow Rate')

plt.axvline(x = rs_km, color= 'black', linestyle = '--', alpha = .8, label = 'Event Horizon ($R_s$)')
plt.fill_betweenx([0, 1.05], 0, rs_km, color='black', alpha=0.15, label='Inside Horizon')


plt.title("GRAVITATIONAL TIME DILATION NEAR A BLACK HOLE", fontsize = 14, pad=15)
plt.xlabel("RADIAL DISTANCE FROM CENTER ($r$) in km", fontsize = 14)
plt.ylabel("TIME FLOW RATE RATIO ($t_{local}/t_{distant}$)")
plt.grid(True, linestyle = ":", alpha = .6)
plt.ylim(0,1.05)
plt.xlim(0,max(r_distances)/1000)
plt.legend(loc = 'lower right', frameon =True)

plt.tight_layout()
plt.show()