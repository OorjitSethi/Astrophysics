import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 6.6743e-11  
M = 1.989e30    

# Hardcoded for now
eccentricity = 0.2056 
semi_major_axis_au = 0.387098  


focus_distance = semi_major_axis_au * eccentricity

semi_major_axis_m = semi_major_axis_au * 1.496e11

period = 2 * np.pi * np.sqrt(semi_major_axis_m**3 / (G * M))  

unit_velocity = np.sqrt(G * M / semi_major_axis_m) / 1000  

semi_minor_axis = semi_major_axis_au * np.sqrt(1 - eccentricity**2)

time_steps = 1000

padding_factor = 1.2
plot_limit = max(semi_major_axis_au, semi_minor_axis) * padding_factor

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal')
ax.set_xlim(-plot_limit, plot_limit)
ax.set_ylim(-plot_limit, plot_limit)
ax.set_title("Kepler's First Law: Elliptical Orbit")
ax.set_xticks([])  
ax.set_yticks([])  

theta = np.linspace(0, 2 * np.pi, 1000)
radius = semi_major_axis_au * (1 - eccentricity**2) / (1 + eccentricity * np.cos(theta))
orbit_x = radius * np.cos(theta) - focus_distance
orbit_y = radius * np.sin(theta)
ax.plot(orbit_x, orbit_y, 'b--', label='Orbit Path')
ax.plot(-focus_distance, 0, 'yo', markersize=10, label='Central Object')

planet_marker, = ax.plot([], [], 'ro', label='Planet')
velocity_vector = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=0.05, color='g')
velocity_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top')
ax.legend()

AU_in_km = 1.496e8  
omega = 2 * np.pi / period  

E = np.linspace(0, 2 * np.pi, time_steps)
nu = 2 * np.arctan(np.sqrt((1 + eccentricity)/(1 - eccentricity)) * np.tan(E/2))

def init():
    planet_marker.set_data([], [])
    velocity_vector.set_UVC([], [])
    velocity_text.set_text('')
    return planet_marker, velocity_vector, velocity_text

def update(frame):
    true_anomaly = nu[frame]
    radius = semi_major_axis_au * (1 - eccentricity**2) / (1 + eccentricity * np.cos(true_anomaly))
    planet_x = radius * np.cos(true_anomaly) - focus_distance
    planet_y = radius * np.sin(true_anomaly)
    planet_marker.set_data([planet_x], [planet_y])

    h = np.sqrt(G * M * semi_major_axis_m * (1 - eccentricity**2))  
    
    radius_m = radius * 1.496e11
    
    v_r = h * eccentricity * np.sin(true_anomaly) / radius_m  
    v_theta = h / radius_m  
    
    vx = (v_r * np.cos(true_anomaly) - v_theta * np.sin(true_anomaly))
    vy = (v_r * np.sin(true_anomaly) + v_theta * np.cos(true_anomaly))
    
    vx /= 1000
    vy /= 1000
    
    velocity_magnitude = np.sqrt(vx**2 + vy**2)

    velocity_scale = plot_limit * 0.05  
    velocity_vector.set_offsets([planet_x, planet_y])
    velocity_vector.set_UVC(vx * velocity_scale / velocity_magnitude, 
                          vy * velocity_scale / velocity_magnitude)

    velocity_text.set_text(f'Velocity: {velocity_magnitude:.2f} km/s')

    return planet_marker, velocity_vector, velocity_text

anim = FuncAnimation(fig, update, frames=np.arange(0, time_steps), init_func=init, blit=True, interval=20)
plt.show()
