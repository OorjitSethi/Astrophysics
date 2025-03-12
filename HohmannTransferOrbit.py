import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import TextBox, Button

G = 6.6743e-11
M = 1.989e30

initial_radius_au = 1.0
initial_eccentricity = 0.0

AU_TO_M = 1.496e11
initial_radius_m = initial_radius_au * AU_TO_M

time_steps = 2000
transfer_initiated = False
burn_animation_frames = 40
pre_burn_frames = 500
post_burn_frames = 500
transfer_complete = False

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal')
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_title("Hohmann Transfer Orbit Simulation")
ax.grid(True, linestyle='--', alpha=0.3)

spacecraft_marker, = ax.plot([], [], 'bo', markersize=8, label='Spacecraft')
central_body = ax.plot(0, 0, 'yo', markersize=15, label='Sun')[0]
velocity_vector = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=0.05, color='g')
burn_indicator, = ax.plot([], [], 'ro', markersize=12, alpha=0.5)
velocity_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
delta_v_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
burn_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, color='red')

theta = np.linspace(0, 2*np.pi, 100)
initial_orbit_x = initial_radius_au * np.cos(theta)
initial_orbit_y = initial_radius_au * np.sin(theta)
initial_orbit_line, = ax.plot(initial_orbit_x, initial_orbit_y, 'b--', label='Initial Orbit')

transfer_orbit_line, = ax.plot([], [], 'r--', label='Transfer Orbit')
final_orbit_line, = ax.plot([], [], 'g--', label='Final Orbit')

axbox = plt.axes([0.15, 0.02, 0.1, 0.04])
text_box = TextBox(axbox, 'Target Orbit (AU): ', initial="1.5")

axbutton = plt.axes([0.3, 0.02, 0.1, 0.04])
start_button = Button(axbutton, 'Start Transfer')

def calculate_orbital_velocity(r, a):
    return np.sqrt(G * M * (2/r - 1/a))

def calculate_hohmann_velocities(r1, r2):
    if r1 <= 0 or r2 <= 0:
        raise ValueError("Orbit radius must be greater than 0")
        
    v1 = np.sqrt(G * M / r1)
    
    a_transfer = (r1 + r2) / 2
    v_transfer_periapsis = np.sqrt(G * M * (2/min(r1,r2) - 1/a_transfer))
    v_transfer_apoapsis = np.sqrt(G * M * (2/max(r1,r2) - 1/a_transfer))
    
    v2 = np.sqrt(G * M / r2)
    
    if r1 > r2:
        return v1, v_transfer_apoapsis, v_transfer_periapsis, v2
    else:
        return v1, v_transfer_periapsis, v_transfer_apoapsis, v2

def start_transfer(event):
    global transfer_initiated, transfer_complete
    transfer_initiated = True
    transfer_complete = False

start_button.on_clicked(start_transfer)

def init():
    spacecraft_marker.set_data([], [])
    velocity_vector.set_UVC([], [])
    velocity_text.set_text('')
    delta_v_text.set_text('')
    burn_indicator.set_data([], [])
    burn_text.set_text('')
    return spacecraft_marker, velocity_vector, velocity_text, transfer_orbit_line, final_orbit_line, burn_indicator, burn_text

def update(frame):
    global transfer_complete, initial_radius_au, initial_radius_m, initial_orbit_line, transfer_initiated
    
    if not transfer_initiated:
        angle = 2 * np.pi * frame / time_steps
        x = initial_radius_au * np.cos(angle)
        y = initial_radius_au * np.sin(angle)
        v = calculate_orbital_velocity(initial_radius_m, initial_radius_m) / 1000
        spacecraft_marker.set_data([x], [y])
        velocity_vector.set_offsets([x, y])
        velocity_vector.set_UVC(-y * 0.05, x * 0.05)
        velocity_text.set_text(f'Velocity: {v:.2f} km/s')
        burn_indicator.set_data([], [])
        burn_text.set_text('')
        return spacecraft_marker, velocity_vector, velocity_text, transfer_orbit_line, final_orbit_line, burn_indicator, burn_text

    if not text_box.text.strip():
        return spacecraft_marker, velocity_vector, velocity_text, transfer_orbit_line, final_orbit_line, burn_indicator, burn_text
        
    target_radius_au = float(text_box.text)
    target_radius_m = target_radius_au * AU_TO_M

    v1, v_trans_peri, v_trans_apo, v2 = calculate_hohmann_velocities(initial_radius_m, target_radius_m)
    
    v1 = v1/1000
    v_trans_peri = v_trans_peri/1000
    v_trans_apo = v_trans_apo/1000
    v2 = v2/1000

    transfer_semi_major = (initial_radius_au + target_radius_au) / 2
    transfer_eccentricity = abs(target_radius_au - initial_radius_au) / (target_radius_au + initial_radius_au)
    
    theta_transfer = np.linspace(0, np.pi, 100)
    r_transfer = transfer_semi_major * (1 - transfer_eccentricity**2) / (1 + transfer_eccentricity * np.cos(theta_transfer))
    transfer_orbit_x = r_transfer * np.cos(theta_transfer)
    transfer_orbit_y = r_transfer * np.sin(theta_transfer)
    transfer_orbit_line.set_data(transfer_orbit_x, transfer_orbit_y)

    final_orbit_x = target_radius_au * np.cos(theta)
    final_orbit_y = target_radius_au * np.sin(theta)
    final_orbit_line.set_data(final_orbit_x, final_orbit_y)

    burn_indicator.set_data([], [])
    burn_text.set_text('')

    inward_transfer = initial_radius_au > target_radius_au

    if frame < pre_burn_frames:
        angle = 2 * np.pi * frame / pre_burn_frames
        r = initial_radius_au
        velocity = v1
    elif frame < pre_burn_frames + burn_animation_frames:
        progress = (frame - pre_burn_frames) / burn_animation_frames
        angle = 0
        r = initial_radius_au
        velocity = v1 + (v_trans_peri - v1) * progress
        burn_indicator.set_data([r * np.cos(angle)], [r * np.sin(angle)])
        burn_text.set_text('Initial Burn in Progress')
        delta_v1 = v_trans_peri - v1
        delta_v_text.set_text(f'ΔV1: {delta_v1:+.2f} km/s')
    elif frame < pre_burn_frames + burn_animation_frames + time_steps // 4:
        progress = (frame - (pre_burn_frames + burn_animation_frames)) / (time_steps // 4)
        angle = np.pi * progress
        r = transfer_semi_major * (1 - transfer_eccentricity**2) / (1 + transfer_eccentricity * np.cos(angle))
        velocity = v_trans_peri + (v_trans_apo - v_trans_peri) * progress
        delta_v1 = v_trans_peri - v1
        delta_v_text.set_text(f'ΔV1: {delta_v1:+.2f} km/s')
    elif frame < pre_burn_frames + burn_animation_frames + time_steps // 4 + burn_animation_frames:
        progress = (frame - (pre_burn_frames + burn_animation_frames + time_steps // 4)) / burn_animation_frames
        angle = np.pi
        r = target_radius_au
        velocity = v_trans_apo + (v2 - v_trans_apo) * progress
        burn_indicator.set_data([r * np.cos(angle)], [r * np.sin(angle)])
        burn_text.set_text('Final Burn in Progress')
        delta_v2 = v2 - v_trans_apo
        delta_v_text.set_text(f'ΔV2: {delta_v2:+.2f} km/s')
    else:
        remaining_frames = frame - (pre_burn_frames + burn_animation_frames + time_steps // 4 + burn_animation_frames)
        if remaining_frames < post_burn_frames:
            progress = remaining_frames / post_burn_frames
            angle = np.pi + 2 * np.pi * progress
            r = target_radius_au
            velocity = v2
            delta_v2 = v2 - v_trans_apo
            delta_v_text.set_text(f'ΔV2: {delta_v2:+.2f} km/s')
        else:
            if not transfer_complete:
                initial_radius_au = target_radius_au
                initial_radius_m = target_radius_m
                initial_orbit_x = initial_radius_au * np.cos(theta)
                initial_orbit_y = initial_radius_au * np.sin(theta)
                initial_orbit_line.set_data(initial_orbit_x, initial_orbit_y)
                transfer_orbit_line.set_data([], [])
                final_orbit_line.set_data([], [])
                transfer_complete = True
                transfer_initiated = False
            
            progress = (remaining_frames - post_burn_frames) / time_steps
            angle = 2 * np.pi * progress
            r = target_radius_au
            velocity = v2
            delta_v_text.set_text('')

    x = r * np.cos(angle)
    y = r * np.sin(angle)
    
    spacecraft_marker.set_data([x], [y])
    velocity_vector.set_offsets([x, y])
    velocity_vector.set_UVC(-y * 0.05 * velocity/v1, x * 0.05 * velocity/v1)
    velocity_text.set_text(f'Velocity: {velocity:.2f} km/s')

    return spacecraft_marker, velocity_vector, velocity_text, transfer_orbit_line, final_orbit_line, burn_indicator, burn_text

anim = FuncAnimation(fig, update, frames=time_steps, init_func=init, 
                    blit=True, interval=20)

plt.legend()
plt.show()
