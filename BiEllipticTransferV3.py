import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import TextBox, Button

# Change the speed of the object, which is currently inversely proportional
# in the animation to the actual/calculated speed of the object

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
ax.set_title("Bi-Elliptic Transfer Orbit Simulation")
ax.grid(True, linestyle='--', alpha=0.3)

spacecraft_marker, = ax.plot([], [], 'bo', markersize=8, label='Spacecraft')
central_body = ax.plot(0, 0, 'yo', markersize=15, label='Sun')[0]
velocity_vector = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=0.05, color='g', label='Velocity Vector')
burn_indicator, = ax.plot([], [], 'ro', markersize=12, alpha=0.5, label='Burn Location')
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

def calculate_bielliptic_velocities(r1, r2, r_intermediate):
    if r1 <= 0 or r2 <= 0 or r_intermediate <= 0:
        raise ValueError("Orbit radius must be greater than 0")
        
    v1 = np.sqrt(G * M / r1)  # Initial circular orbit velocity
    
    # First transfer ellipse (r1 to r_intermediate)
    a1_transfer = (r1 + r_intermediate) / 2
    v1_trans_dep = np.sqrt(G * M * (2/r1 - 1/a1_transfer))
    v1_trans_arr = np.sqrt(G * M * (2/r_intermediate - 1/a1_transfer))
    
    # Second transfer ellipse (r_intermediate to r2)
    a2_transfer = (r_intermediate + r2) / 2
    v2_trans_dep = np.sqrt(G * M * (2/r_intermediate - 1/a2_transfer))
    v2_trans_arr = np.sqrt(G * M * (2/r2 - 1/a2_transfer))
    
    v2 = np.sqrt(G * M / r2)  # Final circular orbit velocity
    
    return v1, v1_trans_dep, v1_trans_arr, v2_trans_dep, v2_trans_arr, v2

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
    
    # Use 5x the larger radius as intermediate orbit for bi-elliptic transfer
    intermediate_radius_au = 5 * max(initial_radius_au, target_radius_au)
    intermediate_radius_m = intermediate_radius_au * AU_TO_M

    v1, v1_trans_dep, v1_trans_arr, v2_trans_dep, v2_trans_arr, v2 = calculate_bielliptic_velocities(
        initial_radius_m, target_radius_m, intermediate_radius_m)
    
    # Convert to km/s
    v1 = v1/1000
    v1_trans_dep = v1_trans_dep/1000
    v1_trans_arr = v1_trans_arr/1000
    v2_trans_dep = v2_trans_dep/1000
    v2_trans_arr = v2_trans_arr/1000
    v2 = v2/1000

    # First transfer ellipse
    transfer_semi_major1 = (initial_radius_au + intermediate_radius_au) / 2
    transfer_ecc1 = abs(intermediate_radius_au - initial_radius_au) / (intermediate_radius_au + initial_radius_au)
    
    # Second transfer ellipse
    transfer_semi_major2 = (target_radius_au + intermediate_radius_au) / 2
    transfer_ecc2 = abs(intermediate_radius_au - target_radius_au) / (intermediate_radius_au + target_radius_au)
    
    # Plot complete transfer orbit
    theta_transfer1 = np.linspace(0, np.pi, 100)
    r_transfer1 = transfer_semi_major1 * (1 - transfer_ecc1**2) / (1 + transfer_ecc1 * np.cos(theta_transfer1))
    transfer_x1 = r_transfer1 * np.cos(theta_transfer1)
    transfer_y1 = r_transfer1 * np.sin(theta_transfer1)
    
    theta_transfer2 = np.linspace(np.pi, 2*np.pi, 100)
    r_transfer2 = transfer_semi_major2 * (1 - transfer_ecc2**2) / (1 + transfer_ecc2 * np.cos(theta_transfer2))
    transfer_x2 = r_transfer2 * np.cos(theta_transfer2)
    transfer_y2 = r_transfer2 * np.sin(theta_transfer2)
    
    transfer_x = np.concatenate([transfer_x1, transfer_x2])
    transfer_y = np.concatenate([transfer_y1, transfer_y2])
    transfer_orbit_line.set_data(transfer_x, transfer_y)

    final_orbit_x = target_radius_au * np.cos(theta)
    final_orbit_y = target_radius_au * np.sin(theta)
    final_orbit_line.set_data(final_orbit_x, final_orbit_y)

    burn_indicator.set_data([], [])
    burn_text.set_text('')

    # Animation phases
    total_transfer_time = time_steps // 2  # Half for first transfer, half for second

    if frame < pre_burn_frames:
        # Initial orbit
        angle = 2 * np.pi * frame / pre_burn_frames
        r = initial_radius_au
        velocity = v1
    elif frame < pre_burn_frames + burn_animation_frames:
        # First burn
        progress = (frame - pre_burn_frames) / burn_animation_frames
        angle = 0
        r = initial_radius_au
        velocity = v1 + (v1_trans_dep - v1) * progress
        burn_indicator.set_data([r * np.cos(angle)], [r * np.sin(angle)])
        burn_text.set_text('First Burn in Progress')
        delta_v1 = v1_trans_dep - v1
        delta_v_text.set_text(f'ΔV1: {delta_v1:+.2f} km/s')
    elif frame < pre_burn_frames + burn_animation_frames + total_transfer_time // 2:
        # First transfer arc
        progress = (frame - (pre_burn_frames + burn_animation_frames)) / (total_transfer_time // 2)
        angle = np.pi * progress
        r = transfer_semi_major1 * (1 - transfer_ecc1**2) / (1 + transfer_ecc1 * np.cos(angle))
        velocity = v1_trans_dep + (v1_trans_arr - v1_trans_dep) * progress
    elif frame < pre_burn_frames + burn_animation_frames + total_transfer_time // 2 + burn_animation_frames:
        # Second burn at intermediate point
        progress = (frame - (pre_burn_frames + burn_animation_frames + total_transfer_time // 2)) / burn_animation_frames
        angle = np.pi
        r = intermediate_radius_au
        velocity = v1_trans_arr + (v2_trans_dep - v1_trans_arr) * progress
        burn_indicator.set_data([r * np.cos(angle)], [r * np.sin(angle)])
        burn_text.set_text('Second Burn in Progress')
        delta_v2 = v2_trans_dep - v1_trans_arr
        delta_v_text.set_text(f'ΔV2: {delta_v2:+.2f} km/s')
    elif frame < pre_burn_frames + burn_animation_frames * 2 + total_transfer_time:
        # Second transfer arc
        progress = (frame - (pre_burn_frames + burn_animation_frames * 2 + total_transfer_time // 2)) / (total_transfer_time // 2)
        angle = np.pi + np.pi * progress
        r = transfer_semi_major2 * (1 - transfer_ecc2**2) / (1 + transfer_ecc2 * np.cos(angle))
        velocity = v2_trans_dep + (v2_trans_arr - v2_trans_dep) * progress
    elif frame < pre_burn_frames + burn_animation_frames * 3 + total_transfer_time:
        # Final burn
        progress = (frame - (pre_burn_frames + burn_animation_frames * 2 + total_transfer_time)) / burn_animation_frames
        angle = 2 * np.pi
        r = target_radius_au
        velocity = v2_trans_arr + (v2 - v2_trans_arr) * progress
        burn_indicator.set_data([r * np.cos(angle)], [r * np.sin(angle)])
        burn_text.set_text('Final Burn in Progress')
        delta_v3 = v2 - v2_trans_arr
        delta_v_text.set_text(f'ΔV3: {delta_v3:+.2f} km/s')
    else:
        # Final orbit
        if not transfer_complete:
            initial_radius_au = target_radius_au
            initial_radius_m = target_radius_m
            initial_orbit_x = initial_radius_au * np.cos(theta)
            initial_orbit_y = initial_radius_au * np.sin(theta)
            initial_orbit_line.set_data(initial_orbit_x, initial_orbit_y)
            transfer_complete = True
            transfer_initiated = False
            
        remaining_frames = frame - (pre_burn_frames + burn_animation_frames * 3 + total_transfer_time)
        progress = remaining_frames / time_steps
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

ax.legend(loc='upper right')
plt.show()
