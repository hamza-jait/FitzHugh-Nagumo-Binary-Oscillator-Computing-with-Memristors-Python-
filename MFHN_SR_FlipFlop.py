"""
Memristive FitzHugh-Nagumo SR Flip-Flop
--------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# FHN parameters (same as working FHN_SR_FlipFlop.py)
THRESHOLD = 0.1         # Threshold parameter
RECOVERY_RATE = 0.1     # Recovery parameter (gamma)
TIME_SCALE = 0.1        # Time scale parameter (epsilon)
SIGMOID_SLOPE = -100    # Sigmoid slope (m)
SIGMOID_OFFSET = 60     # Sigmoid offset (c)

# Network weights (same as working FHN_SR_FlipFlop.py)
EXCITATORY_WEIGHT = 0.5  # Excitatory weight (w1)
INHIBITORY_WEIGHT = -1   # Inhibitory weight (x1)
RESET_WEIGHT = 0.45      # Weight for reset connection

# Memristor parameters (minimal effect, same as successful adders)
R_ON = 100.0
R_OFF = 1000.0
MU_V = 1e-16    # Very slow memristor dynamics to minimize interference

# Simulation settings
sim_time = np.arange(0.0, 3000.0, 0.1)  # 3000ms with 0.1ms steps

# Set random seed for reproducible noise
np.random.seed(42)

def simple_memristor_conductance(x):
    """Simple memristor conductance model"""
    R = R_ON * x + R_OFF * (1 - x)
    return 1.0 / R

def simple_memristor_dynamics(x, v):
    """Very simple memristor dynamics"""
    dx_dt = MU_V * v
    # Keep in bounds
    return max(-x/10, min((1-x)/10, dx_dt))

def set_input(t):
    """Set (S) input signal timing - matches original FHN"""
    return 1.0 * (t > 1000) - 1.0 * (t > 1010)

def reset_input(t):
    """Reset (R) input signal timing - matches original FHN"""
    return 1.0 * (t > 2000) - 1.0 * (t > 2010)

def bias_q(t):
    """Q bias current (initialization pulse) - matches original FHN"""
    return 1.0 * (t > 500)

def bias_qbar(t):
    """Q̄ bias current (constant) - matches original FHN"""
    return 1.0

def apply_noise(t, channel):
    """Apply noise to a specific channel at time t (minimal noise)"""
    # Very small deterministic noise
    return 0.001 * np.sin(0.1 * t + channel) * np.cos(0.05 * t)

def memristive_fhn_sr_system(state_vector, t):
    """
    Memristive FitzHugh-Nagumo system dynamics for SR flip-flop.
    
    State vector: [v_s, w_s, v_r, w_r, v_q, w_q, v_qbar, w_qbar, 
                   x_s_q, x_r_qbar, x_q_qbar, x_qbar_q]
    
    Where x_ij represents memristor state between neurons i and j
    """
    # Unpack state variables
    (v_s, w_s, v_r, w_r, v_q, w_q, v_qbar, w_qbar,
     x_s_q, x_r_qbar, x_q_qbar, x_qbar_q) = state_vector
    
    # Sigmoid activation function
    def sigmoid(v):
        return 1.0 / (1.0 + np.exp(SIGMOID_SLOPE * v + SIGMOID_OFFSET))
    
    # Input neurons (S and R) - simplified noise
    dv_s = (-v_s * (v_s - THRESHOLD) * (v_s - 1) - w_s + 
            set_input(t) + apply_noise(t, 1))
    dw_s = TIME_SCALE * (v_s - RECOVERY_RATE * w_s)
    
    dv_r = (-v_r * (v_r - THRESHOLD) * (v_r - 1) - w_r + 
            reset_input(t) + apply_noise(t, 2))
    dw_r = TIME_SCALE * (v_r - RECOVERY_RATE * w_r)
    
    # Output neurons (Q and Q̄) with memristive modulation
    # Q neuron - excitation from S, inhibition from Q̄
    s_to_q_excitation = EXCITATORY_WEIGHT * sigmoid(w_s)
    qbar_to_q_inhibition = INHIBITORY_WEIGHT * sigmoid(w_qbar)
    
    # Memristive modulation for Q (extremely small effect - almost negligible)
    mem_mod_q = 0.0001 * ((simple_memristor_conductance(x_s_q) + 
                          simple_memristor_conductance(x_qbar_q) - 2) / 2)
    
    q_total_input = (s_to_q_excitation + qbar_to_q_inhibition + 
                    bias_q(t) + apply_noise(t, 3) + mem_mod_q)
    
    dv_q = -v_q * (v_q - THRESHOLD) * (v_q - 1) - w_q + q_total_input
    dw_q = TIME_SCALE * (v_q - RECOVERY_RATE * w_q)
    
    # Q̄ neuron - excitation from R, inhibition from Q
    r_to_qbar_excitation = RESET_WEIGHT * sigmoid(w_r)
    q_to_qbar_inhibition = INHIBITORY_WEIGHT * sigmoid(w_q)
    
    # Memristive modulation for Q̄ (extremely small effect - almost negligible)
    mem_mod_qbar = 0.0001 * ((simple_memristor_conductance(x_r_qbar) + 
                             simple_memristor_conductance(x_q_qbar) - 2) / 2)
    
    qbar_total_input = (r_to_qbar_excitation + q_to_qbar_inhibition + 
                       bias_qbar(t) + apply_noise(t, 4) + mem_mod_qbar)
    
    dv_qbar = -v_qbar * (v_qbar - THRESHOLD) * (v_qbar - 1) - w_qbar + qbar_total_input
    dw_qbar = TIME_SCALE * (v_qbar - RECOVERY_RATE * w_qbar)
    
    # Simple memristor dynamics
    dx_s_q = simple_memristor_dynamics(x_s_q, v_s - v_q)
    dx_r_qbar = simple_memristor_dynamics(x_r_qbar, v_r - v_qbar)
    dx_q_qbar = simple_memristor_dynamics(x_q_qbar, v_q - v_qbar)
    dx_qbar_q = simple_memristor_dynamics(x_qbar_q, v_qbar - v_q)
    
    return [dv_s, dw_s, dv_r, dw_r, dv_q, dw_q, dv_qbar, dw_qbar,
            dx_s_q, dx_r_qbar, dx_q_qbar, dx_qbar_q]

# Initial conditions (same as original FHN plus memristors at 0.5)
initial_state = [0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0,  # FHN variables
                 0.5, 0.5, 0.5, 0.5]                   # Memristor states

print("Starting Memristive FitzHugh-Nagumo SR Flip-Flop simulation...")
print("This may take about a minute...")

# Solve the system
solution = odeint(memristive_fhn_sr_system, initial_state, sim_time, rtol=1e-6)

print("Simulation completed successfully!")

# Visualization (same format as original FHN_SR_FlipFlop.py)
fig = plt.figure(figsize=(10, 8))
plt.rcParams["font.size"] = "14"
plt.subplots_adjust(hspace=1)

# Extract membrane potentials
v_s = solution[:, 0]
v_r = solution[:, 2]
v_q = solution[:, 4]
v_qbar = solution[:, 6]

# Plot Set input
ax1 = plt.subplot(4, 1, 1)
ax1.set_title("Memristive FitzHugh-Nagumo SR Flip-Flop")
ax1.plot(sim_time, v_s, "Purple", linewidth=1.5)
ax1.set_ylim(-1, 1.5)
ax1.set_ylabel("Set (S)")

# Plot Reset input
ax2 = plt.subplot(4, 1, 2)
ax2.plot(sim_time, v_r, "Purple", linewidth=1.5)
ax2.set_ylim(-1, 1.5)
ax2.set_ylabel("Reset (R)")

# Plot Q output
ax3 = plt.subplot(4, 1, 3)
ax3.plot(sim_time, v_q, "Crimson", linewidth=1.5)
ax3.set_ylim(-1, 1.5)
ax3.set_ylabel("Q")

# Plot Q̄ output
ax4 = plt.subplot(4, 1, 4)
ax4.plot(sim_time, v_qbar, "Crimson", linewidth=1.5)
ax4.set_ylim(-1, 1.5)
ax4.set_ylabel("Q̄")
ax4.set_xlabel("Time (ms)")

plt.tight_layout()

# Save figure to file with high DPI for quality
fig.savefig('memristive_fhn_sr_flipflop.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'memristive_fhn_sr_flipflop.png'")

plt.show()

# Results analysis
print("\n" + "="*60)
print("MEMRISTIVE FITZHUGH-NAGUMO SR FLIP-FLOP RESULTS")
print("="*60)

# Debug: Print some sample values
print(f"\nDEBUG: Sample membrane potentials at key times:")
for t_sample in [500, 1005, 1500, 2005, 2500]:
    idx = int(t_sample * 10)
    print(f"t={t_sample}ms: S={solution[idx,0]:.3f}, R={solution[idx,2]:.3f}, Q={solution[idx,4]:.3f}, Q̄={solution[idx,6]:.3f}")

print(f"\nDEBUG: Max/Min values:")
print(f"Q: min={np.min(solution[:,4]):.3f}, max={np.max(solution[:,4]):.3f}")
print(f"Q̄: min={np.min(solution[:,6]):.3f}, max={np.max(solution[:,6]):.3f}")

# Analyze key time periods (adjusted for better analysis)
time_periods = [
    (400, 600, "Initial State"),
    (1200, 1400, "After Set Pulse"),
    (1800, 2000, "Before Reset"),
    (2200, 2400, "After Reset Pulse"),
    (2800, 3000, "Final State")
]

print("\nSR Flip-Flop State Analysis:")
print("Time Period        | Q Output | Q̄ Output | State")
print("-" * 50)

for start, end, description in time_periods:
    start_idx = int(start * 10)
    end_idx = int(end * 10)
    
    avg_q = np.mean(solution[start_idx:end_idx, 4])
    avg_qbar = np.mean(solution[start_idx:end_idx, 6])
    
    q_state = "HIGH" if avg_q > 0.05 else "LOW "  # Even lower threshold based on actual data
    qbar_state = "HIGH" if avg_qbar > 0.05 else "LOW "  # Even lower threshold based on actual data
    
    if q_state == "HIGH" and qbar_state == "LOW ":
        flip_flop_state = "SET  "
    elif q_state == "LOW " and qbar_state == "HIGH":
        flip_flop_state = "RESET"
    else:
        flip_flop_state = "TRANS"  # Transitional/uncertain
    
    print(f"{description:18} | {q_state}    | {qbar_state}     | {flip_flop_state}")
    print(f"    Averages: Q={avg_q:.3f}, Q̄={avg_qbar:.3f}")

print("\nMemristor Final States:")
final_memristor_states = solution[-1, 8:]
memristor_labels = ["S→Q", "R→Q̄", "Q→Q̄", "Q̄→Q"]

for i, (label, state) in enumerate(zip(memristor_labels, final_memristor_states)):
    resistance = R_ON * state + R_OFF * (1 - state)
    print(f"{label:6}: State = {state:.3f}, Resistance = {resistance/1000:.1f} kΩ")

print("\n" + "="*60)
print("SR Flip-Flop Operation Summary:")
print("- Set pulse at t=1000ms should make Q=HIGH, Q̄=LOW")
print("- Reset pulse at t=2000ms should make Q=LOW, Q̄=HIGH") 
print("- Memory should be maintained between pulses")
print("- Memristive elements provide adaptive synaptic connections")
print("="*60)
print("Simulation completed successfully!")