"""
FitzHugh-Nagumo Model Implementation for Half Adder
--------------------------------------------------

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define input signals with different time patterns
def first_input(t):
    """
    First input signal (A) with specific timing pattern.
    Returns 1 or 0 at different time intervals.
    """
    return 1.0*(t>500) - 1.0*(t>1000) + 1.0*(t>1500) - 1.0*(t>2000)

def second_input(t):
    """
    Second input signal (B) with specific timing pattern.
    Returns 1 or 0 at different time intervals.
    """
    return 1.0*(t>1000) - 1.0*(t>2000)

# System parameters
THRESHOLD = 0.1         # Threshold for FHN model
RECOVERY_RATE = 0.1     # Recovery rate parameter
TIME_SCALE = 0.1        # Time scale parameter
SIGMOID_SLOPE = -100    # Slope for sigmoid activation
SIGMOID_OFFSET = 60     # Offset for sigmoid activation

# Network weights
EXCITATORY_WEIGHT = 0.8  # Weight for excitatory connections
INHIBITORY_WEIGHT = 0.45 # Weight for inhibitory connections
CROSS_INHIBITION = 1.5   # Weight for cross-inhibitory connections

# Simulation settings
sim_time = np.arange(0.0, 2000.0, 0.1)  # 2000ms with 0.1ms steps

def fhn_system(state_vector, t):
    """
    FitzHugh-Nagumo system dynamics for the half adder circuit.
    
    Parameters:
        state_vector (array): Current state of all neurons
        t (float): Current time
        
    Returns:
        list: Derivatives of all state variables
    """
    # Unpack state variables
    v1, w1, v2, w2, v_sum, w_sum, v_carry, w_carry = state_vector
    
    # Sigmoid activation function
    def sigmoid(v):
        return 1.0 / (1.0 + np.exp(SIGMOID_SLOPE * v + SIGMOID_OFFSET))
    
    # Input neurons dynamics
    dv1 = -v1*(v1-THRESHOLD)*(v1-1) - w1 + first_input(t)
    dw1 = TIME_SCALE * (v1 - RECOVERY_RATE * w1)
    
    dv2 = -v2*(v2-THRESHOLD)*(v2-1) - w2 + second_input(t)
    dw2 = TIME_SCALE * (v2 - RECOVERY_RATE * w2)
    
    # Sum neuron (XOR gate)
    dv_sum = -v_sum*(v_sum-THRESHOLD)*(v_sum-1) - w_sum + \
             EXCITATORY_WEIGHT * sigmoid(w1) + \
             EXCITATORY_WEIGHT * sigmoid(w2) - \
             CROSS_INHIBITION * sigmoid(w_carry) - \
             2 * INHIBITORY_WEIGHT * sigmoid(w1) * sigmoid(w2)  # Inhibition when both inputs active
    
    dw_sum = TIME_SCALE * (v_sum - RECOVERY_RATE * w_sum)
    
    # Carry neuron (AND gate)
    dv_carry = -v_carry*(v_carry-THRESHOLD)*(v_carry-1) - w_carry + \
               INHIBITORY_WEIGHT * sigmoid(w1) + \
               INHIBITORY_WEIGHT * sigmoid(w2)
    
    dw_carry = TIME_SCALE * (v_carry - RECOVERY_RATE * w_carry)
    
    return [dv1, dw1, dv2, dw2, dv_sum, dw_sum, dv_carry, dw_carry]

# Initial conditions (small nonzero values)
initial_state = [0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0]

# Solve the system
solution = odeint(fhn_system, initial_state, sim_time, rtol=1e-6)

# Visualization
plt.figure(figsize=(10, 10))
plt.subplots_adjust(hspace=1.2)
plt.rcParams["font.size"] = "14"

# Plot each component
axes = []
labels = ["Input A", "Input B", "Sum (XOR)", "Carry (AND)"]
indices = [0, 2, 4, 6]  # Indices for membrane potentials
colors = ['purple', 'purple', 'crimson', 'crimson']

for i in range(4):
    ax = plt.subplot(4, 1, i+1)
    if i == 0:
        ax.set_title("FitzHugh-Nagumo Half Adder Circuit")
    
    ax.plot(sim_time, solution[:, indices[i]], color=colors[i], linewidth=1.5)
    ax.set_ylim(-1, 1.5)
    ax.set_ylabel(labels[i])
    
    # Add threshold line to help visualize logic values
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Only add x-label to the bottom plot
    if i == 3:
        ax.set_xlabel("Time (ms)")
    
    axes.append(ax)

plt.tight_layout()
plt.savefig('fhn_half_adder_oscillations.png', dpi=300, bbox_inches='tight')
plt.show()