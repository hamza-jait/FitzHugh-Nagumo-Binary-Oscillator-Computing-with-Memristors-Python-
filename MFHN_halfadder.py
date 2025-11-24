"""
Memristive FitzHugh-Nagumo Half Adder
------------------------------------------------

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Input signals (exact same as working FHN)
def first_input(t):
    return 1.0*(t>500) - 1.0*(t>1000) + 1.0*(t>1500) - 1.0*(t>2000)

def second_input(t):
    return 1.0*(t>1000) - 1.0*(t>2000)

# FHN parameters (exact same as working FHN)
THRESHOLD = 0.1
RECOVERY_RATE = 0.1
TIME_SCALE = 0.1
SIGMOID_SLOPE = -100
SIGMOID_OFFSET = 60

# Network weights (exact same as working FHN)
EXCITATORY_WEIGHT = 0.8
INHIBITORY_WEIGHT = 0.45
CROSS_INHIBITION = 1.5

# Memristor parameters (simplified)
R_ON = 100.0
R_OFF = 1000.0  # Smaller range for stability
MU_V = 1e-15    # Much slower memristor dynamics

# Simulation settings
sim_time = np.arange(0.0, 2000.0, 0.1)

def simple_memristor_conductance(x):
    """Simple memristor conductance model"""
    R = R_ON * x + R_OFF * (1 - x)
    return 1.0 / R

def simple_memristor_dynamics(x, v):
    """Very simple memristor dynamics"""
    dx_dt = MU_V * v
    # Keep in bounds
    return max(-x/10, min((1-x)/10, dx_dt))

def memristive_fhn_system(state_vector, t):
    """
    Simplified system: start with working FHN, add small memristive modulation
    """
    # Unpack: [v1, w1, v2, w2, v_sum, w_sum, v_carry, w_carry, x1_sum, x2_sum, x1_carry, x2_carry]
    v1, w1, v2, w2, v_sum, w_sum, v_carry, w_carry, x1_sum, x2_sum, x1_carry, x2_carry = state_vector
    
    # Sigmoid (exact same as working FHN)
    def sigmoid(v):
        return 1.0 / (1.0 + np.exp(SIGMOID_SLOPE * v + SIGMOID_OFFSET))
    
    # Input neurons (exact same as working FHN)
    dv1 = -v1*(v1-THRESHOLD)*(v1-1) - w1 + first_input(t)
    dw1 = TIME_SCALE * (v1 - RECOVERY_RATE * w1)
    
    dv2 = -v2*(v2-THRESHOLD)*(v2-1) - w2 + second_input(t)
    dw2 = TIME_SCALE * (v2 - RECOVERY_RATE * w2)
    
    # Sum neuron (start with exact working FHN logic)
    sum_base = (-v_sum*(v_sum-THRESHOLD)*(v_sum-1) - w_sum + 
                EXCITATORY_WEIGHT * sigmoid(w1) + 
                EXCITATORY_WEIGHT * sigmoid(w2) - 
                CROSS_INHIBITION * sigmoid(w_carry) - 
                2 * INHIBITORY_WEIGHT * sigmoid(w1) * sigmoid(w2))
    
    # Add tiny memristive modulation (< 1% effect)
    mem_mod_sum = 0.01 * (simple_memristor_conductance(x1_sum) + simple_memristor_conductance(x2_sum) - 2) * 0.5
    dv_sum = sum_base + mem_mod_sum
    
    dw_sum = TIME_SCALE * (v_sum - RECOVERY_RATE * w_sum)
    
    # Carry neuron (start with exact working FHN logic)
    carry_base = (-v_carry*(v_carry-THRESHOLD)*(v_carry-1) - w_carry + 
                  INHIBITORY_WEIGHT * sigmoid(w1) + 
                  INHIBITORY_WEIGHT * sigmoid(w2))
    
    # Add tiny memristive modulation
    mem_mod_carry = 0.01 * (simple_memristor_conductance(x1_carry) + simple_memristor_conductance(x2_carry) - 2) * 0.5
    dv_carry = carry_base + mem_mod_carry
    
    dw_carry = TIME_SCALE * (v_carry - RECOVERY_RATE * w_carry)
    
    # Simple memristor dynamics
    dx1_sum = simple_memristor_dynamics(x1_sum, v1 - v_sum)
    dx2_sum = simple_memristor_dynamics(x2_sum, v2 - v_sum)
    dx1_carry = simple_memristor_dynamics(x1_carry, v1 - v_carry)
    dx2_carry = simple_memristor_dynamics(x2_carry, v2 - v_carry)
    
    return [dv1, dw1, dv2, dw2, dv_sum, dw_sum, dv_carry, dw_carry,
            dx1_sum, dx2_sum, dx1_carry, dx2_carry]

# Initial conditions (exact same as working FHN plus memristors at 0.5)
initial_state = [0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5]

# Solve
solution = odeint(memristive_fhn_system, initial_state, sim_time, rtol=1e-6)

# Visualization (exact same as working FHN)
plt.figure(figsize=(10, 10))
plt.subplots_adjust(hspace=1.2)
plt.rcParams["font.size"] = "14"

labels = ["Input A", "Input B", "Sum (XOR)", "Carry (AND)"]
indices = [0, 2, 4, 6]
colors = ['purple', 'purple', 'crimson', 'crimson']

for i in range(4):
    ax = plt.subplot(4, 1, i+1)
    if i == 0:
        ax.set_title("Memristive FitzHugh-Nagumo Half Adder Circuit")
    
    ax.plot(sim_time, solution[:, indices[i]], color=colors[i], linewidth=1.5)
    ax.set_ylim(-1, 1.5)
    ax.set_ylabel(labels[i])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    if i == 3:
        ax.set_xlabel("Time (ms)")

plt.tight_layout()
plt.savefig('memristive_fhn_halfadder_simple.png', dpi=300, bbox_inches='tight')
plt.show()

# Results
print("="*60)
print("SIMPLIFIED MEMRISTIVE FHN HALF-ADDER RESULTS")
print("="*60)

# Debug values
print(f"\nSample values at key times:")
for t_sample in [500, 750, 1250, 1750]:
    idx = int(t_sample * 10)
    print(f"t={t_sample}ms: V1={solution[idx,0]:.3f}, V2={solution[idx,2]:.3f}, Sum={solution[idx,4]:.3f}, Carry={solution[idx,6]:.3f}")

# Truth table
time_periods = [
    (400, 600, "A=0, B=0"),
    (900, 1100, "A=0, B=1"),
    (1400, 1600, "A=1, B=0"),
    (1900, 2000, "A=1, B=1")
]

print("\nTruth Table:")
for start, end, condition in time_periods:
    start_idx = int(start * 10)
    end_idx = int(end * 10)
    
    avg_sum = np.mean(solution[start_idx:end_idx, 4])
    avg_carry = np.mean(solution[start_idx:end_idx, 6])
    
    sum_logic = "HIGH" if avg_sum > 0.5 else "LOW "
    carry_logic = "HIGH" if avg_carry > 0.5 else "LOW "
    
    print(f"{condition}: Sum={sum_logic} ({avg_sum:.3f}), Carry={carry_logic} ({avg_carry:.3f})")

print("\nSimulation completed!")
