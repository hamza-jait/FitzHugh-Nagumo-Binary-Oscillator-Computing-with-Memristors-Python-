"""
Simplified Memristive FitzHugh-Nagumo Full Adder - Oscillations Plot
------------------------------------------------------------------

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Input signals for full adder (matching original FHN_fulladder.py patterns)
def input_a(t):
    """Input A signal - matches first_input from FHN_fulladder.py"""
    return 1.0*(t>500) - 1.0*(t>1000) + 1.0*(t>2000) - 1.0*(t>2500) + 1.0*(t>3000)

def input_b(t):
    """Input B signal - matches second_input from FHN_fulladder.py"""
    return 1.0*(t>1000) - 1.0*(t>1500) + 1.0*(t>2500)

def input_cin(t):
    """Carry-in signal - matches third_input from FHN_fulladder.py"""
    return 1.0*(t>1500) - 1.0*(t>3000) + 1.0*(t>3500)

# FHN parameters (same as working half-adder)
THRESHOLD = 0.1
RECOVERY_RATE = 0.1
TIME_SCALE = 0.1
SIGMOID_SLOPE = -100
SIGMOID_OFFSET = 60

# Network weights for full adder (matching original FHN_fulladder.py)
EXCITATORY_WEIGHT = 0.8      # Same as original
INHIBITORY_WEIGHT = 0.45     # Same as original  
CROSS_INHIBITION = 1.5       # Same as original

# Memristor parameters (very slow dynamics to minimize interference)
R_ON = 100.0
R_OFF = 1000.0
MU_V = 1e-16    # Much slower memristor dynamics

# Simulation settings (extended time for full adder)
sim_time = np.arange(0.0, 4000.0, 0.1)  # 4000ms to show all combinations

def simple_memristor_conductance(x):
    """Simple memristor conductance model"""
    R = R_ON * x + R_OFF * (1 - x)
    return 1.0 / R

def simple_memristor_dynamics(x, v):
    """Very simple memristor dynamics"""
    dx_dt = MU_V * v
    # Keep in bounds
    return max(-x/10, min((1-x)/10, dx_dt))

def memristive_fhn_fulladder_system(state_vector, t):
    """
    Simplified memristive FHN system for full adder
    State: [vA, wA, vB, wB, vCin, wCin, vSum, wSum, vCarry, wCarry, 
            xA_sum, xB_sum, xCin_sum, xA_carry, xB_carry, xCin_carry]
    """
    # Unpack state variables
    (vA, wA, vB, wB, vCin, wCin, vSum, wSum, vCarry, wCarry,
     xA_sum, xB_sum, xCin_sum, xA_carry, xB_carry, xCin_carry) = state_vector
    
    # Sigmoid activation function
    def sigmoid(v):
        return 1.0 / (1.0 + np.exp(SIGMOID_SLOPE * v + SIGMOID_OFFSET))
    
    # Input neurons (same structure as half-adder)
    dvA = -vA*(vA-THRESHOLD)*(vA-1) - wA + input_a(t)
    dwA = TIME_SCALE * (vA - RECOVERY_RATE * wA)
    
    dvB = -vB*(vB-THRESHOLD)*(vB-1) - wB + input_b(t)
    dwB = TIME_SCALE * (vB - RECOVERY_RATE * wB)
    
    dvCin = -vCin*(vCin-THRESHOLD)*(vCin-1) - wCin + input_cin(t)
    dwCin = TIME_SCALE * (vCin - RECOVERY_RATE * wCin)
    
    # Sum neuron (matches original FHN_fulladder.py logic)
    # Simple excitation from all three inputs with cross-inhibition from carry
    sum_excitation = (EXCITATORY_WEIGHT * sigmoid(wA) + 
                     EXCITATORY_WEIGHT * sigmoid(wB) + 
                     EXCITATORY_WEIGHT * sigmoid(wCin))
    
    carry_to_sum_inhibition = CROSS_INHIBITION * sigmoid(wCarry)
    
    # Memristive modulation for sum (extremely small effect - almost negligible)
    mem_mod_sum = 0.001 * ((simple_memristor_conductance(xA_sum) + 
                           simple_memristor_conductance(xB_sum) + 
                           simple_memristor_conductance(xCin_sum) - 3) / 3)
    
    sum_total_input = sum_excitation - carry_to_sum_inhibition + mem_mod_sum
    
    dvSum = -vSum*(vSum-THRESHOLD)*(vSum-1) - wSum + sum_total_input
    dwSum = TIME_SCALE * (vSum - RECOVERY_RATE * wSum)
    
    # Carry neuron (matches original FHN_fulladder.py logic)
    # Simple excitation from all three inputs (carry high when at least two inputs are high)
    carry_excitation = (INHIBITORY_WEIGHT * sigmoid(wA) + 
                       INHIBITORY_WEIGHT * sigmoid(wB) + 
                       INHIBITORY_WEIGHT * sigmoid(wCin))
    
    # Memristive modulation for carry (extremely small effect - almost negligible)
    mem_mod_carry = 0.001 * ((simple_memristor_conductance(xA_carry) + 
                             simple_memristor_conductance(xB_carry) + 
                             simple_memristor_conductance(xCin_carry) - 3) / 3)
    
    carry_total_input = carry_excitation + mem_mod_carry
    
    dvCarry = -vCarry*(vCarry-THRESHOLD)*(vCarry-1) - wCarry + carry_total_input
    dwCarry = TIME_SCALE * (vCarry - RECOVERY_RATE * wCarry)
    
    # Simple memristor dynamics
    dxA_sum = simple_memristor_dynamics(xA_sum, vA - vSum)
    dxB_sum = simple_memristor_dynamics(xB_sum, vB - vSum)
    dxCin_sum = simple_memristor_dynamics(xCin_sum, vCin - vSum)
    dxA_carry = simple_memristor_dynamics(xA_carry, vA - vCarry)
    dxB_carry = simple_memristor_dynamics(xB_carry, vB - vCarry)
    dxCin_carry = simple_memristor_dynamics(xCin_carry, vCin - vCarry)
    
    return [dvA, dwA, dvB, dwB, dvCin, dwCin, dvSum, dwSum, dvCarry, dwCarry,
            dxA_sum, dxB_sum, dxCin_sum, dxA_carry, dxB_carry, dxCin_carry]

# Initial conditions (same structure as half-adder)
initial_state = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0,  # FHN variables
                 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]                    # Memristor states

# Solve the system
print("Running Memristive FitzHugh-Nagumo Full Adder simulation...")
solution = odeint(memristive_fhn_fulladder_system, initial_state, sim_time, rtol=1e-6)

# Visualization (same format as half-adder)
plt.figure(figsize=(12, 12))
plt.subplots_adjust(hspace=1.2)
plt.rcParams["font.size"] = "14"

labels = ["Input A", "Input B", "Input Cin", "Sum (XOR)", "Carry (Majority)"]
indices = [0, 2, 4, 6, 8]  # Indices for membrane potentials
colors = ['purple', 'blue', 'green', 'crimson', 'orange']

for i in range(5):
    ax = plt.subplot(5, 1, i+1)
    if i == 0:
        ax.set_title("Memristive FitzHugh-Nagumo Full Adder Circuit")
    
    ax.plot(sim_time, solution[:, indices[i]], color=colors[i], linewidth=1.5)
    ax.set_ylim(-1, 1.5)
    ax.set_ylabel(labels[i])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    if i == 4:  # Only add x-label to bottom plot
        ax.set_xlabel("Time (ms)")

plt.tight_layout()
plt.savefig('memristive_fhn_fulladder_oscillations.png', dpi=300, bbox_inches='tight')
plt.show()

# Results analysis
print("="*80)
print("MEMRISTIVE FITZHUGH-NAGUMO FULL-ADDER SIMULATION RESULTS")
print("="*80)

# Debug values at key time points
print(f"\nSample values at key times:")
sample_times = [500, 750, 1250, 1750, 2250, 2750, 3250, 3750]
for t_sample in sample_times:
    idx = int(t_sample * 10)
    print(f"t={t_sample:4.0f}ms: A={solution[idx,0]:.3f}, B={solution[idx,2]:.3f}, Cin={solution[idx,4]:.3f}, Sum={solution[idx,6]:.3f}, Carry={solution[idx,8]:.3f}")

# Truth table verification for all 8 combinations
print("\nFull Adder Truth Table Verification:")
print("A | B | Cin | Sum (A⊕B⊕Cin) | Carry (AB+ACin+BCin) | Expected")
print("-" * 70)

# Define time periods for each input combination (matching original FHN patterns)
time_periods = [
    (400, 600, 0, 0, 0),    # A=0, B=0, Cin=0
    (800, 1200, 1, 0, 0),   # A=1, B=0, Cin=0  
    (1200, 1700, 1, 1, 0),  # A=1, B=1, Cin=0
    (1700, 2200, 1, 1, 1),  # A=1, B=1, Cin=1
    (2200, 2700, 0, 1, 1),  # A=0, B=1, Cin=1
    (2700, 3200, 0, 0, 1),  # A=0, B=0, Cin=1
    (3200, 3700, 1, 0, 1),  # A=1, B=0, Cin=1
    (3700, 4000, 1, 0, 0),  # A=1, B=0, Cin=0 (end state)
]

for start, end, A, B, Cin in time_periods:
    start_idx = int(start * 10)
    end_idx = int(end * 10)
    
    avg_sum = np.mean(solution[start_idx:end_idx, 6])
    avg_carry = np.mean(solution[start_idx:end_idx, 8])
    
    # Also check what the actual input states are during this period
    avg_vA = np.mean(solution[start_idx:end_idx, 0])
    avg_vB = np.mean(solution[start_idx:end_idx, 2])
    avg_vCin = np.mean(solution[start_idx:end_idx, 4])
    
    sum_logic = "1" if avg_sum > 0.5 else "0"
    carry_logic = "1" if avg_carry > 0.5 else "0"
    
    # Calculate expected outputs
    expected_sum = A ^ B ^ Cin  # 3-way XOR
    expected_carry = (A & B) | (A & Cin) | (B & Cin)  # Majority
    
    # Check correctness
    sum_correct = "✓" if int(sum_logic) == expected_sum else "✗"
    carry_correct = "✓" if int(carry_logic) == expected_carry else "✗"
    
    print(f"{A} | {B} | {Cin:3} | {sum_logic:11} | {carry_logic:16} | Sum={expected_sum}, Carry={expected_carry} {sum_correct}{carry_correct}")
    print(f"    Actual inputs: A={avg_vA:.3f}, B={avg_vB:.3f}, Cin={avg_vCin:.3f} | Outputs: Sum={avg_sum:.3f}, Carry={avg_carry:.3f}")

print("\nMemristor Final States:")
final_memristor_states = solution[-1, 10:]
memristor_labels = ["A→Sum", "B→Sum", "Cin→Sum", "A→Carry", "B→Carry", "Cin→Carry"]

for i, (label, state) in enumerate(zip(memristor_labels, final_memristor_states)):
    resistance = R_ON * state + R_OFF * (1 - state)
    print(f"{label:8}: State = {state:.3f}, Resistance = {resistance/1000:.1f} kΩ")

print("\n" + "="*80)
print("Full Adder Logic Summary:")
print("Sum = A ⊕ B ⊕ Cin (3-way XOR)")
print("Carry = AB + ACin + BCin (Majority function)")
print("="*80)
print("Simulation completed successfully!")
print("Plot saved as 'memristive_fhn_fulladder_oscillations.png'")
