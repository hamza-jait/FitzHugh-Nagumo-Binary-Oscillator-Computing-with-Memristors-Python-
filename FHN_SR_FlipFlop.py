"""
FitzHugh-Nagumo Model Implementation for SR Flip-Flop with Noise
---------------------------------------------------------------

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class FhnSrFlipFlop:
    """
    Class implementing an SR flip-flop using FitzHugh-Nagumo neurons.
    Includes noise resistance and visualization capabilities.
    """
    
    def __init__(self):
        """Initialize model parameters and simulation settings"""
        # Model parameters
        self.theta = 0.1      # Threshold parameter
        self.gamma = 0.1      # Recovery parameter
        self.epsilon = 0.1    # Time scale parameter
        self.m = -100         # Sigmoid slope
        self.c = 60           # Sigmoid offset
        self.w1 = 0.5         # Excitatory weight
        self.x1 = -1          # Inhibitory weight
        
        # Time settings
        self.t_max = 3000.0   # Maximum simulation time
        self.dt = 0.1         # Time step
        self.time = np.arange(0.0, self.t_max, self.dt)
        
        # Initialize noise
        self.noise_array = self.generate_noise()
    
    def generate_noise(self):
        """
        Generate Gaussian noise for the system.
        
        Returns:
            array: Noise values for the simulation
        """
        length = len(self.time)
        return 2.5 * np.random.normal(0, 0.05, length * 4)
    
    def apply_noise(self, t, channel):
        """
        Apply noise to a specific channel at time t.
        
        Parameters:
            t (float): Current time
            channel (int): Channel index (1-4)
            
        Returns:
            float: Noise value for the specified channel at time t
        """
        idx = round(10 * t - 1) * channel
        return self.noise_array[idx] if 0 <= idx < len(self.noise_array) else 0
    
    def set_input(self, t):
        """
        Define Set (S) input signal timing.
        
        Parameters:
            t (float): Current time
            
        Returns:
            float: Set signal value (0 or 1)
        """
        return 1 * (t > 1000) - 1 * (t > 1010)
    
    def reset_input(self, t):
        """
        Define Reset (R) input signal timing.
        
        Parameters:
            t (float): Current time
            
        Returns:
            float: Reset signal value (0 or 1)
        """
        return 1 * (t > 2000) - 1 * (t > 2010)
    
    def bias_q(self, t):
        """
        Q bias current (initialization pulse).
        
        Parameters:
            t (float): Current time
            
        Returns:
            float: Bias current value
        """
        return 1 * (t > 500)
    
    def bias_qbar(self, t):
        """
        Q̄ bias current (constant).
        
        Parameters:
            t (float): Current time
            
        Returns:
            float: Bias current value (constant 1)
        """
        return 1
    
    def sigmoid(self, v):
        """
        Sigmoid activation function.
        
        Parameters:
            v (float): Input value
            
        Returns:
            float: Sigmoid output (0 to 1)
        """
        return 1 / (1 + np.exp(self.m * v + self.c))
    
    def system_dynamics(self, state, t):
        """
        Define the FitzHugh-Nagumo system dynamics.
        
        Parameters:
            state (array): Current state vector
            t (float): Current time
            
        Returns:
            list: Derivatives of all state variables
        """
        # Unpack state variables (8 variables: 4 neurons, each with v and w)
        v_s, w_s, v_r, w_r, v_q, w_q, v_qbar, w_qbar = state
        
        # Input neurons (S and R)
        dv_s = -v_s * (v_s - self.theta) * (v_s - 1) - w_s + self.set_input(t) + self.apply_noise(t, 1)
        dw_s = self.epsilon * (v_s - self.gamma * w_s)
        
        dv_r = -v_r * (v_r - self.theta) * (v_r - 1) - w_r + self.reset_input(t) + self.apply_noise(t, 2)
        dw_r = self.epsilon * (v_r - self.gamma * w_r)
        
        # Output neurons (Q and Q̄) with cross-inhibition
        dv_q = (-v_q * (v_q - self.theta) * (v_q - 1) - w_q + 
                self.w1 * self.sigmoid(w_s) +          # Excitation from S
                self.x1 * self.sigmoid(w_qbar) +       # Inhibition from Q̄
                self.bias_q(t) +                       # Bias current
                self.apply_noise(t, 3))                # Noise
        dw_q = self.epsilon * (v_q - self.gamma * w_q)
        
        dv_qbar = (-v_qbar * (v_qbar - self.theta) * (v_qbar - 1) - w_qbar + 
                   0.45 * self.sigmoid(w_r) +          # Excitation from R (using 0.45 weight)
                   self.x1 * self.sigmoid(w_q) +       # Inhibition from Q
                   self.bias_qbar(t) +                 # Bias current
                   self.apply_noise(t, 4))             # Noise
        dw_qbar = self.epsilon * (v_qbar - self.gamma * w_qbar)
        
        return [dv_s, dw_s, dv_r, dw_r, dv_q, dw_q, dv_qbar, dw_qbar]
    
    def simulate(self):
        """
        Perform the simulation by solving the system of ODEs.
        
        Returns:
            array: Solution array with all state variables over time
        """
        # Initial conditions
        initial_state = [0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0]
        
        # Solve the system
        solution = odeint(
            self.system_dynamics,
            initial_state,
            self.time,
            rtol=1e-6
        )
        
        return solution
    
    def visualize_results(self, solution):
        """
        Create plots of the simulation results.
        
        Parameters:
            solution (array): Simulation results
        """
        # Extract membrane potentials from solution
        v_s = solution[:, 0]
        v_r = solution[:, 2]
        v_q = solution[:, 4]
        v_qbar = solution[:, 6]
        
        # Configure plot
        fig = plt.figure(figsize=(10, 8))  # Save figure reference
        plt.rcParams["font.size"] = "14"
        plt.subplots_adjust(hspace=1)
        
        # Plot Set input
        ax1 = plt.subplot(4, 1, 1)
        ax1.set_title("Fitzhugh-Nagumo SR Flip-Flop With Noise")
        ax1.plot(self.time, v_s, "Purple")
        ax1.set_ylim(-1, 1.5)
        ax1.set_ylabel("Set (S)")
        
        # Plot Reset input
        ax2 = plt.subplot(4, 1, 2)
        ax2.plot(self.time, v_r, "Purple")
        ax2.set_ylim(-1, 1.5)
        ax2.set_ylabel("Reset (R)")
        
        # Plot Q output
        ax3 = plt.subplot(4, 1, 3)
        ax3.plot(self.time, v_q, "Crimson")
        ax3.set_ylim(-1, 1.5)
        ax3.set_ylabel("Q")
        
        # Plot Q̄ output
        ax4 = plt.subplot(4, 1, 4)
        ax4.plot(self.time, v_qbar, "Crimson")
        ax4.set_ylim(-1, 1.5)
        ax4.set_ylabel("Q̄")
        ax4.set_xlabel("Time")
        
        # Save figure to file with high DPI for quality
        fig.savefig('FHN_SR_FlipFlop.png', dpi=300, bbox_inches='tight')
        print("Figure saved as 'FHN_SR_FlipFlop.png'")
        
        # Display plot
        plt.show()

# Run simulation
if __name__ == "__main__":
    # Create model instance
    flip_flop = FhnSrFlipFlop()
    
    # Run simulation
    print("Starting simulation (this may take about a minute)...")
    results = flip_flop.simulate()
    
    # Visualize results
    flip_flop.visualize_results(results)