# FitzHugh–Nagumo Binary Oscillator Computing with Memristors (Python)

This repository contains Python implementations of **binary logic circuits** built from networks of FitzHugh–Nagumo (FHN) neuron models, with and without **memristive synapses**.

Implemented circuits:

- Half-adder (Sum + Carry)
- Full adder (with carry-in)
- SR (Set–Reset) flip-flop
- Memristive variants of all of the above

The project is intended for experiments and demonstrations in:

- Nonlinear dynamics and excitable systems  
- Neuromorphic and in-memory computing  
- Oscillator-based binary logic  

---

## Repository structure

### FHN (baseline) circuits

- **`FHN_halfadder.py`**  
  - FitzHugh–Nagumo model of a **half-adder**.  
  - Two input neurons (A, B), one Sum neuron, one Carry neuron.  
  - Time-varying input pulses encode the four input combinations (00, 01, 10, 11).  
  - Simulates 2000 ms and saves a plot of the membrane potentials as  
    `fhn_half_adder_oscillations.png`.

- **`FHN_fulladder.py`**  
  - FitzHugh–Nagumo model of a **full adder**.  
  - Three inputs (A, B, Cin), one Sum neuron, one Carry neuron.  
  - Time-varying pulses encode all eight input combinations.  
  - Simulates 4000 ms and saves  
    `fhn_fulladder_oscillations.png`.

- **`FHN_SR_FlipFlop.py`**  
  - Object-oriented implementation of an **SR flip-flop**.  
  - Neurons: S, R, Q, and Q̄, each with membrane potential and recovery variable.  
  - Cross-coupling and noise produce bistable memory behaviour.  
  - Simulates 3000 ms and saves  
    `FHN_SR_FlipFlop.png`.

### Memristive FHN (MFHN) circuits

These scripts extend the FHN circuits with **memristor state variables** on specific synapses. Each memristor has an internal state that slowly modulates its resistance, and therefore the effective coupling between neurons.

- **`MFHN_halfadder.py`**  
  - Memristive version of the half-adder.  
  - Memristors on connections A→Sum, B→Sum, A→Carry, B→Carry.  
  - Simulates 2000 ms and saves  
    `memristive_fhn_halfadder_simple.png`.  
  - Prints a simple truth-table-style summary of output levels.

- **`MFHN_fulladder.py`**  
  - Memristive version of the full adder.  
  - Memristors on connections from A, B, Cin to Sum and Carry.  
  - Simulates 4000 ms and saves  
    `memristive_fhn_fulladder_oscillations.png`.  
  - Automatically checks the full adder truth table by averaging outputs over each input window.

- **`MFHN_SR_FlipFlop.py`**  
  - Memristive version of the SR flip-flop.  
  - Memristors on S→Q, R→Q̄, Q→Q̄, and Q̄→Q connections.  
  - Uses small sinusoidal noise and reports the logical state of Q/Q̄ over several time windows (initial, after set, before reset, after reset, final).  
  - Simulates 3000 ms and saves  
    `memristive_fhn_sr_flipflop.png`, along with final memristor resistances.

---

## Requirements

The scripts use the standard scientific Python stack:

- Python 3.8 or later
- NumPy
- SciPy
- Matplotlib

Install the dependencies with:

```bash
pip install numpy scipy matplotlib
