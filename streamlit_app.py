import streamlit as st
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from qx_ir import Circuit, Op, CircuitDrawer, Program
from qx_ir.simulator import StatevectorSimulator
from qx_ir.backend import LocalBackend

# Add parent directory to path to import the required modules
import sys
sys.path.append(str(Path(__file__).parent))
# Load hardware configuration
with open('qxir_v1.json', 'r') as f:
    HARDWARE_CONFIG = json.load(f)

MAX_QUBITS = HARDWARE_CONFIG.get('n_qubits', 5)
AVAILABLE_GATES = list(HARDWARE_CONFIG.get('gate_fidelities', {}).keys())
GATE_DISPLAY = {
    'h': 'H',
    'x': 'X',
    'y': 'Y',
    'z': 'Z',
    'sx': '√X',
    'cx': '●',
    'cz': '●',
    't': 'T',
    'tdg': 'T†',
    'id': 'I'
}

def create_bell_circuit():
    """Create a Bell state circuit."""
    circuit = Circuit(n_qubits=2)
    circuit.add_op(Op('h', qubits=[0]))
    circuit.add_op(Op('cx', qubits=[0, 1]))
    return circuit

def create_qft_circuit(n_qubits=3):
    """Create a Quantum Fourier Transform circuit."""
    circuit = Circuit(n_qubits=n_qubits)
    
    for i in range(n_qubits):
        circuit.add_op(Op('h', qubits=[i]))
        for j in range(i + 1, n_qubits):
            angle = 2 * 3.14159 / (2 ** (j - i + 1))
            circuit.add_op(Op('cp', qubits=[j, i], params=[angle]))
    
    for i in range(n_qubits // 2):
        circuit.add_op(Op('swap', qubits=[i, n_qubits - 1 - i]))
    
    return circuit

def run_circuit_simulation(circuit, shots=1000):
    """Run a circuit simulation and return results."""
    simulator = StatevectorSimulator()
    program = Program(circuits=[circuit], config={'shots': shots})
    results = simulator.run(program)
    return results

def add_gate(gate_type, qubits):
    """Add a gate to the circuit"""
    if 'circuit' not in st.session_state:
        st.session_state.circuit = Circuit(n_qubits=max(qubits)+1 if isinstance(qubits, (list, tuple)) else qubits+1)
        st.session_state.gate_history = []
    
    # Handle both single and multi-qubit gates
    if isinstance(qubits, (list, tuple)):
        if gate_type in ['cx', 'cz'] and len(qubits) >= 2:
            st.session_state.circuit.add_op(Op(gate_type, qubits=qubits[:2]))
        else:
            st.session_state.circuit.add_op(Op(gate_type, qubits=[qubits[0]]))
    else:
        st.session_state.circuit.add_op(Op(gate_type, qubits=[qubits]))
    
    st.session_state.gate_history.append((gate_type, qubits if isinstance(qubits, (list, tuple)) else [qubits]))

def display_circuit(gate_history, num_qubits):
    """Display the circuit with error handling"""
    if not gate_history:
        st.info("No gates added yet. Use the sidebar to add gates to your circuit.")
        return
        
    try:
        circuit = Circuit(n_qubits=num_qubits)
        for gate_type, qubits in gate_history:
            if not qubits:  # Skip if no qubits specified
                continue
                
            if gate_type in ['cx', 'cz'] and len(qubits) >= 2:
                circuit.add_op(Op(gate_type, qubits=qubits[:2]))
            else:
                circuit.add_op(Op(gate_type, qubits=[qubits[0]] if isinstance(qubits, (list, tuple)) else [qubits]))
        
        # Create and display the circuit visualization
        with st.container():
            fig = CircuitDrawer.draw_mpl(circuit, show=False)
            if fig:
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)  # Close the figure to prevent text output
            else:
                st.warning("Could not display circuit")
        
    except Exception as e:
        st.error(f"Error displaying circuit: {str(e)}")
        st.exception(e)

def generate_code(gate_history, num_qubits):
    """Generate code for the circuit"""
    code = "import qx_ir\n\ncircuit = qx_ir.Circuit(n_qubits={})\n".format(num_qubits)
    for gate_type, qubits in gate_history:
        if gate_type in ['cx', 'cz']:
            code += "circuit.add_op(qx_ir.Op('{}', qubits={}))\n".format(gate_type, qubits)
        else:
            code += "circuit.add_op(qx_ir.Op('{}', qubits=[{}]))\n".format(gate_type, qubits[0])
    
    return code

def main():
    st.set_page_config(
        page_title="Quantum Circuit Builder",
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'circuit' not in st.session_state:
        st.session_state.circuit = Circuit(n_qubits=2)
        st.session_state.gate_history = []
        st.session_state.num_qubits = 2
        st.session_state.simulation_results = None
        st.session_state.show_demo = False
    
    # Sidebar for circuit configuration
    with st.sidebar:
        st.header("⚙️ Circuit Configuration")
        
        num_qubits = st.number_input(
            "Number of Qubits",
            min_value=1,
            max_value=MAX_QUBITS,
            value=st.session_state.num_qubits,
            key="num_qubits_input"
        )
        
        # Update circuit if number of qubits changes
        if num_qubits != st.session_state.num_qubits:
            st.session_state.num_qubits = num_qubits
            st.session_state.circuit = Circuit(n_qubits=num_qubits)
            st.session_state.gate_history = []
            st.rerun()
            # st.experimental_rerun()
        
        st.subheader("➕ Add Gate")
        
        # Gate selection
        gate_type = st.selectbox("Select Gate", AVAILABLE_GATES, key="gate_select")
        
        # For multi-qubit gates, show appropriate controls
        if gate_type in ['cx', 'cz']:
            col1, col2 = st.columns(2)
            with col1:
                control_qubit = st.number_input(
                    "Control Qubit",
                    min_value=0,
                    max_value=num_qubits-1,
                    value=0,
                    key=f"control_qubit_{gate_type}"
                )
            with col2:
                target_qubit = st.number_input(
                    "Target Qubit",
                    min_value=0,
                    max_value=num_qubits-1,
                    value=1 if num_qubits > 1 else 0,
                    key=f"target_qubit_{gate_type}"
                )
            
            if st.button("Add Gate", key=f"add_{gate_type}"):
                if control_qubit == target_qubit:
                    st.error("Control and target qubits must be different")
                else:
                    add_gate(gate_type, [control_qubit, target_qubit])
        else:
            qubit = st.number_input(
                "Qubit",
                min_value=0,
                max_value=num_qubits-1,
                value=0,
                key=f"qubit_{gate_type}"
            )
            
            if st.button("Add Gate", key=f"add_{gate_type}"):
                add_gate(gate_type, [qubit])
        
        # Add a divider
        st.markdown("---")
        
        # Add circuit controls
        if st.button("🔄 Reset Circuit"):
            st.session_state.circuit = Circuit(n_qubits=num_qubits)
            st.session_state.gate_history = []
            st.rerun()
            # st.experimental_rerun()
            
        if st.button("❌ Clear All"):
            st.session_state.circuit = None
            st.session_state.gate_history = []
            st.rerun()
            # st.experimental_rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_circuit(st.session_state.gate_history, st.session_state.num_qubits)
    
    with col2:
        st.header("📜 Generated Code")
        code = generate_code(st.session_state.gate_history, st.session_state.num_qubits)
        st.code(code, language='python')
        
        # Add copy to clipboard button
        st.download_button(
            label="💾 Save as .py",
            data=code,
            file_name="quantum_circuit.py",
            mime="text/plain"
        )
    
    # # Run simulation button below the circuit
    # if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
    #     if not st.session_state.gate_history:
    #         st.warning("Please add some gates to the circuit first!")
    #     else:
    #         run_simulation(st.session_state.circuit)
    
    # Demo circuits section below everything
    # st.markdown("---")
    # st.header("🎯 Demo Circuits")
    demo_option = st.selectbox(
        "Choose a demo circuit:",
        ["Select a demo", "Bell State", "Quantum Fourier Transform (3-qubit)"]
    )
    
    # Demo circuit display and simulation - Single column layout
    if demo_option == "Bell State":
        st.info("Creating a Bell state (entangled pair) between two qubits.")
        
        # Create two columns for circuit and results side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Bell State Circuit")
            circuit = create_bell_circuit()
            fig = CircuitDrawer.draw_mpl(circuit, show=False)
            if fig:
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            
            if st.button("Run Bell State Simulation"):
                with st.spinner("Running Bell state simulation..."):
                    results = run_circuit_simulation(circuit)
                    st.session_state.bell_results = results
        
        # Show results if available
        if 'bell_results' in st.session_state and st.session_state.bell_results is not None:
            with col2:
                st.subheader("Measurement Results")
                fig = CircuitDrawer.plot_results(
                    counts=st.session_state.bell_results,
                    title="Bell State Measurement Results",
                    xlabel="Quantum State",
                    ylabel="Counts",
                    color='#4b6cb7',
                    show=False
                )
                if fig:
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)
    
    elif demo_option == "Quantum Fourier Transform (3-qubit)":
        st.info("3-qubit Quantum Fourier Transform circuit.")
        
        # Create two columns for circuit and results side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("QFT Circuit (with |100> input)")
            
            # Prepare QFT circuit with input state |100>
            circuit = create_qft_circuit(3)
            input_circuit = Circuit(n_qubits=3)
            input_circuit.add_op(Op('x', qubits=[0]))
            
            # Combine input and QFT circuits
            full_circuit = Circuit(n_qubits=3)
            for op in input_circuit.instructions + circuit.instructions:
                full_circuit.add_op(op)
            
            # Display circuit
            fig = CircuitDrawer.draw_mpl(full_circuit, show=False)
            if fig:
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            
            if st.button("Run QFT Simulation"):
                with st.spinner("Running QFT simulation..."):
                    results = run_circuit_simulation(full_circuit)
                    st.session_state.qft_results = results
        
        # Show results if available
        if 'qft_results' in st.session_state and st.session_state.qft_results is not None:
            with col2:
                st.subheader("QFT Measurement Results")
                fig = CircuitDrawer.plot_results(
                    counts=st.session_state.qft_results,
                    title="3-Qubit QFT Measurement Results",
                    xlabel="Quantum State",
                    ylabel="Counts",
                    color='#8e44ad',
                    show=False
                )
                if fig:
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)


if __name__ == "__main__":
    # Import these here to avoid circular imports
    import pandas as pd
    import time
    
    # Suppress DeltaGenerator output
    import sys
    from contextlib import redirect_stdout
    from io import StringIO
    
    # Redirect stdout to suppress DeltaGenerator debug output
    with redirect_stdout(StringIO()):
        main()
