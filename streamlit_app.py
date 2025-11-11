import streamlit as st
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
from qx_ir import Circuit, Op, CircuitDrawer, Program
from qx_ir.simulator import StatevectorSimulator
from qx_ir.backend import LocalBackend


class QuantumDevice:
    """
    A quantum device that can execute quantum circuits and return results.
    """
    
    def __init__(self, backend: str = 'local'):
        """
        Initialize the quantum device with the specified backend.
        
        Args:
            backend: The backend to use for execution. Currently only 'local' is supported.
        """
        self.backend_type = backend
        self.backend = LocalBackend()
        self.last_execution_time = None
        self.last_circuit = None
        
    def execute(self, circuit: Circuit, shots: int = 1024) -> dict:
        """
        Execute a quantum circuit and return the results.
        
        Args:
            circuit: The quantum circuit to execute
            shots: Number of shots to run the circuit for (not used in local backend)
            
        Returns:
            Dictionary containing the execution results
        """
        start_time = time.time()
        self.last_circuit = circuit
        
        try:
            # Create a Program with the circuit
            program = Program(circuits=[circuit])
            
            # Submit the job to the backend
            job = self.backend.submit(program)
            
            # Add debug output
            st.write("Job submitted. Waiting for completion...")
            
            # Get the status as a string for comparison
            def get_status_str():
                status = str(job.status)
                # Handle both JobStatus.DONE and 'DONE' formats
                if '.' in status:
                    return status.split('.')[-1]
                return status
            
            # Wait for the job to complete with a timeout
            max_retries = 30  # 3 seconds max wait (30 * 0.1s)
            retry_count = 0
            
            while retry_count < max_retries:
                status = get_status_str()
                st.write(f"Current job status: {status}")
                
                if status in ['DONE', 'COMPLETED', 'FINISHED', 'SUCCESS']:
                    break
                elif status in ['FAILED', 'ERROR', 'CANCELLED']:
                    return {
                        'status': 'error',
                        'error': f"Job failed with status: {status}",
                        'job_id': getattr(job, 'job_id', None)
                    }
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                retry_count += 1
            
            final_status = get_status_str()
            
            # Get the results if the job completed successfully
            if final_status in ['DONE', 'COMPLETED', 'FINISHED', 'SUCCESS']:
                results = job.result()
                # The result should contain 'counts' or 'result' with the measurement results
                counts = results.get('counts', results.get('result', {}))
                self.last_execution_time = time.time() - start_time
                return {
                    'status': 'success',
                    'results': counts,
                    'execution_time': self.last_execution_time,
                    'job_id': getattr(job, 'job_id', None)
                }
            else:
                return {
                    'status': 'error',
                    'error': f"Job failed with status: {job.status()}",
                    'job_id': job.job_id if hasattr(job, 'job_id') else None
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'job_id': None
            }
    
    def get_statevector(self, circuit: Circuit = None) -> np.ndarray:
        """
        Get the statevector of a circuit.
        
        Args:
            circuit: The circuit to get the statevector for. If None, uses the last executed circuit.
            
        Returns:
            The statevector as a numpy array
        """
        circuit = circuit or self.last_circuit
        if circuit is None:
            raise ValueError("No circuit provided and no previous circuit executed")
            
        # For local simulation, we can get the statevector directly
        if hasattr(self.backend, 'get_statevector'):
            return self.backend.get_statevector(circuit)
        else:
            # Fallback: Simulate and compute statevector
            simulator = StatevectorSimulator()
            return simulator.compute_statevector(circuit)


# Global quantum device instance
quantum_device = QuantumDevice()

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

def run_circuit_simulation(circuit, shots=None):
    """Run a circuit simulation using the quantum device and return results.
    
    Args:
        circuit: The quantum circuit to execute
        shots: Number of shots to run. If None, uses the default from hardware config.
    """
    try:
        # Get max_shots from hardware config, default to 1024 if not found
        max_shots = HARDWARE_CONFIG.get('max_shots', 1024)
        
        # If shots is not provided, use max_shots
        if shots is None:
            shots = max_shots
        else:
            # Ensure shots doesn't exceed max_shots
            shots = min(shots, max_shots)
            
        st.info(f"Running simulation with {shots} shots (max allowed: {max_shots})")
        
        return quantum_device.execute(circuit, shots=shots)
    except Exception as e:
        st.error(f"Error in simulation: {str(e)}")
        return None

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
        page_title="ZenaQuantum - Quantum Circuit Builder",
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add your name and title at the top of the page
    st.markdown("""
    <h1 style='text-align: center; color: #1f77b4;'>
        ZenaQuantum - Quantum Circuit Simulator
    </h1>
    <hr style='height: 2px; background-color: #1f77b4; border: none;'>
    """, unsafe_allow_html=True)
    
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
        
        # Get gate information from the hardware config
        basis_gates = HARDWARE_CONFIG.get('basis_gates', [])
        gate_fidelities = HARDWARE_CONFIG.get('gate_fidelities', {})
        
        # Create a mapping of gate names to their display symbols
        GATE_DISPLAY = {
            'id': 'I',
            'rz': 'Rz',
            'sx': '√X',
            'x': 'X',
            'cx': '●',
            'h': 'H',
            't': 'T',
            'tdg': 'T†'
        }
        
        # Create a list of gate options with their display names and fidelities
        gate_options = []
        for gate in basis_gates:
            fidelity = gate_fidelities.get(gate, 0.0) * 100
            display_name = f"{GATE_DISPLAY.get(gate, gate)} ({gate})"
            gate_options.append((gate, display_name, fidelity))
        
        # Sort gates by type (single-qubit first, then multi-qubit) and then by name
        def get_gate_order(gate_info):
            gate = gate_info[0]
            if gate in ['cx', 'cz']:  # Multi-qubit gates
                return (1, gate)
            return (0, gate)  # Single-qubit gates
            
        gate_options.sort(key=get_gate_order)
        
        # Show gate selection dropdown with formatted names
        selected_gate_display = st.selectbox(
            "Select Gate",
            [opt[1] for opt in gate_options],
            key="gate_select"
        )
        
        # Get the actual gate name from the display name
        gate_type = next(opt[0] for opt in gate_options if opt[1] == selected_gate_display)
        
        # Show gate fidelity if available
        gate_fidelity = next((opt[2] for opt in gate_options if opt[0] == gate_type), None)
        if gate_fidelity is not None:
            st.caption(f"Fidelity: {gate_fidelity:.1f}%")
        
        # Show appropriate controls based on gate type
        if gate_type in ['cx']:  # Multi-qubit gates
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
        else:  # Single-qubit gates
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
        
        # Add shots configuration
        max_shots = HARDWARE_CONFIG.get('max_shots', 8192)
        default_shots = min(1024, max_shots)  # Default to 1024 or max_shots, whichever is smaller
        
        if 'shots' not in st.session_state:
            st.session_state.shots = default_shots
            
        shots = st.number_input(
            "Number of Shots",
            min_value=1,
            max_value=max_shots,
            value=st.session_state.get('shots', default_shots),
            help=f"Number of times to run the circuit (max {max_shots} based on hardware)",
            key="shots_input"
        )
        st.session_state.shots = min(shots, max_shots)  # Ensure it doesn't exceed max_shots
        
        # Add circuit controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Circuit", use_container_width=True):
                st.session_state.circuit = Circuit(n_qubits=num_qubits)
                st.session_state.gate_history = []
                st.rerun()
        
        with col2:
            if st.button("❌ Clear All", use_container_width=True):
                st.session_state.circuit = None
                st.session_state.gate_history = []
                st.rerun()
    
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
    
    # Run simulation button below the circuit
    # if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
    #     if not st.session_state.gate_history:
    #         st.warning("Please add some gates to the circuit first!")
    #     else:
    #         with st.spinner("Running simulation..."):
    #             st.session_state.simulation_results = run_circuit_simulation(
    #                 st.session_state.circuit,
    #                 shots=st.session_state.get('shots', 1024)  # Use the selected number of shots
    #             )
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
                # Extract counts from results if it's a dictionary with a 'results' key
                results = st.session_state.bell_results
                counts = results.get('results', results)  # Try to get 'results' key, fallback to the full results
                
                # Ensure counts is a dictionary with string keys and numeric values
                if isinstance(counts, dict):
                    # Convert any non-string keys to strings
                    counts = {str(k): int(v) for k, v in counts.items()}
                    
                    fig = CircuitDrawer.plot_results(
                        counts=counts,
                        title="Bell State Measurement Results",
                        xlabel="Quantum State",
                        ylabel="Counts",
                        color='#4b6cb7',
                        show=False
                    )
                    if fig:
                        st.pyplot(fig, clear_figure=True)
                        plt.close(fig)
                else:
                    st.error("Invalid results format. Expected a dictionary of counts.")
                    st.json(results)  # Show the actual results for debugging
    
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
                # Extract counts from results if it's a dictionary with a 'results' key
                results = st.session_state.qft_results
                counts = results.get('results', results)  # Try to get 'results' key, fallback to the full results
                
                # Ensure counts is a dictionary with string keys and numeric values
                if isinstance(counts, dict):
                    # Convert any non-string keys to strings
                    counts = {str(k): int(v) for k, v in counts.items()}
                    
                    fig = CircuitDrawer.plot_results(
                        counts=counts,
                        title="3-Qubit QFT Measurement Results",
                        xlabel="Quantum State",
                        ylabel="Counts",
                        color='#8e44ad',
                        show=False
                    )
                    if fig:
                        st.pyplot(fig, clear_figure=True)
                        plt.close(fig)
                else:
                    st.error("Invalid results format. Expected a dictionary of counts.")
                    st.json(results)  # Show the actual results for debugging


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
