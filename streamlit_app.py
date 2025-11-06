import streamlit as st
import json
import os
from pathlib import Path

# Add parent directory to path to import the required modules
import sys
sys.path.append(str(Path(__file__).parent))

from qx_ir.core import Circuit, Op
# from qx_ir.core import Circuit, Op
# from qx_ir.backend import LocalBackend
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

def main():
    st.set_page_config(
        page_title="Quantum Circuit Builder",
        page_icon="⚛️",
        layout="wide"
    )
    
    st.title("⚛️ Quantum Circuit Builder")
    
    # Initialize session state
    if 'circuit' not in st.session_state:
        st.session_state.circuit = None
        st.session_state.gate_history = []
        st.session_state.num_qubits = 2
    
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
            st.experimental_rerun()
        
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
            st.experimental_rerun()
            
        if st.button("❌ Clear All"):
            st.session_state.circuit = None
            st.session_state.gate_history = []
            st.experimental_rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎛️ Circuit Visualization")
        
        # Initialize circuit if needed
        if st.session_state.circuit is None:
            st.session_state.circuit = Circuit(n_qubits=num_qubits)
        
        # Display the circuit
        display_circuit(st.session_state.gate_history, st.session_state.num_qubits)
        
        # Add some space
        st.markdown("\n\n")
        
        # Run simulation button
        # if st.button("▶️ Run Simulation"):
        #     if not st.session_state.gate_history:
        #         st.warning("Please add some gates to the circuit first!")
        #     else:
        #         with st.spinner("Running simulation..."):
        #             run_simulation(st.session_state.circuit)
    
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

def add_gate(gate_type, qubits):
    """Add a gate to the circuit"""
    if 'circuit' not in st.session_state:
        st.session_state.circuit = Circuit(n_qubits=max(qubits)+1)
        st.session_state.gate_history = []
    
    # Add gate to circuit
    st.session_state.circuit.add_op(Op(gate_type, qubits))
    st.session_state.gate_history.append((gate_type, qubits))
    st.experimental_rerun()

def display_circuit(gate_history, num_qubits):
    """Display the circuit using Streamlit"""
    if not gate_history:
        st.info("No gates added yet. Use the sidebar to add gates to your circuit.")
        return
    
    # Create a container for the circuit
    circuit_container = st.container()
    
    # Calculate the number of columns needed (1 for qubit labels + 1 for each gate)
    num_columns = len(gate_history) + 1
    
    # Create columns for the circuit
    cols = circuit_container.columns([1] + [2] * len(gate_history))
    
    # Draw qubit lines and labels
    for q in range(num_qubits):
        with cols[0]:
            st.markdown(f"**q[{q}]** ───", unsafe_allow_html=True)
    
    # Draw gates
    for i, (gate, qubits) in enumerate(gate_history, 1):
        with cols[i]:
            for q in range(num_qubits):
                if q in qubits:
                    if gate in GATE_DISPLAY:
                        display_char = GATE_DISPLAY[gate]
                    else:
                        display_char = gate.upper()
                    
                    # Special handling for CNOT and CZ gates
                    if gate in ['cx', 'cz']:
                        if q == qubits[0]:  # Control qubit
                            st.markdown(f"───<span style='font-size: 24px;'>●</span>───", unsafe_allow_html=True)
                        else:  # Target qubit
                            if gate == 'cx':
                                st.markdown("─┳─<span style='font-size: 24px;'>⊕</span>─┳─", unsafe_allow_html=True)
                            else:  # cz
                                st.markdown("─┳─<span style='font-size: 24px;'>Z</span>─┳─", unsafe_allow_html=True)
                    else:
                        st.markdown(f"───<span style='font-size: 20px;'>{display_char}</span>───", unsafe_allow_html=True)
                else:
                    # Draw wire continuation
                    if i > 1 and any(q in g[1] for g in gate_history[i-1:i+1] if g[0] in ['cx', 'cz']):
                        st.markdown("───│───", unsafe_allow_html=True)
                    else:
                        st.markdown("───────", unsafe_allow_html=True)

def generate_code(gate_history, num_qubits):
    """Generate the Python code for the circuit"""
    if not gate_history:
        return "# No gates added yet. Use the sidebar to add gates to your circuit."
    
    code = [
        "from qx_ir.core import Circuit, Op\n\n",
        f"# Create a quantum circuit with {num_qubits} qubits",
        f"circuit = Circuit(n_qubits={num_qubits})\n"
    ]
    
    for i, (gate, qubits) in enumerate(gate_history, 1):
        if len(qubits) == 1:
            code.append(f"# Add {gate.upper()} gate on qubit {qubits[0]}")
            code.append(f"circuit.add_op(Op('{gate}', [{qubits[0]}]))")
        else:
            code.append(f"# Add {gate.upper()} gate with control {qubits[0]} and target {qubits[1]}")
            code.append(f"circuit.add_op(Op('{gate}', {qubits}))")
    
    code.append("\n# Run the circuit on a backend")
    code.append("# from qx_ir.backend import LocalBackend")
    code.append("# backend = LocalBackend()")
    code.append("# job = backend.submit(circuit)")
    code.append("# result = job.result()")
    code.append("# print(result)")
    
    return "\n".join(code)

def run_simulation(circuit):
    """Run the circuit simulation"""
    try:
        from qx_ir.backend import LocalBackend
        
        backend = LocalBackend()
        job = backend.submit(circuit)
        
        # Show job status
        with st.spinner(f"Job {job.job_id} is running..."):
            while job.status() not in ['DONE', 'FAILED', 'CANCELLED']:
                time.sleep(0.5)
        
        if job.status() == 'DONE':
            result = job.result()
            st.success("✅ Simulation completed successfully!")
            
            # Display results in an expandable section
            with st.expander("📊 View Results"):
                st.json(result)
                
                # If there are measurement results, show them in a bar chart
                if 'result' in result and isinstance(result['result'], dict):
                    import plotly.express as px
                    
                    # Convert counts to DataFrame for visualization
                    counts = result['result']
                    df = pd.DataFrame({
                        'State': list(counts.keys()),
                        'Count': list(counts.values())
                    })
                    
                    # Create bar chart
                    fig = px.bar(
                        df,
                        x='State',
                        y='Count',
                        title='Measurement Results',
                        color='State',
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title="Quantum State",
                        yaxis_title="Count",
                        showlegend=False
                    )
                    
                    # Show the plot
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"❌ Simulation failed with status: {job.status()}")
            
    except Exception as e:
        st.error(f"❌ Error running simulation: {str(e)}")

if __name__ == "__main__":
    # Import these here to avoid circular imports
    import pandas as pd
    import time
    
    main()
