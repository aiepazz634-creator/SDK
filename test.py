from qx_ir.core import Circuit, Op


# Create a quantum circuit with 2 qubits
circuit = Circuit(n_qubits=2)

# Add CX gate with control 0 and target 1
circuit.add_op(Op('cx', [0, 1]))

# Run the circuit on a backend
# from qx_ir.backend import LocalBackend
# backend = LocalBackend()
# job = backend.submit(circuit)
# result = job.result()
# print(result)