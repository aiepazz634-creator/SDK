"""
Quantum Device Implementation for executing quantum circuits.
"""
from qx_ir import Circuit, Op
from qx_ir.backend import LocalBackend
import numpy as np
from typing import Dict, Any, Optional
import time


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
        
    def execute(self, circuit: Circuit, shots: int = 1024) -> Dict[str, Any]:
        """
        Execute a quantum circuit and return the results.
        
        Args:
            circuit: The quantum circuit to execute
            shots: Number of shots to run the circuit for
            
        Returns:
            Dictionary containing the execution results
        """
        start_time = time.time()
        self.last_circuit = circuit
        
        try:
            # Submit the job to the backend
            job = self.backend.submit(circuit, shots=shots)
            
            # Wait for the job to complete
            while job.status() not in ['DONE', 'FAILED', 'CANCELLED']:
                time.sleep(0.1)
            
            # Get the results
            if job.status() == 'DONE':
                results = job.result()
                self.last_execution_time = time.time() - start_time
                return {
                    'status': 'success',
                    'results': results.get('result', {}),
                    'execution_time': self.last_execution_time,
                    'job_id': job.job_id
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
    
    def get_statevector(self, circuit: Optional[Circuit] = None) -> np.ndarray:
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
            from qx_ir.simulators import StatevectorSimulator
            simulator = StatevectorSimulator()
            return simulator.compute_statevector(circuit)
    
    def get_counts(self, circuit: Optional[Circuit] = None, shots: int = 1024) -> Dict[str, int]:
        """
        Get the measurement counts from a circuit execution.
        
        Args:
            circuit: The circuit to execute. If None, uses the last executed circuit.
            shots: Number of shots to run the circuit for.
            
        Returns:
            Dictionary mapping measurement outcomes to counts
        """
        if circuit is not None or self.last_circuit is None:
            result = self.execute(circuit or self.last_circuit, shots=shots)
            if result['status'] != 'success':
                raise RuntimeError(f"Failed to get counts: {result.get('error', 'Unknown error')}")
            return result['results']
        
        # If we already have results, return them
        if hasattr(self, 'last_results') and 'results' in self.last_results:
            return self.last_results['results']
            
        raise RuntimeError("No circuit has been executed yet")


# Global instance for easy access
device = QuantumDevice()


def get_quantum_device() -> QuantumDevice:
    """
    Get the global quantum device instance.
    
    Returns:
        The global QuantumDevice instance
    """
    return device
