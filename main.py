import numpy as np
import matplotlib.pyplot as plt
import qutip 
import time 


try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception as e:
    print(f"Warning: Could not apply 'seaborn-v0_8-whitegrid' style. Using default. Error: {e}")
plt.rcParams.update({'font.size': 12})


N_QUBITS = 10 
D_HILBERT = 2**N_QUBITS 


def plot_eigenvalue_spectrum(eigenvalues, state_name="State", log_scale=False):
    """Plots the eigenvalue spectrum (bar chart)."""
    if eigenvalues is None:
        print(f"Skipping eigenvalue spectrum plot for {state_name} as eigenvalues are None.")
        return

    num_eigenvalues = len(eigenvalues)
    indices = np.arange(num_eigenvalues)

    plt.figure(figsize=(12, 7)) 
    if log_scale:
        
        plt.bar(indices, eigenvalues + 1e-18, color='skyblue', edgecolor='black', width=0.8)
        plt.yscale('log')
        plt.ylabel("Eigenvalue (Population/Weight) - Log Scale")
        min_val = np.min(eigenvalues[eigenvalues > 1e-17]) if np.any(eigenvalues > 1e-17) else 1e-18
        plt.ylim(bottom=min_val * 0.1, top=1.2)
    else:
        plt.bar(indices, eigenvalues, color='skyblue', edgecolor='black', width=0.8)
        plt.ylabel("Eigenvalue (Population/Weight)")
        plt.ylim(0, 1.05)

    plt.xlabel("Principal Component Index (Sorted by Eigenvalue)")
    plt.title(f"Eigenvalue Spectrum for {state_name} ({num_eigenvalues} components)")

    if num_eigenvalues > 30:
        tick_step = max(1, num_eigenvalues // 20) 
        plt.xticks(indices[::tick_step], rotation=45, ha='right')
    else:
        plt.xticks(indices)

    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_cumulative_eigenvalues(eigenvalues, state_name="State"):
    """Plots the cumulative sum of sorted eigenvalues."""
    if eigenvalues is None:
        print(f"Skipping cumulative eigenvalue plot for {state_name} as eigenvalues are None.")
        return

    cumulative_sum = np.cumsum(eigenvalues) 
    indices = np.arange(1, len(eigenvalues) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(indices, cumulative_sum, marker='o', linestyle='-', color='coral')
    plt.xlabel("Number of Principal Components Included")
    plt.ylabel("Cumulative Sum of Eigenvalues (Captured 'Trace')")
    plt.title(f"Cumulative Eigenvalue Sum for {state_name}")
    plt.xlim(0, len(eigenvalues) + 1)
    plt.ylim(0, 1.05)
    plt.axhline(1.0, color='grey', linestyle='--', alpha=0.7, label='Total Trace (1.0)')
    plt.axhline(0.99, color='lightgreen', linestyle=':', alpha=0.7, label='99% Captured')
    plt.axhline(0.95, color='gold', linestyle=':', alpha=0.7, label='95% Captured')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()



print(f"\n--- Section 1: Quantum Objects for {N_QUBITS} Qubits ---")
# Create list of |0> and |1> kets for tensor products
kets0_list = [qutip.basis(2, 0) for _ in range(N_QUBITS)]
kets1_list = [qutip.basis(2, 1) for _ in range(N_QUBITS)]


ket_all_zeros = qutip.tensor(kets0_list)
ket_all_ones = qutip.tensor(kets1_list)

# Identity operator 
identity_Nq = qutip.identity([2] * N_QUBITS)

print(f"Defined {N_QUBITS}-qubit basis states (e.g., |0...0>) and identity operator (I_{D_HILBERT}).")
print(f"Dimension of {N_QUBITS}-qubit Hilbert space: {D_HILBERT}")
print(f"Shape of {N_QUBITS}-qubit identity operator: {identity_Nq.shape}")
print(f"Dims of {N_QUBITS}-qubit identity operator: {identity_Nq.dims}")



def analyze_density_matrix_pca(rho_qobj, state_name="State"):
    if not isinstance(rho_qobj, qutip.Qobj) or not rho_qobj.isoper:
        
        if isinstance(rho_qobj, np.ndarray):
            try:
                if rho_qobj.ndim == 2 and rho_qobj.shape[0] == rho_qobj.shape[1]:
                    dim_size = rho_qobj.shape[0]
                    num_qubits_inferred = 0
                    if dim_size > 0 and (dim_size & (dim_size - 1) == 0): 
                         num_qubits_inferred = int(np.log2(dim_size))

                    if num_qubits_inferred > 0 :
                        qubit_dims = [2] * num_qubits_inferred
                        dims = [qubit_dims, qubit_dims]
                    else: 
                        dims = [[dim_size],[dim_size]]
                    rho_qobj = qutip.Qobj(rho_qobj, dims=dims)
                else:
                    print(f"Warning: Input NumPy array {state_name} may not be a square matrix.")
                    return None, None, None
            except Exception as e:
                print(f"Error: Failed to convert NumPy array {state_name} to Qobj: {e}")
                return None, None, None
        else:
            print(f"Error: Input {state_name} is not a valid density matrix format.")
            return None, None, None
        if not rho_qobj.isoper:
             print(f"Error: Converted {state_name} is still not a valid operator.")
             return None, None, None

    print(f"\n--- PCA Analysis for: {state_name} ---")
    print(f"Density Matrix (dims={rho_qobj.dims}, shape={rho_qobj.shape})")
    
    if N_QUBITS < 4:
        print(rho_qobj.full().round(4))
    else:
        print(f"(Matrix is {rho_qobj.shape[0]}x{rho_qobj.shape[0]}, too large to print fully here)")

    tr = rho_qobj.tr(); herm = rho_qobj.isherm; purity = rho_qobj.purity()
    print(f"Trace: {tr:.4f}, Is Hermitian: {herm}, Purity: {purity:.4f}")
    if not np.isclose(tr, 1.0): print(f"Warning: Trace of {state_name} is not 1.")
    if not herm: print(f"Warning: {state_name} is not Hermitian.")

    
    start_time_eigcheck = time.time()
    raw_eig_check = np.linalg.eigvalsh(rho_qobj.full())
    end_time_eigcheck = time.time()
    print(f"(Time for np.linalg.eigvalsh check: {end_time_eigcheck - start_time_eigcheck:.3f}s)")
    if np.any(raw_eig_check < -1e-9): print(f"Warning: {state_name} not positive semi-definite (min raw eigenvalue: {np.min(raw_eig_check):.2e}).")

    print("Performing QuTiP eigendecomposition (can take time for large N)...")
    start_time_qutip_eig = time.time()
    eigenvalues, eigenvectors = rho_qobj.eigenstates()
    end_time_qutip_eig = time.time()
    print(f"(Time for QuTiP rho.eigenstates(): {end_time_qutip_eig - start_time_qutip_eig:.3f}s)")

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = [eigenvectors[i] for i in sorted_indices]

    print(f"\nEigenvalues (λ_i - Weights of Principal Quantum States) - Top {min(10, D_HILBERT)} shown:")
    for i in range(min(len(sorted_eigenvalues), 10)): print(f"  λ_{i}: {sorted_eigenvalues[i]:.6f}")
    if len(sorted_eigenvalues) > 10: print(f"  ... ({len(sorted_eigenvalues) - 10} more eigenvalues not shown here, but plotted)")

    print(f"\nDominant Eigenvector(s) (|v_i⟩) - Top {min(2, D_HILBERT)} shown:")
    for i in range(min(len(sorted_eigenvectors), 2)):
        vec = sorted_eigenvectors[i]
        print(f"  |v_{i}⟩ (for λ_{i}={sorted_eigenvalues[i]:.6f}):")
        print(f"    (Vector is {D_HILBERT}-dimensional. Fidelity with known states might be more informative than printing elements.)")
    if len(sorted_eigenvectors) > min(2,D_HILBERT) and D_HILBERT > 2: print(f"  ... ({len(sorted_eigenvectors) - min(2,D_HILBERT)} more eigenvectors not shown)")

    return sorted_eigenvalues, sorted_eigenvectors, rho_qobj


print(f"\n\n--- Section 3: PCA Demo with {N_QUBITS}-Qubit GHZ State and Noise ---")
print(f"Goal: Demonstrate PCA's power on a high-dimensional ({N_QUBITS}-qubit) entangled state subjected to noise.")


print(f"\nStep A: Define Original True {N_QUBITS}-Qubit State (ρ_true_{N_QUBITS}q)")
ghz_Nq_ket = (ket_all_zeros + ket_all_ones).unit()
rho_true_Nq = qutip.ket2dm(ghz_Nq_ket)

print(f"Analyzing Original True {N_QUBITS}-Qubit Density Matrix (ρ_true_{N_QUBITS}q - GHZ State):")
eigvals_true_Nq, eigvecs_true_Nq, _ = analyze_density_matrix_pca(rho_true_Nq, f"ρ_true_{N_QUBITS}q (GHZ State)")
if eigvals_true_Nq is not None:
    plot_eigenvalue_spectrum(eigvals_true_Nq, f"ρ_true_{N_QUBITS}q (GHZ State)")
    plot_cumulative_eigenvalues(eigvals_true_Nq, f"ρ_true_{N_QUBITS}q (GHZ State)")
    print(f"  Interpretation of ρ_true_{N_QUBITS}q: Pure GHZ state (Purity ≈ 1), one dominant eigenvalue ≈ 1.")
    
    if eigvecs_true_Nq:
        fid_eigvec_ghz = qutip.fidelity(eigvecs_true_Nq[0], ghz_Nq_ket)
        print(f"  Fidelity of dominant eigenvector of ρ_true with the defined GHZ ket: {fid_eigvec_ghz**2:.6f} (should be ~1.0)")


print(f"\nStep B: Simulate Noise - Apply Depolarizing Channel to ρ_true_{N_QUBITS}q")
noise_probability = 0.5 # Make noise 
rho_noisy_Nq = (1 - noise_probability) * rho_true_Nq + noise_probability * (identity_Nq / D_HILBERT)

print(f"\nAnalyzing Noisy {N_QUBITS}-Qubit Density Matrix (ρ_noisy_{N_QUBITS}q) after {noise_probability*100}% depolarizing noise:")
eigvals_noisy_Nq, eigvecs_noisy_Nq, rho_noisy_Nq_analyzed = analyze_density_matrix_pca(rho_noisy_Nq, f"ρ_noisy_{N_QUBITS}q")

if eigvals_noisy_Nq is not None:
    plot_eigenvalue_spectrum(eigvals_noisy_Nq, f"ρ_noisy_{N_QUBITS}q", log_scale=True) # Use log scale for many small eigenvalues
    plot_cumulative_eigenvalues(eigvals_noisy_Nq, f"ρ_noisy_{N_QUBITS}q")
    print(f"\n  'With PCA' insight for ρ_noisy_{N_QUBITS}q:")
    print(f"  Purity significantly less than 1. Eigenvalue spectrum shows the spread due to noise.")
    print(f"  The largest eigenvalue λ_0 = {eigvals_noisy_Nq[0]:.6f} (corresponding to eigenvector |v_0⟩) represents the most dominant pure state component.")
    print(f"  The cumulative plot shows how many components are needed to capture most of the state's trace.")
    if eigvecs_noisy_Nq:
        fid_noisy_dom_eigvec_vs_ghz = qutip.fidelity(eigvecs_noisy_Nq[0], ghz_Nq_ket)
        print(f"  Fidelity of dominant eigenvector of ρ_noisy with the original GHZ ket: {fid_noisy_dom_eigvec_vs_ghz**2:.6f}")
        print(f"  This indicates if the 'main feature' picked by PCA still resembles the original GHZ structure.")



print(f"\nStep C: 'Filter' ρ_noisy_{N_QUBITS}q using its Dominant PCA Component (ρ_filtered_pure_approx_{N_QUBITS}q)")
rho_filtered_pure_approximation_Nq = None # Initialize
if eigvals_noisy_Nq is not None and len(eigvals_noisy_Nq) > 0:
    dominant_eigenvalue_noisy_Nq = eigvals_noisy_Nq[0]
    dominant_eigenvector_noisy_Nq = eigvecs_noisy_Nq[0]
    rho_filtered_pure_approximation_Nq = qutip.ket2dm(dominant_eigenvector_noisy_Nq)

    print(f"Pure state approximation based on the dominant eigenvector of ρ_noisy_{N_QUBITS}q (ρ_filtered_pure_approx_{N_QUBITS}q):")
    eigvals_filt, _, _ = analyze_density_matrix_pca(rho_filtered_pure_approximation_Nq, f"ρ_filtered_pure_approx_{N_QUBITS}q")
    if eigvals_filt is not None:
        plot_eigenvalue_spectrum(eigvals_filt, f"ρ_filtered_pure_approx_{N_QUBITS}q")
        # Cumulative plot for a pure state is trivial (one step to 1.0)
    print(f"  PCA Advantage: Extracted a simple pure state (Purity ≈ 1) representing ρ_noisy_{N_QUBITS}q's main feature.")
else:
    print(f"Skipped filtering as PCA on ρ_noisy_{N_QUBITS}q failed or yielded no components.")


print(f"\nStep D: Compare States using Fidelity - Highlighting PCA's Advantages for {N_QUBITS} Qubits")
if rho_true_Nq is not None and rho_noisy_Nq_analyzed is not None:
    fidelity_true_vs_noisy_Nq = qutip.fidelity(rho_true_Nq, rho_noisy_Nq_analyzed)
    print(f"\n1. Fidelity F(ρ_true_{N_QUBITS}q, ρ_noisy_{N_QUBITS}q): {fidelity_true_vs_noisy_Nq:.6f}")
    print(f"   Measures similarity of 'raw' noisy state to original ideal {N_QUBITS}-qubit GHZ state.")

if rho_true_Nq is not None and rho_filtered_pure_approximation_Nq is not None:
    fidelity_true_vs_filtered_Nq = qutip.fidelity(rho_true_Nq, rho_filtered_pure_approximation_Nq)
    print(f"\n2. Fidelity F(ρ_true_{N_QUBITS}q, ρ_filtered_pure_approx_{N_QUBITS}q): {fidelity_true_vs_filtered_Nq:.6f}")
    print(f"   Measures similarity of 'PCA-filtered' pure state to original ideal GHZ state.")

    if rho_noisy_Nq_analyzed is not None :
        print(f"\n   Comparison for PCA's Power:")
        print(f"   - Fidelity of full Noisy State to True State: {fidelity_true_vs_noisy_Nq:.6f}")
        print(f"   - Fidelity of PCA-Filtered Pure State to True State: {fidelity_true_vs_filtered_Nq:.6f}")
        if fidelity_true_vs_filtered_Nq > fidelity_true_vs_noisy_Nq:
            print(f"   >>> PCA ADVANTAGE (Signal Recovery): PCA-filtered pure state is CLOSER to original true GHZ state!")
        elif np.isclose(fidelity_true_vs_filtered_Nq, fidelity_true_vs_noisy_Nq):
             print(f"   >>> PCA ADVANTAGE (Comparable Fidelity with Drastic Simplification): Achieves comparable fidelity with a much simpler pure state representation.")
        else:
            print(f"   >>> PCA ADVANTAGE (Simplification & Dominant Feature): Provides BEST PURE STATE approximation of ρ_noisy, simplifying a {D_HILBERT}x{D_HILBERT} mixed matrix.")

if rho_noisy_Nq_analyzed is not None and rho_filtered_pure_approximation_Nq is not None and eigvals_noisy_Nq is not None:
    fidelity_noisy_vs_filtered_Nq = qutip.fidelity(rho_noisy_Nq_analyzed, rho_filtered_pure_approximation_Nq)
    print(f"\n3. Fidelity F(ρ_noisy_{N_QUBITS}q, ρ_filtered_pure_approx_{N_QUBITS}q): {fidelity_noisy_vs_filtered_Nq:.6f}")
    print(f"   (Should be equal to largest eigenvalue of ρ_noisy_{N_QUBITS}q: {eigvals_noisy_Nq[0]:.6f}).")
    print(f"   This shows the PCA-filtered pure state captures {eigvals_noisy_Nq[0]:.2%} of the character of the full noisy state.")


