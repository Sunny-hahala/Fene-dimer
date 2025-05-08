import numpy as np
import matplotlib.pyplot as plt
import os;

print("="*80); 
script_basenames = ['correlation_0_01','correlation_0_001', 'correlation_0_0001', 'correlation_0_00001']
output_basenames = ['several_polymer_y','Several_y_0_05','Several_y_0_01', 'Several_y_0_005', 'Several_y_0_001']
Shear_rate = [0.1, 0.05, 0.01, 0.005, 0.001]


TIMESTEP_INTERVAL = 0.01
DISCARD_FRACTION = 0.0

for i, output_base in enumerate(output_basenames):
    run_name = f'test_00{i}'
    base_dir = f'./output/{output_base}/{run_name}'

    print("base_dir = " + base_dir)
    os.makedirs(base_dir, exist_ok=True)

    data_filename = f'output/{output_base}/harmonic_test000/polymer_xz_stress_output.txt'
    print("data_filename = " + data_filename)

    stress = np.loadtxt(data_filename, comments="#")
    n_total = len(stress)

    # === DISCARD INITIAL TRANSIENTS ===
    start_index = int(DISCARD_FRACTION * n_total)
    stress_trimmed = stress[start_index:]
    time_trimmed = np.arange(start_index, n_total) * TIMESTEP_INTERVAL

    # === COMPUTE VISCOSITY OVER TIME ===
    cumulative_avg_stress = np.cumsum(stress_trimmed) / np.arange(1, len(stress_trimmed) + 1)
    viscosity_over_time = np.abs(cumulative_avg_stress) / Shear_rate[i]

    # === COMPUTE FINAL STABLE VISCOSITY VALUE ===
    final_window_fraction = 0.3  # average over last 30%
    final_start = int((1 - final_window_fraction) * len(viscosity_over_time))
    eta_final = np.mean(viscosity_over_time[final_start:])
    eta_std = np.std(viscosity_over_time[final_start:])

    # === PLOT WITH MEAN AND BAND ===
    plt.figure(figsize=(10, 6))
    plt.plot(time_trimmed, viscosity_over_time, color='red', label='Shear Viscosity (|avg| / γ̇)')
    plt.axhline(eta_final, color='blue', linestyle='--', label=f'Mean η ≈ {eta_final:.2f}')
    plt.fill_between(time_trimmed[final_start:], eta_final - eta_std, eta_final + eta_std,
                     color='blue', alpha=0.2, label='±1 Std Dev')
    plt.xlabel("Time")
    plt.ylabel("Shear Viscosity η")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_name}_viscosity_vs_time.png", dpi=300)
    plt.show()
