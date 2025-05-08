import matplotlib.pyplot as plt
import numpy as np

shear_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
viscosities = [285786.44, 244599.75, 231515.09, 94315.10 , 82497.62]

plt.figure(figsize=(8, 6))
plt.loglog(shear_rates, viscosities, 'o-', label='Simulated Viscosity')
plt.xlabel("Shear Rate (γ̇)")
plt.ylabel("Shear Viscosity (η)")
plt.title("Viscosity vs Shear Rate")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.savefig("viscosity_vs_shear_rate.png", dpi=300)
plt.show()
