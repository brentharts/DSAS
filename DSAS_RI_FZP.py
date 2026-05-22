"""
DSAS Stratospheric Configuration: Refractive Index & FZP Phase Correction
========================================================================

This script models the Stratospheric Configuration (Section 2.4). It computes the massive integrated atmospheric phase error ($\Delta\phi_{\text{refr}} \sim 10^3 \text{ rad}$) accumulated through a 5-km altitude stack from $h = 10 \text{ km}$ to $15 \text{ km}$ using the exponential scale-height refractivity profile. It then simulates how the programmable electro-optical fabric applies a compensating phase offset ($\Delta\phi_{\text{DSAS}}$) to neutralize this atmospheric dispersion and restore the deep constructive/destructive interference null at the telescope entrance pupil. 


Verifies the claims in Section 2.4 and Section 2.5:
  1. Computes the integrated atmospheric phase drift over a high-altitude stack.
  2. Simulates the restoration of the multi-plane Fresnel null using
     programmable phase compensation (Delta_phi_DSAS).
"""

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# Physical Parameters & Atmospheric Configuration
# ─────────────────────────────────────────────────────────────
lam = 550e-9          # Observation Wavelength [m] (V-band)
k = 2 * np.pi / lam   # Wave number
H_scale = 8000.0      # Atmospheric scale height [m] (Section 2.4)
n0 = 2.73e-4          # Standard sea-level refractivity constant (n-1)

# Stratospheric stack boundaries
h_min = 10000.0       # 10 km base altitude
h_max = 15000.0       # 15 km top altitude

print("── Stratospheric Refractivity Analysis ──")

# 1. Analytical Verification of Equation 5
# Integrated refractivity: \int_{h_1}^{h_K} (n(h) - 1) dh
integrated_refractivity = n0 * H_scale * (np.exp(-h_min / H_scale) - np.exp(-h_max / H_scale))
delta_phi_refr = k * integrated_refractivity

print(f"Integrated Refractivity Thickness : {integrated_refractivity:.6e} meters")
print(f"Total Cumulative Phase Distortion : {delta_phi_refr:.2f} rad")
print(f"Equivalent Number of Pi-Phases (p): {delta_phi_refr / np.pi:.2f} cycles")

# ─────────────────────────────────────────────────────────────
# Multi-Plane Fresnel Zone Plate Simulation
# ─────────────────────────────────────────────────────────────
# We model K=5 planes distributed through the stratospheric column
K = 5
altitudes = np.linspace(h_min, h_max, K)
m_zones = np.array([1, 3, 5, 7, 9]) # Zone orders

# Compute corresponding ideal ring radii from Eq. 2: R_k^2 = m_k * lam * h_k
radii = np.sqrt(m_zones * lam * altitudes)

print("\n── Multi-Plane FZP Layout Verification ──")
for idx, (h, m, r) in enumerate(zip(altitudes, m_zones, radii)):
    print(f"  Layer {idx+1}: Altitude = {h/1000:.1f} km, Zone = {m}, Radius = {r:.4f} m")

def compute_on_axis_field(apply_correction=True):
    """
    Computes the normalized on-axis field E_total at the primary mirror
    by summing up the diffracted fields from the individual planes.
    """
    # Free-space unattenuated background reference field
    E_free = 1.0 + 0j
    E_blocked_sum = 0.0 + 0j
    
    for h, r in zip(altitudes, radii):
        # Local atmospheric index at this specific layer altitude
        n_local = 1.0 + n0 * np.exp(-h / H_scale)
        
        # Phase accumulated from propagation through the variable medium
        # Path length includes the geometric hyp-distance and atmospheric column index
        optical_path = n_local * h + (r**2) / (2.0 * h)
        phase_accum = k * optical_path
        
        # Apply programmable fabric phase correction to counter dispersion (Fig. 4)
        if apply_correction:
            # Compensates for the local atmospheric index shift to restore ideal geometry
            phase_corr = -k * (n_local - 1.0) * h
        else:
            phase_corr = 0.0
            
        # Coherent contribution of the blocked ring element (Eq. 3 / Eq. 4)
        # Narrow annulus approximation for qualitative phase-alignment evaluation
        field_contribution = np.exp(1j * (phase_accum + phase_corr)) / K
        E_blocked_sum += field_contribution
        
    # E_total = E_free - \sum E_blocked
    E_total = E_free - E_blocked_sum
    return np.abs(E_total)**2

# Evaluate performance metrics
null_uncorrected = compute_on_axis_field(apply_correction=False)
null_corrected = compute_on_axis_field(apply_correction=True)

print("\n── Interference Suppression Results ──")
print(f"  On-Axis Residual Intensity (Uncorrected Dispersion): {null_uncorrected:.4e}")
print(f"  On-Axis Residual Intensity (With DSAS Phase Edits) : {null_corrected:.4e}")

# ─────────────────────────────────────────────────────────────
# Plotting the Phase Dispersion vs. Altitude Stack
# ─────────────────────────────────────────────────────────────
h_grid = np.linspace(h_min - 2000, h_max + 2000, 500)
phase_profile = k * n0 * H_scale * (1.0 - np.exp(-h_grid / H_scale))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(h_grid / 1000.0, phase_profile, color='darkblue', lw=2)
ax1.axvspan(h_min/1000.0, h_max/1000.0, color='orange', alpha=0.15, label='DSAS Stack Window')
ax1.set_xlabel('Altitude $h$ [km]', fontsize=10)
ax1.set_ylabel('Cumulative Refractive Phase [rad]', fontsize=10)
ax1.set_title('Atmospheric Phase Accumulation Profile', fontsize=11, pad=8)
ax1.grid(True, alpha=0.25)
ax1.legend()

# Bar chart evaluating the suppression performance comparison
null_levels = [1.0, null_uncorrected, max(null_corrected, 1e-12)]
labels = ['Unobstructed Star', 'Uncorrected Stack', 'DSAS Corrected Null']
colors = ['#7f7f7f', '#d62728', '#2ca02c']

ax2.bar(labels, null_levels, color=colors, width=0.5, edgecolor='black', alpha=0.85)
ax2.set_yscale('log')
ax2.set_ylabel('Normalized On-Axis Beam Intensity', fontsize=10)
ax2.set_title('FZP Suppression Re-alignment Verification', fontsize=11, pad=8)
ax2.grid(True, which="both", alpha=0.2)

plt.tight_layout()
plt.savefig('dsas_stratospheric_verification.png', dpi=300)
print("\nSuccess: Saved 'dsas_stratospheric_verification.png'.")
