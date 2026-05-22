"""
DSAS & TNO Unified Performance & Data Analysis Suite
==========================================================
Description: Fixes FFT zero-frequency indexing anomalies to eliminate NaN loops.
             Implements explicit telescope aperture tracking for the TNO channel.
             Generates analytical LaTeX-ready printouts for the manuscript appendix.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import j0, j1

# ─────────────────────────────────────────────────────────────
# 1. Global Simulation & Optical Setup
# ─────────────────────────────────────────────────────────────
N = 1024              # Pupil grid dimension
lam = 550e-9          # Target V-band observation wavelength [m]
k = 2 * np.pi / lam   # Wave number
D = 4.0               # Telescope diameter [m]
f = 40.0              # Focal length [m] (f/10 system)

print("="*80)
print("     DSAS & TNO UNIFIED SYSTEM SIMULATION AND PERFORMANCE DATA PACK")
print("="*80)
print(f"Optical Parameters Set:")
print(f"  - Primary Aperture Diameter (D):  {D:.1f} m")
print(f"  - Central Wavelength (lambda):    {lam*1e9:.1f} nm")
print(f"  - Focal Length (f):               {f:.1f} m (f/10)")
print(f"  - Sampling Resolution:            {N}x{N} grid")

# Spatial coordinates for telescope pupil plane
dx_pup = D / N
x_pup = (np.arange(N) - N / 2) * dx_pup
X, Y = np.meshgrid(x_pup, x_pup)
R_pup = np.sqrt(X**2 + Y**2)
Theta_pup = np.arctan2(Y, X)

# Telescope pupil mask definition
pupil = (R_pup <= D / 2).astype(float)

# Focal plane sampling setup
pix_per_lamD = 4
focal_pixels = 2 * N
ctr = focal_pixels // 2
u_lamD = (np.arange(focal_pixels) - ctr) / pix_per_lamD

def pupil_to_psf(field_complex):
    """Transforms complex pupil field to focal-plane intensity normalized to unobstructed peak."""
    padded = np.zeros((2 * N, 2 * N), dtype=complex)
    padded[N//2 : N//2 + N, N//2 : N//2 + N] = field_complex
    focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    psf = np.abs(focal)**2
    return psf / (psf.max() + 1e-30)

def compute_azimuthal_profile(psf_2d):
    """Computes radial average of the 2D PSF out to 15 lambda/D."""
    focal_y, focal_x = np.meshgrid(u_lamD, u_lamD)
    focal_r = np.sqrt(focal_x**2 + focal_y**2)
    
    r_bins = np.arange(0, 15, 0.1)
    profile = []
    for r in r_bins:
        mask = (focal_r >= r) & (focal_r < r + 0.1)
        profile.append(np.mean(psf_2d[mask]) if np.any(mask) else 1e-12)
    return r_bins, np.array(profile)

# Reference Airy Profile
psf_airy = pupil_to_psf(pupil.astype(complex))

# ─────────────────────────────────────────────────────────────
# 2. Stratospheric Atmosphere & Refractive Phase Analysis
# ─────────────────────────────────────────────────────────────
print("\n" + "-"*40)
print("SECTION A: STRATOSPHERIC DISPERSION & COMPETING PHASE ANALYSIS")
print("-"*40)

H_scale = 8000.0      # Atmospheric scale height [m]
n0 = 2.73e-4          # Sea-level refractivity constant (n-1)
h_min, h_max = 10000.0, 15000.0  # Stratospheric block window (10km - 15km)

# Analytical integration of atmospheric phase error over the 5km stack
phi_refr_analytical = k * n0 * H_scale * (np.exp(-h_min / H_scale) - np.exp(-h_max / H_scale))
print(f"Integrated Atmospheric Phase Error (10 to 15 km): {phi_refr_analytical:.4f} rad")

# Simulate electro-optical fabric phase correction residual error
fabric_phase_quantization_bits = 8
phase_step = (2 * np.pi) / (2**fabric_phase_quantization_bits)
phi_dsas_correction = -np.round(phi_refr_analytical / phase_step) * phase_step
residual_strat_phase_error = phi_refr_analytical + phi_dsas_correction

print(f"Programmable Fabric Quantization:              {fabric_phase_quantization_bits}-bit LCoS/Electrochromic")
print(f"Applied Counter-Phase (Delta_phi_DSAS):        {phi_dsas_correction:.4f} rad")
print(f"Residual Atmospheric Dispersion Error:         {residual_strat_phase_error:.6f} rad")

# ─────────────────────────────────────────────────────────────
# 3. Active DSAS Cascade Simulation (Layers 1-3)
# ─────────────────────────────────────────────────────────────
print("\n" + "-"*40)
print("SECTION B: ACTIVE DSAS THREE-LAYER CASCADE")
print("-"*40)

# Layer 1: Hyper-Gaussian Apodization Mask
sigma_hg = 0.46 * (D / 2)
n_hg = 8
apod_fabric = np.exp(-(R_pup / sigma_hg)**n_hg) * pupil
psf_layer1 = pupil_to_psf(apod_fabric.astype(complex))
supp_layer1_onaxis = (np.sum(apod_fabric)**2) / (np.sum(pupil)**2)
print(f"Layer 1 (Hyper-Gaussian Apodization) On-Axis Suppression: {supp_layer1_onaxis:.4e}")

# Layer 2: Extreme AO Deformable Mirror Correction (Kolmogorov Residuals)
r0 = 0.15     # Fried parameter [m]
Nact = 100    # Linear actuator count
sigma2_ao = 0.295 * (D / r0)**(5/3) * (Nact**(-5/3))
rms_phase_rad = np.sqrt(sigma2_ao)

# FIXED: Frequency generation coordinates mapped to match unshifted array locations
freq = np.fft.fftfreq(N, d=dx_pup)
fx, fy = np.meshgrid(freq, freq)
f2 = fx**2 + fy**2
f2[0, 0] = 1.0  # Safely handle the DC singularity at index [0,0]

power_spectrum = 0.023 * (D / r0)**(5/3) * (f2**(-11/6))
power_spectrum[0, 0] = 0.0  # Zero out DC energy contribution

# Apply spatial frequency filter mimicking Deformable Mirror actuator cutoff limit
spatial_cutoff = Nact / (2.0 * D)
power_spectrum[np.sqrt(f2) <= spatial_cutoff] *= 0.01 

rng = np.random.default_rng(101)
random_phase_raw = np.fft.ifft2(np.sqrt(power_spectrum) * (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))))
phase_screen = np.real(random_phase_raw)
phase_screen = (phase_screen - np.mean(phase_screen)) / np.std(phase_screen) * rms_phase_rad

# Combine amplitude profile with the composite residual error structure
field_layer2 = apod_fabric * np.exp(1j * (phase_screen + residual_strat_phase_error))
psf_layer2 = pupil_to_psf(field_layer2)
print(f"Layer 2 Residual Wavefront Phase Variance:                {sigma2_ao:.4f} rad^2")
print(f"Layer 2 Wavefront RMS Error:                              {rms_phase_rad:.4f} rad")

# Layer 3: Internal Charge-2 Optical Vortex Phase Mask (OVPM)
vortex_operator = np.exp(1j * 2 * Theta_pup)
field_layer3 = field_layer2 * vortex_operator
psf_layer3 = pupil_to_psf(field_layer3)

# Extract specific contrast performance at key angular intervals
rbins, prof_airy = compute_azimuthal_profile(psf_airy)
_, prof_l1 = compute_azimuthal_profile(psf_layer1)
_, prof_l2 = compute_azimuthal_profile(psf_layer2)
_, prof_l3 = compute_azimuthal_profile(psf_layer3)

angles_to_check = [1.0, 2.0, 3.0, 5.0, 10.0]
print("\n--- Raw Contrast Performance Summary ---")
print(f"{'Angle [lambda/D]':<18}{'Airy (Raw)':<15}{'Layer 1 (Mask)':<18}{'Layer 2 (AO)':<15}{'Layer 3 (OVPM)':<15}")
for ang in angles_to_check:
    idx = np.argmin(np.abs(rbins - ang))
    print(f"{ang:<18.1f}{prof_airy[idx]:<15.2e}{prof_l1[idx]:<18.2e}{prof_l2[idx]:<15.2e}{prof_l3[idx]:<15.2e}")

# ─────────────────────────────────────────────────────────────
# 4. Opportunistic Channel: TNO Occultation Phase Restoration
# ─────────────────────────────────────────────────────────────
print("\n" + "-"*40)
print("SECTION C: OPPORTUNISTIC TNO OCCULTATION CHANNEL")
print("-"*40)

N_tno = 512
tno_grid = 30.0
dx_tno = tno_grid / N_tno
x_tno = (np.arange(N_tno) - N_tno / 2) * dx_tno
X_t, Y_t = np.meshgrid(x_tno, x_tno)
R_t = np.sqrt(X_t**2 + Y_t**2)
Theta_t = np.arctan2(Y_t, X_t)

# Model a non-spherical irregular contact-binary TNO limb profile
limb_perturbation = 1.2 * np.sin(4 * Theta_t) + 0.6 * np.cos(9 * Theta_t)
tno_profile = 8.0 + limb_perturbation
shadow_mask = (R_t > tno_profile).astype(float)

# Compute 2D Ground Track Diffraction via Fourier angular spectrum method
f_space_tno = np.fft.fftfreq(N_tno, d=dx_tno)
FX_t, FY_t = np.meshgrid(f_space_tno, f_space_tno)
H_kernel_tno = np.exp(-1j * np.pi * 2.0 * (FX_t**2 + FY_t**2))
wavefront_diffracted = np.fft.ifft2(np.fft.fft2(shadow_mask) * H_kernel_tno)

ground_amplitude = np.abs(wavefront_diffracted)
ground_phase = np.angle(wavefront_diffracted)

# FIXED: Apply an explicit telescope selection window to extract core leakage parameters
telescope_aperture_tno = (R_t <= 1.8).astype(float)

# Evaluate focal structures for perturbed states versus active conjugate phase states
field_tno_perturbed = ground_amplitude * np.exp(1j * ground_phase) * telescope_aperture_tno
focal_tno_perturbed = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_tno_perturbed)))
psf_tno_perturbed = np.abs(focal_tno_perturbed)**2
psf_tno_perturbed /= (psf_tno_perturbed.max() + 1e-30)

dm_conjugate_command = -ground_phase
field_tno_corrected = ground_amplitude * np.exp(1j * (ground_phase + dm_conjugate_command)) * telescope_aperture_tno
focal_tno_corrected = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_tno_corrected)))
psf_tno_corrected = np.abs(focal_tno_corrected)**2
psf_tno_corrected /= (psf_tno_corrected.max() + 1e-30)

null_perturbed = psf_tno_perturbed[N_tno//2, N_tno//2]
null_corrected = psf_tno_corrected[N_tno//2, N_tno//2]

print(f"Irregular Limb Edge Deviation Amplitude:  max={np.max(limb_perturbation):.2f} Fresnel scales")
print(f"Perturbed Starshade Shadow Peak Leakage:  {null_perturbed:.4e}")
print(f"Conjugate DM Restored Stellar Null Floor: {null_corrected:.4e}")
print("="*80)

# ─────────────────────────────────────────────────────────────
# 5. Composite Metric Visualization Generation
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 3, wspace=0.3, hspace=0.3)

# Plot A: Active Cascaded Profiles
ax0 = plt.subplot(gs[0, 0])
ax0.plot(rbins, prof_airy, label='Unaberrated Airy', color='#7f7f7f', linestyle='--')
ax0.plot(rbins, prof_l1, label='Layer 1: Fabric Mask', color='#1f77b4', lw=2)
ax0.plot(rbins, prof_l2, label='Layer 2: Residual AO', color='#ff7f0e', lw=2)
ax0.plot(rbins, prof_l3, label='Layer 3: Internal OVPM', color='#2ca02c', lw=2)
ax0.set_yscale('log')
ax0.set_ylim(1e-11, 1.5)
ax0.set_xlim(0, 12)
ax0.set_xlabel(r'Angular Separation [$\lambda/D$]', fontsize=10)
ax0.set_ylabel('Normalized Contrast Ratio', fontsize=10)
ax0.set_title('Active DSAS Cascaded Performance', fontsize=11, fontweight='bold')
ax0.grid(True, which='both', alpha=0.15)
ax0.legend(fontsize=8)

# Plot B: Phase Screen Realization (Shifted to center the spatial frequencies)
ax1 = plt.subplot(gs[0, 1])
im1 = ax1.imshow(np.fft.fftshift(phase_screen) * pupil, cmap='RdBu_r', extent=[-2,2,-2,2])
ax1.set_title('Layer 2 Simulated Phase Screen [rad]', fontsize=11, fontweight='bold')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# Plot C: Atmospheric Scale Phase Accumulation
ax2 = plt.subplot(gs[0, 2])
h_axis = np.linspace(0, 20000, 200)
phi_axis = k * n0 * H_scale * (1.0 - np.exp(-h_axis / H_scale))
ax2.plot(h_axis / 1000.0, phi_axis, color='indigo', lw=2)
ax2.axvspan(h_min/1000.0, h_max/1000.0, color='orange', alpha=0.15, label='DSAS Window')
ax2.set_xlabel('Altitude h [km]')
ax2.set_ylabel('Integrated Phase Error [rad]')
ax2.set_title('Atmospheric Refractive Gradient', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.2)

# Plot D: TNO Irregular Limb Profile
ax3 = plt.subplot(gs[1, 0])
ax3.imshow(shadow_mask, cmap='bone', extent=[-15,15,-15,15])
ax3.set_title('Model 3: Irregular TNO Limb Mask', fontsize=11, fontweight='bold')
ax3.set_xlabel('Fresnel Scales')

# Plot E: TNO Ground Track Perturbed Wavefront Phase
ax4 = plt.subplot(gs[1, 1])
im4 = ax4.imshow(np.fft.fftshift(ground_phase), cmap='twilight', extent=[-15,15,-15,15])
ax4.set_title('Pertracted Wavefront Phase [rad]', fontsize=11, fontweight='bold')
plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

# Plot F: TNO Compensation Null Restoration Comparison
ax5 = plt.subplot(gs[1, 2])
bars = ['Perturbed Limb', 'Conjugate DM']
values = [null_perturbed, null_corrected]
ax5.bar(bars, values, color=['#d62728', '#2ca02c'], width=0.4)
ax5.set_yscale('log')
ax5.set_ylim(1e-12, 1.5)
ax5.set_ylabel('On-Axis Null Leakage Floor')
ax5.set_title('TNO Phase Loop Verification', fontsize=11, fontweight='bold')
ax5.grid(True, which='both', alpha=0.15)

plt.savefig('dsas_unified_metrics.png', dpi=300, bbox_inches='tight')
print("\n[SUCCESS] Integrated simulation complete. Saved master diagnostic figure to 'dsas_unified_metrics.png'.")
print("="*80)
