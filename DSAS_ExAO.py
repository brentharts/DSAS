"""
DSAS & ExAO Physical Optics Propagation (POP) Validation Suite
==============================================================
An end-to-end 2D wave optics simulation modeling atmospheric turbulence,
deformable mirror spatial high-pass filtering, random tip-tilt jitter, 
and a 4f Lyot-cascade Charge-2 Optical Vortex Coronagraph.

This script replaces the old parametric toy-model shortcuts with a full 2D wave optics pipeline: generating stochastic atmospheric phase screens, executing spatial frequency filtering for the Deformable Mirror (DM) loop, injecting random tip-tilt jitter, and passing the aberrated wavefront through a 4f Lyot-cascade coronagraph.Physical Enhancements ImplementedDynamic 2D Phase Screens: Rather than using static analytical equations, the script creates realistic 2D atmospheric phase maps matching a Kolmogorov power spectral density ($\Phi(f) \propto f^{-11/3}$).True Spatial Frequency Control Loop: The DM's correction is modeled directly in the Fourier domain. It applies a high-pass spatial filter matching the exact Nyquist cutoff frequency of the actuator matrix ($f_c = \sqrt{N_{\text{act}}} / 2D$). Inside this control radius, a realistic residual wavefront floor (servo lag/fitting error) is maintained; outside, raw uncorrected atmospheric turbulence dominates.Stochastic Tip-Tilt Injection: Implements a true 2D linear phase ramp using random Gaussian draws scaled to the specified residual jitter ($\sigma_{\text{tt}} = \frac{1}{20}\frac{\lambda}{D}$).4f Vortex Coronagraph Optical Train: Propagates the wave into the focal plane, applies a true 2D Charge-2 Optical Vortex Phase Mask ($e^{i2\theta}$), transforms back to the pupil plane to pass through an optimized Lyot Stop (90% clear aperture), and finally propagates to the science camera where 2D azimuthal radial binning extracts the actual speckled contrast profile.

"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import distance_transform_edt

# ─────────────────────────────────────────────────────────────
# 1. Global Numerical & Optical Architecture Setup
# ─────────────────────────────────────────────────────────────
N = 512               # Grid size of the primary pupil
N_pad = 1024          # Zero-padded array dimension for focal plane sampling (2 pixels per lam/D)
lam = 0.55e-6         # Observing Wavelength [m] (V-band)
D = 4.0               # Primary mirror diameter [m]
r0 = 0.15             # Fried parameter [m] (median seeing baseline)

# Spatial meshes for the Pupil Plane
dx = D / N
x = (np.arange(N) - N / 2) * dx
X, Y = np.meshgrid(x, x)
R_pupil = np.sqrt(X**2 + Y**2)
Theta_pupil = np.arctan2(Y, X)

# Define clear primary aperture and Layer 1 DSAS Apodizer
pupil_mask = (R_pupil <= D / 2).astype(float)
sigma_hg = 0.46 * (D / 2)
n_hg = 8
apod_fabric = np.exp(-(R_pupil / sigma_hg)**n_hg) * pupil_mask

# ─────────────────────────────────────────────────────────────
# 2. Helper Functions: Turbulence, DM Filtering, and Radial Binning
# ─────────────────────────────────────────────────────────────
def generate_kolmogorov_phase(N, dx, r0):
    """Generates a 2D stochastic Kolmogorov phase screen using Fourier filtering."""
    fx = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fx)
    f_rho = np.sqrt(FX**2 + FY**2)
    f_rho[0, 0] = 1e-10  # Shield zero frequency from singularity
    
    # Kolmogorov Phase Power Spectral Density (PSD)
    psd = 0.023 * (r0)**(-5/3) * f_rho**(-11/3)
    
    # White noise randomized injection
    random_amplitude = np.random.normal(size=(N, N)) + 1j * np.random.normal(size=(N, N))
    phase_screen = np.fft.ifft2(random_amplitude * np.sqrt(psd)).real
    # Normalize to zero mean across the clear aperture
    phase_screen -= np.mean(phase_screen[R_pupil <= D / 2])
    return phase_screen

def apply_dm_spatial_filter(phase_screen, N_actuators, dx, D):
    """Models a DM loop by high-pass filtering spatial frequencies below Nyquist."""
    N_linear = np.sqrt(N_actuators)
    f_cutoff = N_linear / (2.0 * D)  # Actuator Nyquist spatial frequency limit
    
    # Move phase screen to frequency domain
    phase_spec = np.fft.fft2(phase_screen)
    fx = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fx)
    f_rho = np.sqrt(FX**2 + FY**2)
    
    # Filter profile: Apply a 99.9% attenuation loop gain inside the control radius,
    # leaving a residual white noise fitting error floor.
    filter_mask = np.ones_like(f_rho)
    filter_mask[f_rho <= f_cutoff] = 0.0316  # -30dB amplitude suppression inside dark hole
    
    filtered_spec = phase_spec * filter_mask
    return np.fft.ifft2(filtered_spec).real

def compute_radial_profile(image_2d, center_px, max_lamD, pix_per_lamD):
    """Extracts a rigorous 1D radial average via azimuthal pixel binning."""
    y, x = np.indices(image_2d.shape)
    r_px = np.sqrt((x - center_px)**2 + (y - center_px)**2)
    r_lamD = r_px / pix_per_lamD
    
    bins = np.linspace(0.1, max_lamD, 150)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    radial_prof = []
    
    for i in range(len(bins)-1):
        mask = (r_lamD >= bins[i]) & (r_lamD < bins[i+1])
        if np.any(mask):
            radial_prof.append(np.mean(image_2d[mask]))
        else:
            radial_prof.append(1e-12)
            
    return bin_centers, np.array(radial_prof)

# ─────────────────────────────────────────────────────────────
# 3. End-to-End Coronagraphic Propagation Pipeline
# ─────────────────────────────────────────────────────────────
def propagate_cascade(N_actuators, use_dsas=True):
    # Step A: Generate pristine atmospheric turbulence phase screen
    raw_atmosphere = generate_kolmogorov_phase(N, dx, r0)
    
    # Step B: Apply DM wavefront correction
    corrected_phase = apply_dm_spatial_filter(raw_atmosphere, N_actuators, dx, D)
    
    # Step C: Inject stochastic tip-tilt jitter as a 2D phase ramp
    sigma_tt_rad = (1.0 / 20.0) * (lam / D)  # Jitter budget: 1/20th lambda/D
    tx, ty = np.random.normal(0, sigma_tt_rad, 2)
    phase_ramp = (2 * np.pi / lam) * (tx * X + ty * Y)
    total_phase = corrected_phase + phase_ramp
    
    # Step D: Construct complex entrance pupil field
    amplitude = apod_fabric if use_dsas else pupil_mask
    E_pupil = amplitude * np.exp(1j * total_phase) * pupil_mask
    
    # Step E: Zero-pad and propagate to Focal Plane 1 (FFT)
    E_padded = np.zeros((N_pad, N_pad), dtype=complex)
    E_padded[N_pad//4 : N_pad//4 + N, N_pad//4 : N_pad//4 + N] = E_pupil
    E_focal1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_padded)))
    
    # Step F: Apply 2D Charge-2 Optical Vortex Phase Mask (OVPM)
    fx_focal = (np.arange(N_pad) - N_pad / 2)
    FX_focal, FY_focal = np.meshgrid(fx_focal, fx_focal)
    Theta_focal = np.arctan2(FY_focal, FX_focal)
    E_vortex = E_focal1 * np.exp(1j * 2.0 * Theta_focal)
    
    # Step G: Propagate back to Pupil Plane 2 / Lyot Plane (IFFT)
    E_lyot_plane = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E_vortex)))
    
    # Step H: Apply an optimized Lyot Stop (90% clear diameter linear downscale)
    Lyot_mask = (np.sqrt(FX_focal**2 + FY_focal**2) <= (N / 2) * 0.90).astype(float)
    E_lyot_post = E_lyot_plane * Lyot_mask
    
    # Step I: Final propagation to the Science Focal Plane (FFT)
    E_science = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_lyot_post)))
    intensity_science = np.abs(E_science)**2
    
    return intensity_science

# ─────────────────────────────────────────────────────────────
# 4. Calibration & Execution of Simulation Configurations
# ─────────────────────────────────────────────────────────────
# Establish absolute peak intensity normalization baseline using an ideal pupil train
E_ideal_padded = np.zeros((N_pad, N_pad), dtype=complex)
E_ideal_padded[N_pad//4 : N_pad//4 + N, N_pad//4 : N_pad//4 + N] = pupil_mask
ideal_focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_ideal_padded)))
airy_peak = np.abs(ideal_focal).max()**2

# Execute the wave optics simulation runs
print("Running POP Pipeline Configuration 1: Baseline (Nact=100)...")
img_baseline = propagate_cascade(N_actuators=100, use_dsas=True) / airy_peak

print("Running POP Pipeline Configuration 2: High-Order (Nact=40000)...")
img_highorder = propagate_cascade(N_actuators=40000, use_dsas=True) / airy_peak

# Compute 1D profiles via azimuthal radial integration
pix_per_lamD = N_pad / N  # Exactly 2 pixels per lambda/D mapping
max_radial_view = 12.0
r_axis, profile_base = compute_radial_profile(img_baseline, N_pad//2, max_radial_view, pix_per_lamD)
_, profile_high = compute_radial_profile(img_highorder, N_pad//2, max_radial_view, pix_per_lamD)

# ─────────────────────────────────────────────────────────────
# 5. Publication-Grade Diagnostic Plot Generation
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6.5), facecolor='white')
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1.4], hspace=0.35, wspace=0.28)

# Panel A: 2D Focal Plane Speckle Field (Baseline System)
ax0 = fig.add_subplot(gs[0, 0])
im0 = ax0.imshow(np.log10(img_baseline + 1e-12), extent=[-max_radial_view, max_radial_view, -max_radial_view, max_radial_view],
                 cmap='inferno', vmin=-8, vmax=-2)
ax0.set_title(r'Baseline Speckle Field ($N_{\mathrm{act}}=100$)', fontsize=10, fontweight='bold')
ax0.set_ylabel(r'Spatial Scale $[\lambda/D]$', fontsize=9)
plt.colorbar(im0, ax=ax0, label=r'$\log_{10}$ Absolute Contrast', fraction=0.046, pad=0.04)

# Panel B: 2D Focal Plane Speckle Field (High-Order System)
ax1 = fig.add_subplot(gs[1, 0])
im1 = ax1.imshow(np.log10(img_highorder + 1e-12), extent=[-max_radial_view, max_radial_view, -max_radial_view, max_radial_view],
                 cmap='inferno', vmin=-11, vmax=-5)
ax1.set_title(r'Cleared Dark Hole ($N_{\mathrm{act}}=40,000$)', fontsize=10, fontweight='bold')
ax1.set_xlabel(r'Spatial Scale $[\lambda/D]$', fontsize=9)
ax1.set_ylabel(r'Spatial Scale $[\lambda/D]$', fontsize=9)
plt.colorbar(im1, ax=ax1, label=r'$\log_{10}$ Absolute Contrast', fraction=0.046, pad=0.04)

# Panels C & D: Pupil Phase Geometry Diagrams
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(generate_kolmogorov_phase(N, dx, r0) * pupil_mask, cmap='bwr')
ax2.set_title('Raw Input Wavefront Phase Screen', fontsize=10, fontweight='bold')
ax2.axis('off')

ax3 = fig.add_subplot(gs[1, 1])
ax3.imshow(apply_dm_spatial_filter(generate_kolmogorov_phase(N, dx, r0), 100, dx, D) * pupil_mask, cmap='bwr')
ax3.set_title('DM High-Pass Filtered Residual Phase', fontsize=10, fontweight='bold')
ax3.axis('off')

# Panel E: Consolidated Radial Average Contrast Curves
ax4 = fig.add_subplot(gs[:, 2])
ax4.semilogy(r_axis, profile_base, color='#ff7f0e', lw=2.2, label=r'Baseline Stack ($N_{\mathrm{act}}=100$)')
ax4.semilogy(r_axis, profile_high, color='#2ca02c', lw=2.2, label=r'High-Order Upgrade ($N_{\mathrm{act}}=40,000$)')
ax4.axhline(1e-10, color='purple', lw=1.5, ls='-.', label='Earth-Analog Detection Target')
ax4.fill_between(r_axis, 1e-12, 1e-10, color='purple', alpha=0.04, label='Habitable Search Window')

ax4.set_xlim(1.0, max_radial_view)
ax4.set_ylim(1e-12, 1e-2)
ax4.set_xlabel(r'Inner Working Angle (IWA) $[\lambda/D]$', fontsize=10)
ax4.set_ylabel('Azimuthally Binned Raw Contrast', fontsize=10)
ax4.set_title('Physical Optics Performance Profile', fontsize=11, fontweight='bold', pad=10)
ax4.legend(fontsize=8.5, loc='upper right')
ax4.grid(True, which='both', alpha=0.2)

# Convert top axis coordinates to milliarcseconds for immediate parsing
scale_arcsec = (lam / D) * 206265
ax4_top = ax4.twiny()
ax4_top.set_xlim(ax4.get_xlim())
tick_locations = np.array([2, 4, 6, 8, 10, 12])
ax4_top.set_xticks(tick_locations)
ax4_top.set_xticklabels([f'{v*scale_arcsec*1000:.0f}' for v in tick_locations], fontsize=8)
ax4_top.set_xlabel(r'Angular Separation [mas]', fontsize=9)

fig.savefig('./pop_cascade_verification.png', dpi=300, bbox_inches='tight')

# ─────────────────────────────────────────────────────────────
# 6. High-Fidelity Data Extraction and Terminal Presentation
# ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("       PHYSICAL OPTICS PROPAGATION (POP) CRITICAL DATA METRICS PACK")
print("="*80)
print(f"Operational Parameters:")
print(f"  - Grid Dimension (Zero-padded Primary)        : {N}x{N} ({N_pad}x{N_pad})")
print(f"  - Jitter Budget Vector Boundary (sigma_tt)    : 0.05 λ/D (Stochastic Phase Ramp)")
print(f"  - Nyquist Spatial Cutoff Limit (Nact=100)     : {np.sqrt(100)/2:.1f} λ/D")
print(f"  - Nyquist Spatial Cutoff Limit (Nact=40,000)  : {np.sqrt(40000)/2:.1f} λ/D")
print(f"  - Pixel Resolution Plate Scale Factor         : 1 λ/D = {scale_arcsec*1000:.1f} mas")

print("\n── High-Contrast Spatial Scaling Vectors (Data Analysis Ready) ──")
print(f"{'Separation (IWA)':<20} | {'Baseline Profile (Nact=100)':<28} | {'High-Order Profile (Nact=40k)':<25}")
print("-"*80)
for target_iwa in [2.0, 3.0, 4.0, 5.0, 8.0]:
    val_base = float(np.interp(target_iwa, r_axis, profile_base))
    val_high = float(np.interp(target_iwa, r_axis, profile_high))
    print(f"  IWA = {target_iwa:.1f} λ/D         |    {val_base:.3e}               |    {val_high:.3e}")
print("="*80 + "\n")
