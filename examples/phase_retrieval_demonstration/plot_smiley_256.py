"""Plot the saved Gaussian exit wave directly on a 256 x 256 specimen grid."""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from temgym_core.gaussian import make_gaussian
from temgym_core.evaluate import evaluate_gaussians

folder = Path(__file__).resolve().parent
data = np.load(folder / 'dataset.npz')
calibration = json.loads((folder / 'calibration.json').read_text())
size = 256
width = data['specimen_truth'].shape[1] * float(data['specimen_pitch_m'])
pitch = width / size
axis = (np.arange(size) - (size - 1) / 2) * pitch
x, y = np.meshgrid(axis, axis)
grid = SimpleNamespace(shape=(size, size), coords=jnp.asarray(np.c_[x.ravel(), y.ravel()]))
centres = data['gaussian_centres_m']
waist = float(data['gaussian_waist_m'])
beam = make_gaussian(x=centres[:, 0], y=centres[:, 1], z=0.,
                     voltage=calibration['voltage_V'], waist_x=waist, waist_y=waist,
                     amp=data['truth_coefficients'] / np.pi)
wave = np.asarray(evaluate_gaussians(beam, grid, method='scan'))
amplitude, phase = np.abs(wave), np.angle(wave)
assert wave.shape == (256, 256) and np.isfinite(wave).all()
out = folder / 'figures'
np.savez_compressed(folder / 'smiley_256.npz', exit_wave=wave, amplitude=amplitude,
                    phase_rad=phase, pixel_size_m=pitch)
plt.imsave(out / 'smiley_amplitude_256.png', amplitude, cmap='gray', vmin=0, vmax=1, origin='lower')
plt.imsave(out / 'smiley_phase_256.png', phase, cmap='viridis', vmin=0, vmax=1.2, origin='lower')
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout='constrained')
extent = np.array([-1, 1, -1, 1]) * width * 1e9 / 2
for ax, values, title, cmap, limit, label in zip(
        axes, [amplitude, phase], ['Exit-wave amplitude', 'Exit-wave phase'],
        ['gray', 'viridis'], [1, 1.2], ['Amplitude (a.u.)', 'Phase (rad)']):
    im = ax.imshow(values, origin='lower', extent=extent, cmap=cmap,
                   vmin=0, vmax=limit, interpolation='nearest')
    ax.set(title=title, xlabel='x (nm)', ylabel='y (nm)')
    fig.colorbar(im, ax=ax, shrink=.86, label=label)
fig.suptitle('Smiley at the specimen · 256 × 256 samples', fontsize=15)
fig.savefig(out / 'smiley_amplitude_phase_256.png', dpi=180)
fig.savefig(out / 'smiley_amplitude_phase_256.pdf')
print(f'Saved amplitude and phase: {wave.shape}, {pitch*1e9:.2f} nm/pixel.')
