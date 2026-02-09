"""
Simplified forward model using Collins FFT for rotation, magnification, and defocus.

This uses a fast FFT-based approach instead of Gaussian beams for efficiency.
"""

import sys
sys.path.insert(0, '../../src')

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import pickle
from pathlib import Path

jax.config.update("jax_enable_x64", True)

# Physical constants
VOLTAGE = 200e3  # 200 kV
E_CHARGE = 1.602176634e-19
M_E = 9.1093837015e-31
C_LIGHT = 299792458.0


def energy2wavelength(voltage):
    """Calculate electron wavelength from accelerating voltage."""
    gamma = 1 + E_CHARGE * voltage / (M_E * C_LIGHT**2)
    beta = np.sqrt(1 - 1/gamma**2)
    return 1.226e-9 / np.sqrt(voltage / 1000) / np.sqrt(1 + voltage / 1e6)


def create_square_aperture(size: int, aperture_fraction: float = 0.3) -> jnp.ndarray:
    """
    Create a square aperture in the center of the field.
    
    Parameters
    ----------
    size : int
        Grid size (pixels)
    aperture_fraction : float
        Fraction of grid size for aperture
        
    Returns
    -------
    aperture : jnp.ndarray
        Binary aperture mask
    """
    aperture = jnp.zeros((size, size))
    center = size // 2
    half_ap = int(size * aperture_fraction / 2)
    
    # Create square aperture
    aperture = aperture.at[
        center - half_ap:center + half_ap,
        center - half_ap:center + half_ap
    ].set(1.0)
    
    return aperture


def rotate_field_fft(field: jnp.ndarray, angle: float) -> jnp.ndarray:
    """
    Rotate field using Fourier-domain shearing.
    
    Parameters
    ----------
    field : jnp.ndarray
        Input field
    angle : float
        Rotation angle (radians)
        
    Returns
    -------
    rotated_field : jnp.ndarray
        Rotated field
    """
    if abs(angle) < 1e-10:
        return field
    
    ny, nx = field.shape
    
    # Create coordinate grids
    y = jnp.arange(ny) - ny // 2
    x = jnp.arange(nx) - nx // 2
    Y, X = jnp.meshgrid(y, x, indexing='ij')
    
    # Rotation matrix
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    
    # Rotated coordinates (inverse rotation for sampling)
    X_rot = cos_a * X + sin_a * Y
    Y_rot = -sin_a * X + cos_a * Y
    
    # Use Fourier interpolation for smooth rotation
    # For simplicity, use nearest neighbor resampling
    X_idx = jnp.round(X_rot + nx // 2).astype(int)
    Y_idx = jnp.round(Y_rot + ny // 2).astype(int)
    
    # Clip to valid range
    X_idx = jnp.clip(X_idx, 0, nx - 1)
    Y_idx = jnp.clip(Y_idx, 0, ny - 1)
    
    # Resample
    rotated_field = field[Y_idx, X_idx]
    
    return rotated_field


def scale_field_fourier(field: jnp.ndarray, scale: float) -> jnp.ndarray:
    """
    Scale field by magnification factor using Fourier padding/cropping.
    
    Parameters
    ----------
    field : jnp.ndarray
        Input field
    scale : float
        Scaling factor (>1 magnifies, <1 shrinks)
        
    Returns
    -------
    scaled_field : jnp.ndarray
        Scaled field
    """
    if abs(scale - 1.0) < 1e-10:
        return field
    
    ny, nx = field.shape
    
    # Fourier transform
    F = jnp.fft.fftshift(jnp.fft.fft2(field))
    
    # Create new grid
    new_ny, new_nx = int(ny / scale), int(nx / scale)
    
    if scale > 1.0:
        # Magnify: crop Fourier space
        cy, cx = ny // 2, nx // 2
        hy, hx = new_ny // 2, new_nx // 2
        F_cropped = F[cy - hy:cy + hy, cx - hx:cx + hx]
        scaled = jnp.fft.ifft2(jnp.fft.ifftshift(F_cropped))
        
        # Pad to original size
        pad_y = (ny - new_ny) // 2
        pad_x = (nx - new_nx) // 2
        scaled_field = jnp.pad(scaled, ((pad_y, ny - new_ny - pad_y), (pad_x, nx - new_nx - pad_x)))
    else:
        # Shrink: pad Fourier space
        pad_y = (new_ny - ny) // 2
        pad_x = (new_nx - nx) // 2
        F_padded = jnp.pad(F, ((pad_y, new_ny - ny - pad_y), (pad_x, new_nx - nx - pad_x)))
        scaled = jnp.fft.ifft2(jnp.fft.ifftshift(F_padded))
        
        # Crop to original size
        cy, cx = new_ny // 2, new_nx // 2
        hy, hx = ny // 2, nx // 2
        scaled_field = scaled[cy - hy:cy + hy, cx - hx:cx + hx]
    
    return scaled_field


def fresnel_propagate(
    field: jnp.ndarray,
    wavelength: float,
    defocus: float,
    pixel_size: float
) -> jnp.ndarray:
    """
    Fresnel propagation using FFT.
    
    Parameters
    ----------
    field : jnp.ndarray
        Input field
    wavelength : float
        Wavelength (metres)
    defocus : float
        Propagation distance (metres)
    pixel_size : float
        Pixel size (metres)
        
    Returns
    -------
    propagated_field : jnp.ndarray
        Propagated field
    """
    ny, nx = field.shape
    
    # Frequency coordinates
    fx = jnp.fft.fftfreq(nx, d=pixel_size)
    fy = jnp.fft.fftfreq(ny, d=pixel_size)
    FX, FY = jnp.meshgrid(fx, fy)
    
    # Transfer function
    H = jnp.exp(-1j * jnp.pi * wavelength * defocus * (FX**2 + FY**2))
    
    # Propagate
    F = jnp.fft.fft2(field)
    F_prop = H * F
    propagated = jnp.fft.ifft2(F_prop)
    
    # Add on-axis phase
    propagated *= jnp.exp(1j * 2 * jnp.pi * defocus / wavelength)
    
    return propagated


def collins_forward_model(
    input_aperture: jnp.ndarray,
    wavelength: float,
    defocus: float,
    magnification: float,
    rotation: float,
    pixel_size: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Forward model using Collins FFT approach.
    
    The sequence is:
    1. Start with aperture
    2. Apply rotation
    3. Apply Fresnel propagation (defocus)
    4. Apply magnification (scaling)
    
    Parameters
    ----------
    input_aperture : jnp.ndarray
        Input aperture field
    wavelength : float
        Electron wavelength (metres)
    defocus : float
        Defocus distance (metres)
    magnification : float
        Magnification factor
    rotation : float
        Rotation angle (radians)
    pixel_size : float
        Pixel size (metres)
        
    Returns
    -------
    intensity : jnp.ndarray
        Output intensity
    field : jnp.ndarray
        Output complex field
    """
    # Initialize field
    field = input_aperture.astype(jnp.complex128)
    
    # Step 1: Rotate
    field = rotate_field_fft(field, rotation)
    
    # Step 2: Fresnel propagation with defocus
    field = fresnel_propagate(field, wavelength, defocus, pixel_size)
    
    # Step 3: Apply magnification
    field = scale_field_fourier(field, magnification)
    
    # Compute intensity
    intensity = jnp.abs(field) ** 2
    
    return intensity, field


def generate_single_image(
    grid_size: int,
    wavelength: float,
    defocus: float,
    magnification: float,
    rotation: float,
    pixel_size: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate a single forward image.
    
    Parameters
    ----------
    grid_size : int
        Grid size (pixels)
    wavelength : float
        Wavelength (metres)
    defocus : float
        Defocus (metres)
    magnification : float
        Magnification
    rotation : float
        Rotation (radians)
    pixel_size : float
        Pixel size (metres)
        
    Returns
    -------
    intensity : jnp.ndarray
        Intensity image
    field : jnp.ndarray
        Complex field
    """
    # Create square aperture
    aperture = create_square_aperture(grid_size, aperture_fraction=0.3)
    
    # Generate forward image
    intensity, field = collins_forward_model(
        aperture, wavelength, defocus, magnification, rotation, pixel_size
    )
    
    return intensity, field


def generate_dataset(
    output_dir: str = "forward_model_data",
    n_samples: int = 50,
    grid_size: int = 128
):
    """
    Generate a dataset of images with varying parameters.
    
    Parameters
    ----------
    output_dir : str
        Output directory
    n_samples : int
        Number of samples
    grid_size : int
        Image size (pixels)
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Physical parameters
    voltage = 200e3  # 200 kV
    wavelength = energy2wavelength(voltage)
    detector_size = 1e-3  # 1 mm
    pixel_size = detector_size / grid_size
    
    # Parameter ranges (based on n_lens_inversion)
    magnifications = np.linspace(0.8, 1.5, 7)  # 0.8x to 1.5x magnification
    defocus_values = np.linspace(1e-5, 50e-4, 7)  # 10 microns to 0.5 cm
    rotation_angles = np.linspace(0, np.pi/3, 7)  # 0 to 60 degrees
    
    dataset = {
        'images': [],
        'parameters': [],
        'metadata': {
            'voltage': voltage,
            'wavelength': wavelength,
            'grid_size': grid_size,
            'pixel_size': pixel_size,
            'detector_size': detector_size
        }
    }
    
    print(f"Generating {n_samples} samples...")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Pixel size: {pixel_size*1e6:.3f} microns")
    print(f"Wavelength: {wavelength*1e12:.3f} pm")
    
    for i in range(n_samples):
        # Randomly sample parameters
        magnification = np.random.choice(magnifications)
        defocus = np.random.choice(defocus_values)
        rotation = np.random.choice(rotation_angles)
        
        try:
            intensity, field = generate_single_image(
                grid_size, wavelength, defocus, magnification, rotation, pixel_size
            )
            
            # Store results
            dataset['images'].append(np.array(intensity))
            dataset['parameters'].append({
                'defocus': defocus,
                'magnification': magnification,
                'rotation': rotation
            })
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{n_samples} samples")
                
        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save dataset
    output_file = Path(output_dir) / 'dataset.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\nDataset saved to {output_file}")
    print(f"Total samples: {len(dataset['images'])}")
    
    # Visualize samples
    if len(dataset['images']) > 0:
        visualize_samples(dataset, output_dir, n_samples=min(9, len(dataset['images'])))
    
    return dataset


def visualize_samples(dataset, output_dir, n_samples=9):
    """Visualize sample images."""
    n_rows = int(np.sqrt(n_samples))
    n_cols = int(np.ceil(n_samples / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    if n_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, ax in enumerate(axes[:n_samples]):
        if i < len(dataset['images']):
            img = dataset['images'][i]
            params = dataset['parameters'][i]
            
            ax.imshow(img, cmap='gray')
            ax.set_title(
                f"M={params['magnification']:.2f}\n"
                f"Δf={params['defocus']*1e4:.1f}e-4m\n"
                f"θ={np.rad2deg(params['rotation']):.1f}°",
                fontsize=8
            )
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'sample_images.png', dpi=150)
    print(f"Sample visualization saved to {output_dir}/sample_images.png")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("Fast Forward Model Generation (Collins FFT)")
    print("="*70)
    
    # Generate dataset
    dataset = generate_dataset(
        output_dir="forward_model_data",
        n_samples=50,
        grid_size=128
    )
    
    if len(dataset['images']) > 0:
        print("\nDataset generation complete!")
        print(f"Images shape: {dataset['images'][0].shape}")
        print(f"Number of images: {len(dataset['images'])}")
    else:
        print("\nNo images were generated. Check for errors above.")
