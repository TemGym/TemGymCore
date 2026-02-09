"""
Forward model for generating synthetic images with rotation, magnification, and defocus.

This script generates a dataset of images using a Gaussian beam model with:
- Rotation: Applied via coordinate transformation
- Magnification: Controlled via lens focal length
- Defocus: Controlled via propagation distance

The goal is to generate test images for fitting these parameters.
"""

import sys
sys.path.insert(0, '../../src')

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import pickle
from pathlib import Path

from temgym_core.gaussian import (
    make_gaussian, 
    square_input_wave,
    run_to_end,
    Lens,
    FreeSpacePropagator,
    Detector
)
from temgym_core.evaluate import evaluate_gaussians_for
from temgym_core.constants import energy2wavelength
from temgym_core.coordinate_transforms import apply_transformation, _rotate

jax.config.update("jax_enable_x64", True)

# Physical constants
VOLTAGE = 200e3  # 200 kV


def create_square_aperture_beams(
    aperture_size: float,
    n_beams_side: int,
    voltage: float,
    waist: float = 1e-9
) -> List:
    """
    Create a square grid of Gaussian beams to represent a square aperture.
    
    Parameters
    ----------
    aperture_size : float
        Side length of square aperture (metres)
    n_beams_side : int
        Number of beams along each side
    voltage : float
        Accelerating voltage (V)
    waist : float
        Gaussian beam waist (metres)
        
    Returns
    -------
    beams : list
        List of GaussianBeam objects
    """
    # Create individual Gaussian beams in a grid
    beams = []
    x_coords = jnp.linspace(-aperture_size/2, aperture_size/2, n_beams_side)
    y_coords = jnp.linspace(-aperture_size/2, aperture_size/2, n_beams_side)
    
    for x in x_coords:
        for y in y_coords:
            beam = make_gaussian(
                waist_x=waist,
                waist_y=waist,
                voltage=voltage,
                x=float(x),
                y=float(y)
            )
            beams.append(beam)
    
    return beams


def propagate_through_system(
    beams: List,
    focal_length: float,
    defocus: float,
    detector_distance: float
) -> List:
    """
    Propagate beams through a simple lens system.
    
    Parameters
    ----------
    beams : list
        Input beams
    focal_length : float
        Lens focal length (metres)
    defocus : float
        Defocus offset added to propagation distance (metres)
    detector_distance : float
        Nominal distance to detector (metres)
        
    Returns
    -------
    output_beams : list
        Beams at detector plane
    """
    # Create lens
    lens = Lens(focal_length=focal_length, z=0.0)
    
    # Create propagator to detector
    propagator = FreeSpacePropagator()
    
    # Propagate each beam
    output_beams = []
    for beam in beams:
        # Pass through lens
        beam_after_lens = lens(beam)
        
        # Propagate to detector with defocus
        total_distance = detector_distance + defocus
        beam_at_detector = propagator(beam_after_lens, total_distance)
        
        output_beams.append(beam_at_detector)
    
    return output_beams


def evaluate_on_rotated_detector(
    beams: List,
    detector: Detector,
    rotation_angle: float
) -> jnp.ndarray:
    """
    Evaluate Gaussian beams on a rotated detector grid.
    
    Parameters
    ----------
    beams : list
        Beams to evaluate
    detector : Detector
        Detector grid
    rotation_angle : float
        Rotation angle (radians)
        
    Returns
    -------
    field : jnp.ndarray
        Complex field on detector
    """
    # Evaluate field on detector
    field = evaluate_gaussians_for(beams, detector)
    
    # If rotation is needed, apply coordinate transformation
    if abs(rotation_angle) > 1e-10:
        # Create coordinate grids
        ny, nx = detector.shape
        y_pix = jnp.arange(ny) - ny // 2
        x_pix = jnp.arange(nx) - nx // 2
        Y_pix, X_pix = jnp.meshgrid(y_pix, x_pix, indexing='ij')
        
        # Apply rotation to coordinates
        rotation_matrix = _rotate(-rotation_angle)  # Negative for inverse rotation
        
        # Transform coordinates
        Y_rot, X_rot = apply_transformation(Y_pix, X_pix, rotation_matrix)
        
        # Interpolate field at rotated coordinates
        # For simplicity, use nearest neighbor (can be improved with scipy)
        Y_rot_int = jnp.round(Y_rot + ny // 2).astype(int)
        X_rot_int = jnp.round(X_rot + nx // 2).astype(int)
        
        # Clip to valid range
        Y_rot_int = jnp.clip(Y_rot_int, 0, ny - 1)
        X_rot_int = jnp.clip(X_rot_int, 0, nx - 1)
        
        # Resample field
        field_rotated = field[Y_rot_int, X_rot_int]
        return field_rotated
    
    return field


def generate_single_image(
    aperture_size: float,
    n_beams_side: int,
    voltage: float,
    focal_length: float,
    defocus: float,
    detector_distance: float,
    rotation: float,
    detector_size: float,
    detector_pixels: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate a single forward image with given parameters.
    
    Parameters
    ----------
    aperture_size : float
        Square aperture size (metres)
    n_beams_side : int
        Number of beams per side
    voltage : float
        Accelerating voltage (V)
    focal_length : float
        Lens focal length (metres)
    defocus : float
        Defocus distance (metres)
    detector_distance : float
        Nominal detector distance (metres)
    rotation : float
        Rotation angle (radians)
    detector_size : float
        Detector physical size (metres)
    detector_pixels : int
        Number of pixels per side
        
    Returns
    -------
    intensity : jnp.ndarray
        Intensity image
    field : jnp.ndarray
        Complex field
    """
    # Create beams
    beams = create_square_aperture_beams(
        aperture_size, n_beams_side, voltage
    )
    
    # Propagate through system
    beams_out = propagate_through_system(
        beams, focal_length, defocus, detector_distance
    )
    
    # Create detector
    pixel_size = detector_size / detector_pixels
    detector = Detector(
        z=detector_distance + defocus,
        pixel_size=(pixel_size, pixel_size),
        shape=(detector_pixels, detector_pixels)
    )
    
    # Evaluate on detector with rotation
    field = evaluate_on_rotated_detector(beams_out, detector, rotation)
    intensity = jnp.abs(field) ** 2
    
    return intensity, field


def generate_dataset(
    output_dir: str = "forward_model_data",
    n_samples: int = 50
):
    """
    Generate a dataset of images with varying rotation, magnification, and defocus.
    
    Parameters
    ----------
    output_dir : str
        Directory to save dataset
    n_samples : int
        Number of samples to generate
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Base parameters (similar to n_lens_inversion)
    voltage = 200e3  # 200 kV
    aperture_size = 1e-6  # 1 micron square
    n_beams_side = 8  # 64 beams total for speed
    detector_pixels = 128  # 128x128 detector
    detector_size = 1e-3  # 1 mm detector
    
    # Parameter ranges
    focal_lengths = np.linspace(1e-3, 5e-3, 5)  # 1-5 mm focal length
    defocus_values = np.linspace(0, 50e-4, 5)  # 0-0.5 cm defocus
    rotation_angles = np.linspace(0, np.pi/4, 5)  # 0-45 degrees
    detector_distance = 5e-3  # 5 mm nominal distance
    
    dataset = {
        'images': [],
        'parameters': [],
        'metadata': {
            'voltage': voltage,
            'aperture_size': aperture_size,
            'n_beams_side': n_beams_side,
            'detector_pixels': detector_pixels,
            'detector_size': detector_size,
            'detector_distance': detector_distance
        }
    }
    
    print(f"Generating {n_samples} samples...")
    
    for i in range(n_samples):
        # Randomly sample parameters
        focal_length = np.random.choice(focal_lengths)
        defocus = np.random.choice(defocus_values)
        rotation = np.random.choice(rotation_angles)
        
        try:
            intensity, field = generate_single_image(
                aperture_size=aperture_size,
                n_beams_side=n_beams_side,
                voltage=voltage,
                focal_length=focal_length,
                defocus=defocus,
                detector_distance=detector_distance,
                rotation=rotation,
                detector_size=detector_size,
                detector_pixels=detector_pixels
            )
            
            # Compute magnification (approximate)
            magnification = detector_distance / focal_length
            
            # Store results
            dataset['images'].append(np.array(intensity))
            dataset['parameters'].append({
                'focal_length': focal_length,
                'defocus': defocus,
                'rotation': rotation,
                'magnification': magnification
            })
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{n_samples} samples")
                
        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            continue
    
    # Save dataset
    output_file = Path(output_dir) / 'dataset.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\nDataset saved to {output_file}")
    print(f"Total samples: {len(dataset['images'])}")
    
    # Visualize a few samples
    visualize_samples(dataset, output_dir, n_samples=9)
    
    return dataset


def visualize_samples(dataset, output_dir, n_samples=9):
    """Visualize sample images from the dataset."""
    n_rows = int(np.sqrt(n_samples))
    n_cols = int(np.ceil(n_samples / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for i, ax in enumerate(axes[:n_samples]):
        if i < len(dataset['images']):
            img = dataset['images'][i]
            params = dataset['parameters'][i]
            
            ax.imshow(img, cmap='gray')
            ax.set_title(
                f"M={params['magnification']:.1f}\n"
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
    print("="*60)
    print("Forward Model Generation for Rotation/Magnification/Defocus")
    print("="*60)
    
    # Generate dataset
    dataset = generate_dataset(
        output_dir="forward_model_data",
        n_samples=50
    )
    
    print("\nDataset generation complete!")
    print(f"Images shape: {dataset['images'][0].shape}")
    print(f"Number of images: {len(dataset['images'])}")
