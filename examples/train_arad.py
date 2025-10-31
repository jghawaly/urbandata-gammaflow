"""
ARAD Training Script

Train an ARAD detector on background data and visualize reconstructions.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detectors.arad import ARADDetector
from gammaflow.core.listmode import ListMode
from gammaflow.core.time_series import SpectralTimeSeries
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

# Data settings
TOPCODER_DIR = Path(__file__).parent.parent / 'topcoder'
DATASET = 'training'  # 'training' or 'testing'

# Time series parameters
INTEGRATION_TIME = 5.0  # seconds
STRIDE_TIME = 5.0       # seconds

# Model hyperparameters
LATENT_DIM = 8
DROPOUT = 0.1          # Reduced - too much with L1+L2
BATCH_SIZE = 64         # Increased for more stable BatchNorm
LEARNING_RATE = 0.001   # Reduced 10x - more stable convergence
EPOCHS = 50             # Sufficient for convergence
L1_LAMBDA = 0           # Disabled - too much regularization combined
L2_LAMBDA = 1e-4        # Reduced - was too strong
VALIDATION_SPLIT = 0.2

# Output settings
MODEL_DIR = Path(__file__).parent.parent / 'models'
MODEL_NAME = 'arad_background.pt'
N_RECONSTRUCTIONS = 10  # Number of random reconstructions to plot


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_listmode_run(run_id: int) -> SpectralTimeSeries:
    """Load a listmode run from CSV and convert to SpectralTimeSeries."""
    # Load listmode data from CSV
    run_file = TOPCODER_DIR / DATASET / f'{run_id}.csv'
    
    if not run_file.exists():
        raise FileNotFoundError(f"Run file not found: {run_file}")
    
    data = pd.read_csv(run_file, header=None, names=['time_delta_us', 'energy_keV'])
    
    # Convert to seconds
    time_deltas = data['time_delta_us'].values * 1e-6
    energies = data['energy_keV'].values
    
    # Create ListMode object
    listmode = ListMode(time_deltas=time_deltas, energies=energies)
    
    # Convert to SpectralTimeSeries
    time_series = SpectralTimeSeries.from_list_mode(
        listmode,  # ListMode object as first positional argument
        integration_time=INTEGRATION_TIME,
        stride_time=STRIDE_TIME,
        energy_bins=128,
        energy_range=(20, 2900)
    )
    
    return time_series


def get_background_run_ids() -> np.ndarray:
    """Get all background run IDs from the answer key."""
    answer_key_file = TOPCODER_DIR / 'scorer' / f'answerKey_{DATASET}.csv'
    
    if not answer_key_file.exists():
        raise FileNotFoundError(f"Answer key not found: {answer_key_file}")
    
    answer_key = pd.read_csv(answer_key_file)
    
    # Get all background runs (SourceID == 0)
    background_runs = answer_key[answer_key['SourceID'] == 0]
    background_run_ids = background_runs['RunID'].values
    
    return background_run_ids


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    print("=" * 80)
    print("ARAD TRAINING SCRIPT")
    print("=" * 80)
    print()
    
    # Create models directory if it doesn't exist
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Get all background run IDs
    print("Finding background runs...")
    background_run_ids = get_background_run_ids()
    print(f"Found {len(background_run_ids)} background runs in {DATASET} dataset")
    print()
    
    # Display configuration
    print("Configuration:")
    print(f"  Dataset: {DATASET}")
    print(f"  Integration time: {INTEGRATION_TIME} s")
    print(f"  Stride time: {STRIDE_TIME} s")
    print(f"  Background runs: {len(background_run_ids)} runs")
    print(f"  Latent dimension: {LATENT_DIM}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  L1 lambda: {L1_LAMBDA}")
    print(f"  L2 lambda: {L2_LAMBDA}")
    print(f"  Validation split: {VALIDATION_SPLIT}")
    print()
    
    # Load all background runs
    print("Loading background data...")
    all_spectra = []
    
    for run_id in tqdm(background_run_ids, desc="Loading runs"):
        try:
            time_series = load_listmode_run(run_id)
            all_spectra.append(time_series)
        except Exception as e:
            print(f"\nWarning: Failed to load run {run_id}: {e}")
    
    if not all_spectra:
        raise RuntimeError("No background data loaded!")
    
    # Combine all time series
    print(f"Loaded {len(all_spectra)} runs")
    total_spectra = sum(ts.n_spectra for ts in all_spectra)
    print(f"Total spectra: {total_spectra}")
    
    # Concatenate all spectra into one time series
    combined_counts = np.vstack([ts.counts for ts in all_spectra])
    combined_real_times = np.concatenate([ts.real_times for ts in all_spectra])
    combined_live_times = np.concatenate([ts.live_times for ts in all_spectra])
    combined_timestamps = np.concatenate([
        ts.timestamps + i * 1000  # Offset timestamps to avoid overlap
        for i, ts in enumerate(all_spectra)
    ])
    
    background_training = SpectralTimeSeries.from_array(
        counts=combined_counts,
        energy_edges=all_spectra[0].energy_edges,
        timestamps=combined_timestamps,
        real_times=combined_real_times,
        live_times=combined_live_times
    )
    
    print(f"Combined time series: {background_training.n_spectra} spectra, "
          f"{background_training.n_bins} bins")
    print()
    
    # Initialize ARAD detector
    print("Initializing ARAD detector...")
    detector = ARADDetector(
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        l1_lambda=L1_LAMBDA,
        l2_lambda=L2_LAMBDA,
        validation_split=VALIDATION_SPLIT,
        verbose=True
    )
    print()
    
    # Train the detector
    print("Training ARAD detector...")
    print("=" * 80)
    detector.fit(background_training)
    print("=" * 80)
    print()
    
    # Save the model
    model_path = MODEL_DIR / MODEL_NAME
    detector.save(str(model_path))
    print(f"Model saved to: {model_path}")
    print()
    
    # Plot training history
    print("Plotting training history...")
    history = detector.get_training_history()
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (JSD)', fontsize=12)
    plt.title('ARAD Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'arad_training_history.png', dpi=150)
    print(f"Training history saved to: {MODEL_DIR / 'arad_training_history.png'}")
    plt.close()
    print()
    
    # Plot random reconstructions - all overlaid to see diversity
    print(f"Plotting {N_RECONSTRUCTIONS} random background reconstructions...")
    
    # Select random spectra
    n_total = background_training.n_spectra
    random_indices = np.random.choice(n_total, size=N_RECONSTRUCTIONS, replace=False)
    
    # Create 1x2 subplot: originals on left, reconstructions on right
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Use a colormap to distinguish different spectra
    colors = plt.cm.tab10(np.linspace(0, 1, N_RECONSTRUCTIONS))
    
    for i, idx in enumerate(random_indices):
        spectrum = background_training[idx]
        original = spectrum.count_rate
        reconstructed = detector.reconstruct(spectrum)
        energy_centers = spectrum.energy_centers
        
        # Debug: Print value ranges
        if i == 0:
            print(f"  Debug - First spectrum:")
            print(f"    Original range: [{np.min(original):.2e}, {np.max(original):.2e}]")
            print(f"    Reconstructed range: [{np.min(reconstructed):.2e}, {np.max(reconstructed):.2e}]")
        
        # Add small epsilon to avoid log(0) issues
        original_plot = np.maximum(original, 1e-10)
        reconstructed_plot = np.maximum(reconstructed, 1e-10)
        
        # Plot originals on left
        ax1.plot(energy_centers, original_plot, color=colors[i], linewidth=1.5, 
                alpha=0.7, label=f'Spectrum {idx}')
        
        # Plot reconstructions on right
        ax2.plot(energy_centers, reconstructed_plot, color=colors[i], linewidth=1.5,
                alpha=0.7, label=f'Spectrum {idx}')
    
    # Configure left plot (originals)
    ax1.set_yscale('log')
    ax1.set_xlabel('Energy (keV)', fontsize=12)
    ax1.set_ylabel('Count Rate (s$^{-1}$)', fontsize=12)
    ax1.set_title('Original Spectra (Overlaid)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=1e-2)
    ax1.set_xlim(20, 2900)
    
    # Configure right plot (reconstructions)
    ax2.set_yscale('log')
    ax2.set_xlabel('Energy (keV)', fontsize=12)
    ax2.set_ylabel('Count Rate (s$^{-1}$)', fontsize=12)
    ax2.set_title('Reconstructed Spectra (Overlaid)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=1e-2)
    ax2.set_xlim(20, 2900)
    
    # Add annotation about mode collapse
    fig.text(0.5, 0.02, 
             'Note: If all reconstructions overlap perfectly (one line), model has mode collapse!',
             ha='center', fontsize=11, style='italic', color='red')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(MODEL_DIR / 'arad_reconstructions.png', dpi=150)
    print(f"Reconstructions saved to: {MODEL_DIR / 'arad_reconstructions.png'}")
    plt.close()
    print()
    
    # Mode collapse detection
    print("=" * 80)
    print("MODE COLLAPSE CHECK")
    print("=" * 80)
    
    # Get reconstructions for analysis
    reconstructions = []
    originals = []
    for idx in random_indices:
        spectrum = background_training[idx]
        reconstructions.append(detector.reconstruct(spectrum))
        originals.append(spectrum.count_rate)
    
    reconstructions = np.array(reconstructions)
    originals = np.array(originals)
    
    # Check reconstruction similarity
    pairwise_corr = []
    for i in range(len(reconstructions)):
        for j in range(i+1, len(reconstructions)):
            corr = np.corrcoef(reconstructions[i], reconstructions[j])[0,1]
            pairwise_corr.append(corr)
    
    mean_corr = np.mean(pairwise_corr)
    
    print(f"\nReconstruction diversity metrics:")
    print(f"  Mean pairwise correlation: {mean_corr:.4f}")
    print(f"  Std of reconstructions: {reconstructions.std(axis=0).mean():.2e}")
    print(f"  Std of originals: {originals.std(axis=0).mean():.2e}")
    
    # Normalized shape check
    normalized_recons = reconstructions / reconstructions.max(axis=1, keepdims=True)
    unique_shapes = len(np.unique(normalized_recons.round(4), axis=0))
    
    print(f"\nMode collapse indicators:")
    if mean_corr > 0.98:
        print(f"  ❌ SEVERE MODE COLLAPSE: Mean correlation = {mean_corr:.4f} (> 0.98)")
        print(f"     All reconstructions are nearly identical!")
        print(f"     Action: Reduce regularization further or train longer")
    elif mean_corr > 0.90:
        print(f"  ⚠️  PARTIAL MODE COLLAPSE: Mean correlation = {mean_corr:.4f} (> 0.90)")
        print(f"     Reconstructions lack diversity")
        print(f"     Action: Consider reducing dropout or L2 regularization")
    else:
        print(f"  ✓ GOOD DIVERSITY: Mean correlation = {mean_corr:.4f} (< 0.90)")
        print(f"    Model learned meaningful variation")
    
    print(f"\n  Unique normalized shapes: {unique_shapes}/{len(reconstructions)}")
    if unique_shapes == 1:
        print(f"    ❌ All reconstructions have identical shape (after normalization)")
    elif unique_shapes < len(reconstructions) * 0.3:
        print(f"    ⚠️  Limited shape diversity")
    else:
        print(f"    ✓ Good shape diversity")
    
    print()
    
    # Summary
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Total epochs: {len(history['train_loss'])}")
    print()
    print("Next steps:")
    print("  1. Use the trained model for detection on source runs")
    print("  2. Set threshold using set_threshold_by_far() on background data")
    print("  3. Run detect() on test time series")
    print()


if __name__ == '__main__':
    main()

