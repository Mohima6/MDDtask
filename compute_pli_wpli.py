import os
import numpy as np
from scipy.signal import hilbert
from tqdm import tqdm
from joblib import Parallel, delayed
DATASET_PATH = r"C:\Users\mohimaCHAKRABORTY\Taskupdate"
BAND_DECOMP_DIR = os.path.join(DATASET_PATH, "band_decomposition")
OUTPUT_BASE_PLI = os.path.join(DATASET_PATH, "connectivity", "PLI")
OUTPUT_BASE_WPLI = os.path.join(DATASET_PATH, "connectivity", "wPLI")
FREQUENCY_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

# Create output folders for both metrics
for metric_base in [OUTPUT_BASE_PLI, OUTPUT_BASE_WPLI]:
    for band in FREQUENCY_BANDS:
        os.makedirs(os.path.join(metric_base, band), exist_ok=True)


# ========== HELPER FUNCTIONS ==========
def compute_pli(signal1, signal2):
    """
    Phase Lag Index (PLI) between two signals.
    PLI = |E[sign(Imag(cross-spectrum))]|
    Returns value in [0, 1].
    """
    phase1 = np.angle(hilbert(signal1))
    phase2 = np.angle(hilbert(signal2))
    phase_diff = phase1 - phase2
    # Imaginary part of cross-spectrum = sin(phase_diff)
    imag_cross = np.sin(phase_diff)
    pli = np.abs(np.mean(np.sign(imag_cross)))
    return pli


def compute_wpli(signal1, signal2):
    """
    Weighted Phase Lag Index (wPLI) – magnitude of imaginary part weighted by its absolute value.
    wPLI = |E[|Imag| * sign(Imag)]| / E[|Imag|]
    Returns value in [0, 1].
    """
    phase1 = np.angle(hilbert(signal1))
    phase2 = np.angle(hilbert(signal2))
    phase_diff = phase1 - phase2
    imag_cross = np.sin(phase_diff)
    numerator = np.abs(np.mean(np.abs(imag_cross) * np.sign(imag_cross)))
    denominator = np.mean(np.abs(imag_cross))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_connectivity_matrix(epoch_data, metric='pli'):
    """
    Compute full connectivity matrix (PLI or wPLI) for one epoch.
    epoch_data : (n_channels, n_times)
    metric : 'pli' or 'wpli'
    Returns square matrix (n_channels, n_channels) with diagonal = 1.
    """
    n_channels, _ = epoch_data.shape
    mat = np.eye(n_channels, dtype=np.float32)
    func = compute_pli if metric == 'pli' else compute_wpli
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            val = func(epoch_data[i], epoch_data[j])
            mat[i, j] = val
            mat[j, i] = val
    return mat


def process_metric(metric_name, output_base):
    """
    Process all bands for a single connectivity metric (PLI or wPLI).
    """
    print(f"\n Computing {metric_name.upper()}...")
    for band in FREQUENCY_BANDS:
        band_input_dir = os.path.join(BAND_DECOMP_DIR, band)
        band_output_dir = os.path.join(output_base, band)

        epoch_files = [f for f in os.listdir(band_input_dir) if f.endswith('.npy')]
        if not epoch_files:
            print(f"   No band files for {band}, skipping")
            continue

        # Label folder (original clean epochs)
        label_dir = os.path.join(DATASET_PATH, "processed_data_epochs")

        for epoch_file in tqdm(epoch_files, desc=f"{metric_name.upper()} {band}"):
            band_data = np.load(os.path.join(band_input_dir, epoch_file))
            n_epochs, n_channels, n_times = band_data.shape

            # Get base subject name
            base_name = epoch_file.replace(f"_{band}.npy", "")
            label_file = os.path.join(label_dir, f"{base_name}_labels.npy")
            if os.path.exists(label_file):
                labels = np.load(label_file)
            else:
                print(f"   Warning: no labels for {base_name}, skipping")
                continue

            # Compute connectivity for each epoch
            for ep_idx in range(n_epochs):
                epoch_signal = band_data[ep_idx]  # (n_channels, n_times)
                conn_mat = compute_connectivity_matrix(epoch_signal, metric=metric_name)
                out_name = f"{base_name}_epoch{ep_idx:03d}_{metric_name}.npy"
                out_path = os.path.join(band_output_dir, out_name)
                np.save(out_path, conn_mat)

            # Save labels (once per subject)
            label_out = os.path.join(band_output_dir, f"{base_name}_labels.npy")
            np.save(label_out, labels)

            print(f"   Saved {n_epochs} {metric_name.upper()} matrices for {base_name} ({band})")


# ========== MAIN ==========
if __name__ == "__main__":
    print("Starting PLI and wPLI computation...")

    # Compute PLI
    process_metric('pli', OUTPUT_BASE_PLI)

    # Compute wPLI
    process_metric('wpli', OUTPUT_BASE_WPLI)

    print(f"\n All done. Results saved in:\n {OUTPUT_BASE_PLI}\n {OUTPUT_BASE_WPLI}")