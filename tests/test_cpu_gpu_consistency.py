import pytest
import numpy as np
from pathlib import Path
import scipy.io as sio
import spikeinterface.core as si
import spikeinterface.preprocessing as spre
import probeinterface as pi

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

from mountainsort5.core.detect_spikes import detect_spikes
from mountainsort5.core.extract_snippets import extract_snippets
from mountainsort5.core.compute_pca_features import compute_pca_features
from mountainsort5.schemes.sorting_scheme1 import remove_duplicate_times


# --- CONFIGURATION (Constants only) ---
SAMPLING_FREQ = 30000
NUM_CHANNELS = 384
DTYPE = "int16"


@pytest.fixture(scope="module")
def real_ephys_data(test_config):
    print(f"\nLoading real data from {test_config.npx_bin}...")
    recording = si.read_binary(
        test_config.npx_bin, 
        sampling_frequency=SAMPLING_FREQ, 
        dtype=DTYPE, 
        num_channels=NUM_CHANNELS
    )
    mat_contents = sio.loadmat(test_config.chan_map)
    x, y = mat_contents['xcoords'].flatten(), mat_contents['ycoords'].flatten()
    positions = np.column_stack((x, y))
    probe = pi.Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': 6})
    probe.set_device_channel_indices(np.arange(NUM_CHANNELS))
    recording = recording.set_probe(probe)

    rec_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype='float32')
    rec_preprocessed = spre.whiten(rec_filtered, seed=42)
    
    # Load the traces into memory just like Scheme 1 does
    traces = rec_preprocessed.get_traces()
    channel_locations = rec_preprocessed.get_channel_locations()
    
    return traces.astype(np.float32), channel_locations

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA is required to test GPU consistency")
def test_detect_spikes_consistency(real_ephys_data):
    traces_cpu, channel_locations = real_ephys_data
    
    # 0.5 ms default time radius * 30000 Hz = 15 samples
    time_radius = int(np.ceil(0.5 / 1000 * SAMPLING_FREQ))
    
    params = dict(
        channel_locations=channel_locations,
        time_radius=time_radius,
        channel_radius=150.0,
        detect_threshold=5.5,
        detect_sign=-1,
        margin_left=20, # snippet_T1
        margin_right=20, # snippet_T2
        verbose=False
    )
    
    # Run CPU
    times_cpu, channels_cpu = detect_spikes(traces_cpu, **params)
    
    # Run GPU
    traces_gpu = torch.tensor(traces_cpu, device='cuda')
    times_gpu, channels_gpu = detect_spikes(traces_gpu, **params)
    
    times_gpu_np = times_gpu.cpu().numpy()
    channels_gpu_np = channels_gpu.cpu().numpy()
    
    np.testing.assert_array_equal(times_cpu, times_gpu_np, err_msg="Spike times mismatch between CPU and GPU!")
    np.testing.assert_array_equal(channels_cpu, channels_gpu_np, err_msg="Spike channels mismatch between CPU and GPU!")

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA is required to test GPU consistency")
def test_extract_snippets_consistency(real_ephys_data):
    traces_cpu, channel_locations = real_ephys_data
    
    # First, get realistic spike times using the CPU detect_spikes
    time_radius = int(np.ceil(0.5 / 1000 * SAMPLING_FREQ))
    params_detect = dict(
        channel_locations=channel_locations,
        time_radius=time_radius,
        channel_radius=150.0,
        detect_threshold=5.5,
        detect_sign=-1,
        margin_left=20,
        margin_right=20,
        verbose=False
    )
    times_cpu, channels_cpu = detect_spikes(traces_cpu, **params_detect)
    
    params_extract = dict(
        channel_locations=channel_locations, 
        mask_radius=150.0, 
        T1=20, 
        T2=20
    )
    
    snippets_cpu = extract_snippets(traces_cpu, times=times_cpu, channel_indices=channels_cpu, **params_extract)
    
    traces_gpu = torch.tensor(traces_cpu, device='cuda')
    times_gpu = torch.tensor(times_cpu, device='cuda')
    channels_gpu = torch.tensor(channels_cpu, device='cuda')
    snippets_gpu = extract_snippets(traces_gpu, times=times_gpu, channel_indices=channels_gpu, **params_extract)
    
    # Using allclose for snippets due to negligible floating-point precision differences
    np.testing.assert_allclose(snippets_cpu, snippets_gpu.cpu().numpy(), atol=1e-5, err_msg="Snippets mismatch!")

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA is required to test GPU consistency")
def test_remove_duplicate_times_consistency():
    # Test deterministic array with duplicates
    times_cpu = np.array([10, 20, 20, 30, 40, 40, 40, 50], dtype=np.int64)
    labels_cpu = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    
    times_gpu = torch.tensor(times_cpu, device='cuda')
    labels_gpu = torch.tensor(labels_cpu, device='cuda')
    
    out_times_cpu, out_labels_cpu = remove_duplicate_times(times_cpu, labels_cpu)
    out_times_gpu, out_labels_gpu = remove_duplicate_times(times_gpu, labels_gpu)
    
    np.testing.assert_array_equal(out_times_cpu, out_times_gpu.cpu().numpy(), err_msg="remove_duplicate_times: times mismatch")
    np.testing.assert_array_equal(out_labels_cpu, out_labels_gpu.cpu().numpy(), err_msg="remove_duplicate_times: labels mismatch")

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA is required to test GPU consistency")
def test_compute_pca_features_consistency():
    np.random.seed(42)

    X_cpu = np.random.randn(400, 60).astype(np.float32)
    device = 'cuda' if HAS_CUDA else 'cpu'
    X_gpu = torch.tensor(X_cpu, device=device)

    npca = 9
    tol = 5e-3  # Define once to keep it consistent

    features_cpu = compute_pca_features(X_cpu, npca=npca)
    features_gpu = compute_pca_features(X_gpu, npca=npca).cpu().numpy()

    # 1. Check magnitudes
    np.testing.assert_allclose(
        np.abs(features_cpu),
        np.abs(features_gpu),
        atol=tol,
        err_msg="PCA feature magnitudes mismatch!"
    )

    # 2. Check signs
    diff = np.abs(features_cpu - features_gpu)
    sum_vals = np.abs(features_cpu + features_gpu)
    
    # Elements must be close OR their negations must be close
    is_consistent = np.all((diff < tol) | (sum_vals < tol))
    assert is_consistent, f"PCA features differ by more than just a sign flip at tol {tol}"