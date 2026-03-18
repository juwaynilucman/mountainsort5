import time
from pathlib import Path
import numpy as np
import scipy.io as sio
import spikeinterface.core as si
import spikeinterface.preprocessing as spre
import probeinterface as pi
import mountainsort5 as ms5
from tempfile import TemporaryDirectory

# --- CPU DETERMINISM TEST ---
# This script verifies that the CPU-based execution of MountainSort5
# produces strictly deterministic results across multiple runs.

# --- CONFIGURATION ---
NUM_RUNS = 3
SAMPLING_FREQ = 30000
NUM_CHANNELS = 384
DTYPE = "int16"

BASE_DATA_DIR = Path(r"C:\Users\juway\Documents\Marquees-smith\c46")
NPX_BIN_PATH = BASE_DATA_DIR / "subset_data" / "raw_1pct.bin"
CHAN_MAP_PATH = Path(r"D:\chanMap.mat")

SCHEME1_PARAMS = ms5.Scheme1SortingParameters(
    detect_channel_radius=150,
    detect_threshold=5.5,
    snippet_mask_radius=150
)

def main():
    # --- DETERMINISM SEEDS ---
    # Pin the global numpy seed to prevent isosplit6 and scikit-learn's 
    # randomized SVD from shifting between runs.
    np.random.seed(42)

    print(f"Loading recording: {NPX_BIN_PATH}")
    recording = si.read_binary(NPX_BIN_PATH, sampling_frequency=SAMPLING_FREQ, dtype=DTYPE, num_channels=NUM_CHANNELS)
    mat_contents = sio.loadmat(CHAN_MAP_PATH)
    x, y = mat_contents['xcoords'].flatten(), mat_contents['ycoords'].flatten()
    positions = np.column_stack((x, y))
    probe = pi.Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': 6})
    probe.set_device_channel_indices(np.arange(NUM_CHANNELS))
    recording = recording.set_probe(probe)

    print("Applying preprocessing (Filter + Whiten)...")
    rec_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype='float32')
    
    # Important: We must use a fixed seed for whitening so the input traces are identical!
    rec_preprocessed = spre.whiten(rec_filtered, seed=42)

    sortings = []

    from mountainsort5.util import create_cached_recording
    
    with TemporaryDirectory() as tmpdir:
        print(f"Caching preprocessed data to {tmpdir} to ensure identical disk reads...")
        recording_cached = create_cached_recording(rec_preprocessed, folder=tmpdir)

        for i in range(NUM_RUNS):
            print(f"\n--- Running MountainSort5 CPU (Run {i+1}/{NUM_RUNS}) ---")
            
            from mountainsort5.schemes.sorting_scheme1 import sorting_scheme1
            
            start_time = time.time()
            sorting = sorting_scheme1(
                recording=recording_cached,
                sorting_parameters=SCHEME1_PARAMS
            )

            print(f"Run {i+1} completed in {time.time() - start_time:.2f} seconds. Found {len(sorting.get_unit_ids())} units.")
            
            # Extract all unit ids and spike trains to memory immediately
            # because the underlying tmp files might be deleted or altered
            unit_ids = sorting.get_unit_ids()
            spike_trains = {u: sorting.get_unit_spike_train(u) for u in unit_ids}
            sortings.append({"unit_ids": unit_ids, "spike_trains": spike_trains})

        del recording_cached
    
    print("\n--- Comparing Outputs for Determinism ---")
    is_deterministic = True
    
    reference_run = sortings[0]
    ref_unit_ids = reference_run["unit_ids"]
    
    for i in range(1, NUM_RUNS):
        compare_run = sortings[i]
        comp_unit_ids = compare_run["unit_ids"]
        
        # 1. Check if the exact same Unit IDs were generated in the exact same order
        if not np.array_equal(ref_unit_ids, comp_unit_ids):
            print(f"❌ FAIL: Unit IDs in Run {i+1} do not match Run 1.")
            print(f"  Run 1 Units: {ref_unit_ids}")
            print(f"  Run {i+1} Units: {comp_unit_ids}")
            is_deterministic = False
            continue
        
        # 2. Check if every single spike time matches perfectly
        for unit_id in ref_unit_ids:
            ref_train = reference_run["spike_trains"][unit_id]
            comp_train = compare_run["spike_trains"][unit_id]
            
            if not np.array_equal(ref_train, comp_train):
                print(f"❌ FAIL: Spike trains for Unit {unit_id} differ between Run 1 and Run {i+1}.")
                is_deterministic = False
                break
        
    if is_deterministic:
        print("\n✅ SUCCESS: MountainSort5 outputs are 100% identical across all runs. The pipeline is strictly deterministic.")
    else:
        print("\n❌ FAILED: Differences found across runs. The pipeline is non-deterministic.")

if __name__ == "__main__":
    main()