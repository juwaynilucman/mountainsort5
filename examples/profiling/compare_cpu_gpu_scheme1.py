import time
from pathlib import Path
import numpy as np
import scipy.io as sio
import spikeinterface.core as si
import spikeinterface.preprocessing as spre
import spikeinterface.comparison as sc
import probeinterface as pi
import mountainsort5 as ms5
from tempfile import TemporaryDirectory

try:
    import matplotlib.pyplot as plt
    CAN_PLOT = True
except ImportError:
    CAN_PLOT = False

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

# --- 1. CONFIGURATION ---
SAMPLING_FREQ = 30000
NUM_CHANNELS = 384
DTYPE = "int16"

BASE_DATA_DIR = Path(r"C:\Users\juway\Documents\Marquees-smith")
NPX_BIN_PATH = BASE_DATA_DIR / "c46" / "subset_data" / "raw_1pct.bin"
CHAN_MAP_PATH = Path(r"D:\chanMap.mat")

# --- Sorting Parameters ---
SCHEME1_PARAMS = ms5.Scheme1SortingParameters(
    detect_channel_radius=150,
    detect_threshold=5.5,
    snippet_mask_radius=150,
    skip_alignment=False
)


def run_sorting(recording_preprocessed, use_gpu: bool):
    """Runs scheme 1 and returns the sorting output."""
    with TemporaryDirectory() as tmpdir:
        from mountainsort5.util import create_cached_recording
        recording_cached = create_cached_recording(recording_preprocessed, folder=tmpdir)

        print(f"\n--- Starting MountainSort5 Scheme 1 Run (GPU={'Yes' if use_gpu else 'No'}) ---")
        start_time = time.time()

        sorting = ms5.sorting_scheme1(
            recording=recording_cached,
            sorting_parameters=SCHEME1_PARAMS,
            use_gpu=use_gpu
        )

        total_duration = time.time() - start_time
        print(f"✅ Total Sorting Complete in {total_duration:.2f} seconds.")
        print(f"Found {len(sorting.get_unit_ids())} units.")

        del recording_cached  # Clean up cached recording to free disk space

        return sorting


def main():
    # Pin the global numpy seed for deterministic runs
    np.random.seed(42)

    # --- Data Loading and Preprocessing ---
    print(f"Loading recording: {NPX_BIN_PATH}")
    recording = si.read_binary(NPX_BIN_PATH, sampling_frequency=SAMPLING_FREQ, dtype=DTYPE, num_channels=NUM_CHANNELS)
    mat_contents = sio.loadmat(CHAN_MAP_PATH)
    x, y = mat_contents['xcoords'].flatten(), mat_contents['ycoords'].flatten()
    positions = np.column_stack((x, y))
    probe = pi.Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': 6})
    probe.set_device_channel_indices(np.arange(NUM_CHANNELS))
    recording = recording.set_probe(probe)

    # --- NEW: Slice the recording for development ---
    # Example 1: Grab the first 32 channels
    chan_range = range(155, 167)  # 155:167
    channels_to_keep = recording.get_channel_ids()[:]   # Adjust this range as needed for testing
    
    # Example 2: Grab specific channel IDs (if you know where a good unit is)
    # channels_to_keep = [10, 11, 12, 13, 14, 15] 
    
    print(f"Slicing recording to {len(channels_to_keep)} channels for rapid testing...")
    recording_sliced = recording.select_channels(channel_ids=channels_to_keep)

    print("Applying preprocessing (Filter + Whiten)...")
    # Make sure to use the sliced recording here!
    rec_filtered = spre.bandpass_filter(recording_sliced, freq_min=300, freq_max=6000, dtype='float32')
    rec_preprocessed = spre.whiten(rec_filtered, seed=42)

    # --- Run CPU Sorting ---
    print("\n" + "="*40)
    print("           RUNNING ON CPU")
    print("="*40)
    sorting_cpu = run_sorting(rec_preprocessed, use_gpu=False)

    # --- Run GPU Sorting ---
    sorting_gpu = None
    if HAS_CUDA:
        print("\n" + "="*40)
        print("           RUNNING ON GPU")
        print("="*40)
        try:
            # Reset seed to ensure any stochastic part (e.g., randomized PCA) is identical
            np.random.seed(42)
            sorting_gpu = run_sorting(rec_preprocessed, use_gpu=True)
        except Exception as e:
            print(f"GPU run failed: {e}")
    else:
        print("\nSkipping GPU run: CUDA not available or torch is not installed.")

    # --- Compare CPU and GPU Results ---
    if sorting_cpu and sorting_gpu:
        print("\n" + "="*40)
        print("        COMPARING CPU vs GPU")
        print("="*40)

        # To check for exact spike time matches, we use a delta_time of 0.
        # This means spike times must be identical to be considered a match.
        comparison = sc.compare_two_sorters(
            sorting1=sorting_cpu,
            sorting2=sorting_gpu,
            sorting1_name='CPU',
            sorting2_name='GPU',
            delta_time=0
        )

        print("Full Agreement Scores Matrix (Jaccard Index with delta_time=0):")
        print(comparison.agreement_scores)

        if CAN_PLOT:
            print("\nGenerating agreement score matrix plot ('cpu_vs_gpu_agreement.png')...")
            fig, ax = plt.subplots(figsize=(8, 8))
            
            cax = ax.imshow(comparison.agreement_scores.values, cmap='viridis', vmin=0, vmax=1)
            fig.colorbar(cax, label='Agreement Score')
            
            ax.set_xticks(np.arange(len(comparison.agreement_scores.columns)))
            ax.set_yticks(np.arange(len(comparison.agreement_scores.index)))
            ax.set_xticklabels(comparison.agreement_scores.columns)
            ax.set_yticklabels(comparison.agreement_scores.index)
            
            ax.set_xlabel('GPU Units')
            ax.set_ylabel('CPU Units')
            ax.set_title('CPU vs GPU Spike Time Agreement Scores')
            
            plt.tight_layout()
            plt.savefig('cpu_vs_gpu_agreement.png', dpi=300)
            plt.close()
        else:
            print("\nMatplotlib is not installed. Skipping agreement matrix plot.")

        # --- Detailed Spike-Level Agreement ---
        print("\n--- Spike Time Agreement Details ---")
        
        match_cpu_to_gpu, _ = comparison.get_matching()
        gpu_unit_ids = set(sorting_gpu.get_unit_ids())
        
        # Filter out unmatched units (SpikeInterface uses -1 for unmatched)
        valid_matches = {c: g for c, g in match_cpu_to_gpu.items() if g in gpu_unit_ids}
        
        num_cpu_units = len(sorting_cpu.get_unit_ids())
        num_gpu_units = len(sorting_gpu.get_unit_ids())
        
        unmatched_cpu_units = [u for u in sorting_cpu.get_unit_ids() if u not in valid_matches]
        unmatched_gpu_units = [u for u in sorting_gpu.get_unit_ids() if u not in valid_matches.values()]

        total_spikes_cpu = len(sorting_cpu.to_spike_vector())
        total_spikes_gpu = len(sorting_gpu.to_spike_vector())
        total_matching_spikes = 0
        
        is_identical = (num_cpu_units == num_gpu_units) and not unmatched_cpu_units and not unmatched_gpu_units

        if len(valid_matches) > 0:
            print("\nComparing matched units:")
            for cpu_unit_id, gpu_unit_id in valid_matches.items():
                agreement_score = comparison.agreement_scores.loc[cpu_unit_id, gpu_unit_id]
                num_spikes_cpu = len(sorting_cpu.get_unit_spike_train(cpu_unit_id))
                num_spikes_gpu = len(sorting_gpu.get_unit_spike_train(gpu_unit_id))
                num_matching = comparison.get_matching_event_count(cpu_unit_id, gpu_unit_id)
                total_matching_spikes += num_matching

                if agreement_score == 1.0:
                    print(f"  ✅ Unit CPU:{cpu_unit_id} <=> GPU:{gpu_unit_id} match perfectly ({num_matching} spikes).")
                else:
                    is_identical = False
                    print(f"  ⚠️  Unit CPU:{cpu_unit_id} <=> GPU:{gpu_unit_id} have differences (Agreement: {agreement_score:.4f}):")
                    print(f"     - CPU spikes: {num_spikes_cpu}")
                    print(f"     - GPU spikes: {num_spikes_gpu}")
                    print(f"     - Matching spikes (exact time): {num_matching}")
        
        if unmatched_cpu_units:
            print("\nUnmatched CPU units:")
            for u in unmatched_cpu_units:
                print(f"  - CPU Unit {u} ({len(sorting_cpu.get_unit_spike_train(u))} spikes)")
        
        if unmatched_gpu_units:
            print("\nUnmatched GPU units:")
            for u in unmatched_gpu_units:
                print(f"  - GPU Unit {u} ({len(sorting_gpu.get_unit_spike_train(u))} spikes)")

        print("\n--- Overall Summary ---")
        print(f"Total spikes in CPU sorting: {total_spikes_cpu}")
        print(f"Total spikes in GPU sorting: {total_spikes_gpu}")
        if total_spikes_cpu > 0:
            # Note: total_matching_spikes is only for the units that were matched by the Hungarian algorithm.
            print(f"Total matching spikes in matched units: {total_matching_spikes}")
            match_fraction = total_matching_spikes / total_spikes_cpu
            print(f"Fraction of total CPU spikes found in matched GPU units: {match_fraction:.4f} ({total_matching_spikes}/{total_spikes_cpu})")

        if is_identical:
            print("\n✅ SUCCESS: CPU and GPU outputs are 100% identical at the spike time level.")
        else:
            print("\n⚠️ WARNING: Differences found between CPU and GPU outputs.")


    # Extract structured arrays of all spikes across all units
    vector_cpu = sorting_cpu.to_spike_vector()
    vector_gpu = sorting_gpu.to_spike_vector()

    # Extract and sort just the timestamps (sample indices)
    times_cpu = np.sort(vector_cpu['sample_index'])
    times_gpu = np.sort(vector_gpu['sample_index'])

    print(f"\nTotal clustered spikes (CPU): {len(times_cpu)}")
    print(f"Total clustered spikes (GPU): {len(times_gpu)}")
    print(f"CPU duplicates: {len(times_cpu) - len(np.unique(times_cpu))}")
    print(f"GPU duplicates: {len(times_gpu) - len(np.unique(times_gpu))}")

    if len(times_cpu) == len(times_gpu):
        is_global_match = np.array_equal(times_cpu, times_gpu)
        if is_global_match:
            print("✅ The global set of clustered spikes is 100% identical (ignores unit assignments).")
        else:
            print("⚠️ The global sets of clustered spikes have different timestamps.")
            unique_times_cpu = np.setdiff1d(times_cpu, times_gpu)
            unique_times_gpu = np.setdiff1d(times_gpu, times_cpu)
            print(f"  - Unique to CPU: {len(unique_times_cpu)} spikes")
            print(f"  - Unique to GPU: {len(unique_times_gpu)} spikes") 
            print(f'  - Example unique CPU spike times: {unique_times_cpu[:10]}')
            print(f'  - Example unique GPU spike times: {unique_times_gpu[:10]}')
            intersection = np.intersect1d(times_cpu, times_gpu)
            intersection_fraction = len(intersection) / len(times_cpu) if len(times_cpu) > 0 else 0
            print(f"  - Intersection of spike times: {len(intersection)} spikes ({intersection_fraction:.4f} of CPU spikes)")

    else:
        print("❌ The total number of clustered spikes differs.")
        unique_times_cpu = np.setdiff1d(times_cpu, times_gpu)
        unique_times_gpu = np.setdiff1d(times_gpu, times_cpu)
        print(f"  - Unique to CPU: {len(unique_times_cpu)} spikes")
        print(f"  - Unique to GPU: {len(unique_times_gpu)} spikes")

if __name__ == "__main__":
    main()