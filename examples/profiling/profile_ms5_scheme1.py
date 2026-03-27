import time
from pathlib import Path
import numpy as np
import scipy.io as sio
import spikeinterface.core as si
import spikeinterface.preprocessing as spre
import spikeinterface.comparison as sc
import probeinterface as pi
import mountainsort5 as ms5
from mountainsort5.core.Timer import Timer as MS5Timer
from tempfile import TemporaryDirectory

# It's useful to have plotting libraries for visualization.
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import pandas as pd
    CAN_PLOT = True
except ImportError:
    CAN_PLOT = False

np.random.seed(42)

# --- THE MAGIC FIX FOR DETERMINISM ---
# This ensures SpikeInterface 'whiten' uses fixed chunks, 
# and it globally seeds numpy.random for isosplit6 and PCA!


# --- 1. CONFIGURATION ---
SAMPLING_FREQ = 30000
NUM_CHANNELS = 384
DTYPE = "int16"

BASE_DATA_DIR = Path(r"C:\Users\juway\Documents\Marquees-smith\c46")
NPX_BIN_PATH = BASE_DATA_DIR / "subset_data" / "raw_1pct.bin"
CHAN_MAP_PATH = Path(r"D:\chanMap.mat")
GT_SPIKES_PATH = BASE_DATA_DIR / "subset_st" / "spikes_1pct.npy"

# --- Output (New Clean Directory) ---
EXPERIMENTS_DIR = Path(r"C:\Users\juway\Documents\experiments\ms5\paired")
OUTPUT_DIR = EXPERIMENTS_DIR / "c46_profiled_scheme1_deterministic"
SORTER_OUTPUT_DIR = OUTPUT_DIR / "sorter_output"

# --- Sorting Parameters ---
SCHEME1_PARAMS = ms5.Scheme1SortingParameters(
    detect_channel_radius=150,
    detect_threshold=5.5,
    snippet_mask_radius=150,
)

# --- Evaluation Parameters ---
MATCHING_DELTA_MS = 0.4


class ProfilingContext:
    def __init__(self):
        self.timings = []
        self._original_report = None

    def __enter__(self):
        self._original_report = MS5Timer.report

        def new_report(timer_instance):
            end_time = time.time()
            start_time = getattr(timer_instance, '_start_time', None)
            label = getattr(timer_instance, '_label', 'Unknown')

            if start_time is not None:
                self.timings.append({
                    "label": label,
                    "start": start_time,
                    "duration": end_time - start_time
                })
            self._original_report(timer_instance)

        MS5Timer.report = new_report
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._original_report:
            MS5Timer.report = self._original_report


def run_ms5_sorting(rec_preprocessed):
    with TemporaryDirectory() as tmpdir:
        print(f"Caching preprocessed data to {tmpdir}...")
        from mountainsort5.util import create_cached_recording
        recording_cached = create_cached_recording(rec_preprocessed, folder=tmpdir)

        print("\n--- Starting MountainSort5 Scheme 1 Run ---")
        start_time = time.time()
        sorting = ms5.sorting_scheme1(
            recording=recording_cached,
            sorting_parameters=SCHEME1_PARAMS,
            use_gpu=False
        )
        total_duration = time.time() - start_time
        print(f"\n✅ Total Sorting Complete in {total_duration:.2f} seconds.")

        del recording_cached

    print(f"\nSaving results to: {SORTER_OUTPUT_DIR}")
    SORTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sorting.save(folder=SORTER_OUTPUT_DIR, overwrite=True)
    return sorting


def evaluate_performance(sorting_output):
    print(f"Loading ground truth spikes from: {GT_SPIKES_PATH}")
    gt_timestamps = np.load(GT_SPIKES_PATH)
    gt_labels = np.zeros(len(gt_timestamps), dtype='int')
    sorting_gt = si.NumpySorting.from_samples_and_labels(
        samples_list=[gt_timestamps],
        labels_list=[gt_labels],
        sampling_frequency=SAMPLING_FREQ
    )

    print("Comparing sorter output to ground truth...")
    comparison = sc.compare_sorter_to_ground_truth(
        sorting_gt,
        sorting_output,
        exhaustive_gt=True,
        delta_time=MATCHING_DELTA_MS / 1000 
    )

    agreement_scores = comparison.agreement_scores
    print("\n--- Agreement Scores (All-to-All) ---")
    print(agreement_scores)

    if not agreement_scores.empty:
        best_unit_id = agreement_scores.idxmax(axis=1).iloc[0]
        best_score = agreement_scores.max(axis=1).iloc[0]
        print(f"\nThe best matching sorted unit is '{best_unit_id}' with an agreement score of {best_score:.4f}.")

    perf_df = comparison.get_performance()
    print("\n--- Official Performance Metrics (after matching) ---")
    print(perf_df)
    print(f"\nMountainSort5 found {len(sorting_output.get_unit_ids())} units.")


def plot_true_gantt_chart(timings, output_path):
    if not CAN_PLOT or not timings:
        return

    df = pd.DataFrame(timings)
    min_start = df['start'].min()
    df['rel_start'] = df['start'] - min_start

    pass_counts = {'compute_pca_features': 0}
    components = []
    
    for _, row in df.iterrows():
        lbl = row['label']
        if lbl == 'Preprocessing':
            components.append('Component 0: Preprocessing (SI)')
        elif lbl == 'load_traces':
            components.append('Component 1: Setup & Data Loading')
        elif lbl in ['detect_spikes', 'remove_duplicate_times', 'extract_snippets']:
            components.append('Component 2: Spike Detection & Extraction (Pass 1)')
        elif lbl == 'compute_pca_features':
            if pass_counts['compute_pca_features'] == 0:
                components.append('Component 2: Spike Detection & Extraction (Pass 1)')
            else:
                components.append('Component 3: Clustering & Alignment (Pass 2)')
            pass_counts['compute_pca_features'] += 1
        elif lbl in ['isosplit6_subdivision_method', 'compute_templates', 'align_templates', 'align_snippets']:
            components.append('Component 3: Clustering & Alignment (Pass 2)')
        elif lbl in ['determine_offsets_to_peak', 'sorting times', 'removing out of bounds times', 'reordering units', 'creating sorting object']:
            components.append('Component 4: Finalization & Output')
        else:
            components.append('Other / Unknown')

    df['Component'] = components
    df['y_pos'] = np.arange(len(df))[::-1]

    fig, ax = plt.subplots(figsize=(14, max(6, len(df) * 0.4)))

    comp_types = sorted(df['Component'].unique())
    cmap = plt.get_cmap('Set2')
    colors = {c: cmap(i % 8) for i, c in enumerate(comp_types)}

    for idx, row in df.iterrows():
        ax.barh(row['y_pos'], row['duration'], left=row['rel_start'],
                color=colors[row['Component']], edgecolor='black')
        ax.text(row['rel_start'] + row['duration'] + 0.5, row['y_pos'],
                f"{row['duration']:.2f}s", va='center', fontsize=9)

    ax.set_yticks(df['y_pos'])
    ax.set_yticklabels(df['label'])

    ax.set_xlabel("Execution Time (Seconds)")
    ax.set_title("MountainSort5 Scheme 1 - True Gantt Timeline (Deterministic)")
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    handles = [mpatches.Patch(color=colors[c], label=c) for c in comp_types]
    ax.legend(handles=handles, title="Pipeline Components", loc='lower right')

    fig.tight_layout()
    print(f"\nSaving Gantt chart to: {output_path}")
    plt.savefig(output_path, dpi=300)


if __name__ == "__main__":
    import torch

    # 1. Check if CUDA is available at all
    print(f"Is CUDA available?: {torch.cuda.is_available()}")

    # 2. Get the name of the device
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")



    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_timings = []

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
    preproc_start_time = time.time()
    rec_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype='float32')
    rec_preprocessed = spre.whiten(rec_filtered, seed=42) # deterministic whitening
    all_timings.append({"label": "Preprocessing", "start": preproc_start_time, "duration": time.time() - preproc_start_time})

    profiler = ProfilingContext()
    with profiler:
        sorting_output = run_ms5_sorting(rec_preprocessed)
    all_timings.extend(profiler.timings)

    evaluate_performance(sorting_output)
    plot_true_gantt_chart(all_timings, output_path=OUTPUT_DIR / "mountainsort5_scheme1_deterministic_gantt.png")