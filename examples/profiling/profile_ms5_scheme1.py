import os
import time
import sys
import argparse
import numpy as np
import scipy.io as sio
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory

# SpikeInterface & MountainSort5 Ecosystem
import spikeinterface.core as si
import spikeinterface.preprocessing as spre
import spikeinterface.comparison as sc
import probeinterface as pi
import mountainsort5 as ms5
import mountainsort5.util  # Explicit import to prevent AttributeError
from mountainsort5.core.Timer import Timer as MS5Timer

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    CAN_PLOT = True
except ImportError:
    CAN_PLOT = False

@dataclass
class ProfileConfig:
    """Configuration with your specific local Neuropixel paths as defaults."""
    use_gpu: bool = False
    sampling_freq: int = 30000
    num_channels: int = 384
    dtype: str = "int16"
    
    # --- YOUR SPECIFIC PATHS AS DEFAULTS ---
    npx_bin_path: Path = Path(r"C:\Users\juway\Documents\Marquees-smith\c46\subset_data\raw_1pct.bin")
    chan_map_path: Path = Path(r"D:\chanMap.mat")
    gt_spikes_path: Path = Path(r"C:\Users\juway\Documents\Marquees-smith\c46\subset_st\spikes_1pct.npy")
    output_dir: Path = Path(r"C:\Users\juway\Documents\experiments\ms5\paired\c46_profiled_scheme1_deterministic")

    matching_delta_ms: float = 0.4
    scheme1_params: ms5.Scheme1SortingParameters = field(default_factory=lambda: ms5.Scheme1SortingParameters(
        detect_channel_radius=150,
        detect_threshold=5.5,
        snippet_mask_radius=150,
    ))

    @property
    def sorter_output_dir(self) -> Path:
        return self.output_dir / "sorter_output"

    def validate(self):
        """Standard file existence check across C: and D: drives."""
        required = {
            "Binary Data": self.npx_bin_path,
            "Channel Map": self.chan_map_path,
            "Ground Truth": self.gt_spikes_path
        }
        for label, path in required.items():
            if not path.exists():
                raise FileNotFoundError(f"CRITICAL: {label} not found at {path.absolute()}")

class ProfilingContext:
    """Monkey-patches MS5Timer to intercept and store internal timings."""
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

def setup_recording(config: ProfileConfig) -> si.BaseRecording:
    print(f"Loading binary: {config.npx_bin_path}")
    recording = si.read_binary(
        config.npx_bin_path, 
        sampling_frequency=config.sampling_freq, 
        dtype=config.dtype, 
        num_channels=config.num_channels
    )
    # Load probe geometry from drive
    mat = sio.loadmat(config.chan_map_path)
    positions = np.column_stack((mat['xcoords'].flatten(), mat['ycoords'].flatten()))
    probe = pi.Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': 6})
    probe.set_device_channel_indices(np.arange(config.num_channels))
    return recording.set_probe(probe)

def _create_component_dataframe(timings: list) -> pd.DataFrame:
    """Maps raw labels to plottable components."""
    if not timings: return pd.DataFrame()
    df = pd.DataFrame(timings)
    df['rel_start'] = df['start'] - df['start'].min()
    pass_counts = {'compute_pca_features': 0}
    components = []
    for _, row in df.iterrows():
        lbl = row['label']
        if lbl in ['Recording Setup', 'Preprocessing']: components.append('Component 0: Preprocessing')
        elif lbl == 'Disk Caching': components.append('Component 0.5: Disk Caching')
        elif lbl == 'load_traces': components.append('Component 1: Data Loading')
        elif lbl in ['detect_spikes', 'remove_duplicate_times', 'extract_snippets']: components.append('Component 2: Detection')
        elif lbl == 'compute_pca_features':
            components.append('Component 2: Detection' if pass_counts['compute_pca_features'] == 0 else 'Component 3: Clustering')
            pass_counts['compute_pca_features'] += 1
        elif lbl in ['isosplit6_subdivision_method', 'compute_templates', 'align_templates', 'align_snippets']: components.append('Component 3: Clustering')
        elif lbl in ['determine_offsets_to_peak', 'sorting times', 'reordering units', 'creating sorting object']: components.append('Component 4: Output')
        else: components.append('Other')
    df['Component'] = components
    return df

def plot_sequential_durations(timings: list, output_path: Path):
    if not CAN_PLOT or not timings: return
    df = _create_component_dataframe(timings)
    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig, ax = plt.subplots(figsize=(14, 4))
        comp_types = sorted(df['Component'].unique())
        colors = {c: plt.get_cmap('viridis')(i/len(comp_types)) for i, c in enumerate(comp_types)}
        for _, row in df.iterrows():
            ax.barh(0, row['duration'], left=row['rel_start'], height=0.5, 
                    color=colors[row['Component']], edgecolor='black', linewidth=0.5)
        ax.set_yticks([]); ax.set_xlabel("Time (s)")
        ax.set_title("MountainSort5 Timing Profile", fontweight='bold')
        handles = [mpatches.Patch(color=colors[c], label=c) for c in comp_types]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=3)
        plt.subplots_adjust(bottom=0.35)
        plt.savefig(output_path, dpi=300); plt.close(fig)

def main(config: ProfileConfig):
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    print(f"\n--- Starting MS5 Profiling Run ---\n")

    profiler = ProfilingContext()
    with profiler:
        t_setup = MS5Timer("Recording Setup")
        recording = setup_recording(config)
        t_setup.report()

        t_pre = MS5Timer("Preprocessing")
        rec_f = spre.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype='float32')
        rec_p = spre.whiten(rec_f, seed=42)
        t_pre.report()

        # Robust Temporary Directory Handling for Windows
        tmp_dir_obj = TemporaryDirectory()
        tmp = tmp_dir_obj.name
        try:
            t_cache = MS5Timer("Disk Caching")
            rec_c = ms5.util.create_cached_recording(rec_p, folder=tmp)
            t_cache.report()

            print("Running MountainSort5 Scheme 1...")
            start_time = time.time()
            sorting = ms5.sorting_scheme1(
                recording=rec_c, 
                sorting_parameters=config.scheme1_params, 
                use_gpu=config.use_gpu
            )
            # ASCII replaced emoji to prevent UnicodeEncodeError on Windows
            print(f"DONE: Sorting finished in {time.time() - start_time:.2f}s")

            # CRITICAL: Close file handles before trying to delete the temp folder
            del rec_c
            
        finally:
            # Clean up the temp data manually
            try:
                tmp_dir_obj.cleanup()
            except PermissionError:
                print(f"Warning: Could not auto-delete temp files at {tmp}. You may need to delete them manually later.")

    # Save and Evaluate
    print(f"Saving results to: {config.output_dir}")
    sorting.save(folder=config.sorter_output_dir, overwrite=True)
    
    gt_ts = np.load(config.gt_spikes_path)
    sorting_gt = si.NumpySorting.from_samples_and_labels([gt_ts], [np.zeros(len(gt_ts), dtype='int')], config.sampling_freq)
    comp = sc.compare_sorter_to_ground_truth(sorting_gt, sorting, delta_time=config.matching_delta_ms/1000)
    print("\n--- Performance ---")
    print(comp.get_performance())

    if CAN_PLOT:
        plot_path = config.output_dir / "ms5_timing_profile.png"
        plot_sequential_durations(profiler.timings, plot_path)
        print(f"Plot saved to: {plot_path}")

def parse_args() -> ProfileConfig:
    parser = argparse.ArgumentParser(description="Deterministic MS5 Profiler")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--npx-bin", type=Path)
    parser.add_argument("--chan-map", type=Path)
    parser.add_argument("--gt-spikes", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    config = ProfileConfig(use_gpu=args.use_gpu)
    if args.npx_bin: config.npx_bin_path = args.npx_bin
    if args.chan_map: config.chan_map_path = args.chan_map
    if args.gt_spikes: config.gt_spikes_path = args.gt_spikes
    if args.out_dir: config.output_dir = args.out_dir
    return config

if __name__ == "__main__":
    conf = parse_args()
    try:
        import torch
        if conf.use_gpu and not torch.cuda.is_available():
            conf.use_gpu = False
    except ImportError:
        conf.use_gpu = False
    
    main(conf)