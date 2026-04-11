from typing import List
from dataclasses import dataclass
from packaging import version
import numpy as np
import numpy.typing as npt
import math
import spikeinterface as si
from .Scheme1SortingParameters import Scheme1SortingParameters
from ..core.detect_spikes import detect_spikes
from ..core.extract_snippets import extract_snippets
from ..core.isosplit6_subdivision_method import isosplit6_subdivision_method
from ..core.compute_templates import compute_templates
from ..core.compute_pca_features import compute_pca_features
from ..core.align_templates import align_templates
from ..core.align_snippets import align_snippets
from ..core.offset_times import offset_times
from ..core.determine_offsets_to_peak import determine_offsets_to_peak
from ..core.remove_duplicate_times import remove_duplicate_times
from ..core.Timer import Timer


@dataclass
class SortingSchemeExtraOutput:
    templates: npt.NDArray[np.float32]  # K x T x M
    peak_channel_indices: List[int]
    times: npt.NDArray
    labels: npt.NDArray

def sorting_scheme1(
    recording: si.BaseRecording, *,
    sorting_parameters: Scheme1SortingParameters,
    return_extra_output: bool = False,
    use_gpu: bool = False
):
    """MountainSort 5 sorting scheme 1

    Args:
        recording (si.BaseRecording): SpikeInterface recording object
        sorting_parameters (Scheme2SortingParameters): Sorting parameters

    Returns:
        si.BaseSorting: SpikeInterface sorting object
    """

    ###################################################################
    # Handle multi-segment recordings
    if recording.get_num_segments() > 1:
        print('Recording has multiple segments. Joining segments for sorting...')
        recording_joined = si.concatenate_recordings(recording_list=[recording])
        sorting_joined = sorting_scheme1(recording_joined, sorting_parameters=sorting_parameters)
        print('Splitting sorting into segments to match original multisegment recording...')
        sorting = si.split_sorting(sorting_joined, recording_joined)
        return sorting
    ###################################################################

    M = recording.get_num_channels()
    N = recording.get_num_frames()
    sampling_frequency = recording.sampling_frequency

    channel_locations = recording.get_channel_locations()

    print(f'Number of channels: {M}')
    print(f'Number of timepoints: {N}')
    print(f'Sampling frequency: {sampling_frequency} Hz')


    sorting_parameters.check_valid(M=M, N=N, sampling_frequency=sampling_frequency, channel_locations=channel_locations)

    print('Loading traces')
    tt = Timer('load_traces')
    traces: np.ndarray = recording.get_traces()
    tt.report()

    if use_gpu:
        print("Moving traces to GPU .. ")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but not available.")
        # traces = torch.tensor(traces, device='cuda')
        traces = torch.as_tensor(traces, dtype=torch.float32, device='cuda')

    print('Detecting spikes')
    tt = Timer('detect_spikes')
    time_radius = int(math.ceil(sorting_parameters.detect_time_radius_msec / 1000 * sampling_frequency))
    times, channel_indices = detect_spikes(
        traces=traces,
        channel_locations=channel_locations,
        time_radius=time_radius,
        channel_radius=sorting_parameters.detect_channel_radius,
        detect_threshold=sorting_parameters.detect_threshold,
        detect_sign=sorting_parameters.detect_sign,
        margin_left=sorting_parameters.snippet_T1,
        margin_right=sorting_parameters.snippet_T2,
        verbose=True
    )
    print(f'Detected {len(times)} spikes')
    tt.report()

    print('Removing duplicate times')
    tt = Timer('remove_duplicate_times')
    # this is important because isosplit does not do well with duplicate points
    times, channel_indices = remove_duplicate_times(times, channel_indices)
    tt.report()

    print(f'Extracting {len(times)} snippets')
    tt = Timer('extract_snippets')
    snippets = extract_snippets( # L x T x M
        traces=traces,
        channel_locations=channel_locations,
        mask_radius=sorting_parameters.snippet_mask_radius,
        times=times,
        channel_indices=channel_indices,
        T1=sorting_parameters.snippet_T1,
        T2=sorting_parameters.snippet_T2
    )
    tt.report()
    L = snippets.shape[0]
    T = snippets.shape[1]
    assert snippets.shape[2] == M

    npca = sorting_parameters.npca_per_channel * M
    print(f'Computing PCA features with npca={npca}')
    tt = Timer('compute_pca_features')
    features = compute_pca_features(snippets.reshape((L, T * M)), npca=npca)
    tt.report()

    is_torch_features = type(features).__module__.startswith('torch')
    if is_torch_features:
        features = features.cpu().numpy()
        times = times.cpu().numpy()

    print(f'Isosplit6 clustering with npca_per_subdivision={sorting_parameters.npca_per_subdivision}')
    tt = Timer('isosplit6_subdivision_method')
    labels = isosplit6_subdivision_method(
        X=features,
        npca_per_subdivision=sorting_parameters.npca_per_subdivision
    )
    if len(labels) > 0:
        K = int(np.max(labels))
    else:
        K = 0
    print(f'Found {K} clusters')
    tt.report()

    print('Computing templates')
    tt = Timer('compute_templates')

    templates = compute_templates(snippets=snippets, labels=labels) # K x T x M
    is_torch_templates = type(templates).__module__.startswith('torch')
    if is_torch_templates:
        import torch
        # Note: torch.min with dim=0 returns a namedtuple (values, indices)
        peak_channel_indices = [int(torch.argmin(torch.min(templates[i], dim=0).values).item()) for i in range(K)]
    else:
        peak_channel_indices = [int(np.argmin(np.min(templates[i], axis=0))) for i in range(K)]
    tt.report()

    if not sorting_parameters.skip_alignment:
        print('Determining optimal alignment of templates')
        tt = Timer('align_templates')
        offsets = align_templates(templates)
        tt.report()

        print('Aligning snippets')
        tt = Timer('align_snippets')
        snippets = align_snippets(snippets, offsets, labels)
        # this is tricky - we need to subtract the offset to correspond to shifting the template
        times = offset_times(times, -offsets, labels)
        tt.report()

        print('Clustering aligned snippets')
        npca = sorting_parameters.npca_per_channel * M

        print(f'Computing PCA features with npca={npca}')
        tt = Timer('compute_pca_features')
        features = compute_pca_features(snippets.reshape((L, T * M)), npca=npca)
        tt.report()
        
        is_torch_features = type(features).__module__.startswith('torch')
        if is_torch_features:
            features = features.cpu().numpy()

        print(f'Isosplit6 clustering with npca_per_subdivision={sorting_parameters.npca_per_subdivision}')
        tt = Timer('isosplit6_subdivision_method')
        labels = isosplit6_subdivision_method(
            X=features,
            npca_per_subdivision=sorting_parameters.npca_per_subdivision
        )
        if len(labels) > 0:
            K = int(np.max(labels))
        else:
            K = 0
        tt.report()
        print(f'Found {K} clusters after alignment')

        print('Computing templates')
        tt = Timer('compute_templates')
        templates = compute_templates(snippets=snippets, labels=labels) # K x T x M
        is_torch_templates = type(templates).__module__.startswith('torch')
        if is_torch_templates:
            import torch
            peak_channel_indices = [int(torch.argmin(torch.min(templates[i], dim=0).values).item()) for i in range(K)]
        else:
            peak_channel_indices = [int(np.argmin(np.min(templates[i], axis=0))) for i in range(K)]
        tt.report()

        print('Offsetting times to peak')
        tt = Timer('determine_offsets_to_peak')
        # Now we need to offset the times again so that the spike times correspond to actual peaks
        offsets_to_peak = determine_offsets_to_peak(templates, detect_sign=sorting_parameters.detect_sign, T1=sorting_parameters.snippet_T1)
        print('Offsets to peak:', offsets_to_peak)
        # This time we need to add the offset
        times = offset_times(times, offsets_to_peak, labels)
        tt.report()

    # Now we need to make sure the times are in order, because we have offset them
    print('Sorting times')
    tt = Timer('sorting times')
    sort_inds = np.argsort(times, kind='stable') # FIX: Unstable sort caused jitter
    times = times[sort_inds]
    labels = labels[sort_inds]
    tt.report()

    # also make sure none of the times are out of bounds now that we have offset them a couple times
    print('Removing out of bounds times')
    tt = Timer('removing out of bounds times')
    inds_okay = np.where((times >= sorting_parameters.snippet_T1) & (times < N - sorting_parameters.snippet_T2))[0]
    times = times[inds_okay]
    labels = labels[inds_okay]
    tt.report()

    print('Reordering units')
    # relabel so that units are ordered by channel
    # and we also put any labels that are not used at the end
    tt = Timer('reordering units')
    aa = np.array([float(x) for x in peak_channel_indices])
    for k in range(1, K + 1):
        inds = np.where(labels == k)[0]
        if len(inds) == 0:
            aa[k - 1] = np.inf
    new_labels_mapping = np.argsort(np.argsort(aa, kind='stable'), kind='stable') + 1 # FIX: applied stable sort here too
    labels = new_labels_mapping[labels - 1]
    tt.report()

    print('Creating sorting object')
    tt = Timer('creating sorting object')
       # spikeinterface changed function name in version 0.102.2. They also stopped using the dev tag so parsing with packaging is safer
    if version.parse(si.__version__) < version.parse("0.102.2"):
        sorting = si.NumpySorting.from_times_labels([times], [labels], sampling_frequency=sampling_frequency)
    else:
        sorting = si.NumpySorting.from_samples_and_labels([times], [labels], sampling_frequency=sampling_frequency)
    tt.report()

    if return_extra_output:
        extra_output = SortingSchemeExtraOutput(
            templates=templates,
            peak_channel_indices=peak_channel_indices,
            times=times,
            labels=labels
        )
        return sorting, extra_output
    else:
        return sorting


