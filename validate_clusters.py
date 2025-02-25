import dotenv

dotenv.load_dotenv()

from spyral import (
    Pipeline,
    start_pipeline,
    PointcloudPhase,
)
from spyral import (
    PadParameters,
    GetParameters,
    FribParameters,
    DetectorParameters,
    ClusterParameters,
    OverlapJoinParameters,
    ContinuityJoinParameters,
    DEFAULT_MAP,
)

from validation import ClusterValidationPhase, ValidationParameters

from pathlib import Path
import multiprocessing

# workspace_path = Path("/some/workspace/path/")
# trace_path = Path("/some/trace/path/")
workspace_path = Path("/Volumes/Pattern/simulation/a1975/deuteron/analysis_testing/")
trace_path = Path(
    "/Volumes/Pattern/simulation/a1975/deuteron/analysis_testing/Pointcloud"
)

run_min = 0
run_max = 0
n_processes = 1

pad_params = PadParameters(
    pad_geometry_path=DEFAULT_MAP,
    pad_time_path=DEFAULT_MAP,
    pad_electronics_path=DEFAULT_MAP,
    pad_scale_path=DEFAULT_MAP,
)

get_params = GetParameters(
    baseline_window_scale=20.0,
    peak_separation=50.0,
    peak_prominence=20.0,
    peak_max_width=50.0,
    peak_threshold=40.0,
)

frib_params = FribParameters(
    baseline_window_scale=100.0,
    peak_separation=50.0,
    peak_prominence=20.0,
    peak_max_width=500.0,
    peak_threshold=100.0,
    ic_delay_time_bucket=1100,
    ic_multiplicity=1,
)

det_params = DetectorParameters(
    magnetic_field=2.85,
    electric_field=45000.0,
    detector_length=1000.0,
    beam_region_radius=25.0,
    micromegas_time_bucket=10.0,
    window_time_bucket=560.0,
    get_frequency=6.25,
    garfield_file_path=Path("/path/to/some/garfield.txt"),
    do_garfield_correction=False,
)

cluster_params = ClusterParameters(
    min_cloud_size=50,
    min_points=5,
    min_size_scale_factor=0.0,
    min_size_lower_cutoff=5,
    cluster_selection_epsilon=13.0,
    overlap_join=None,
    continuity_join=ContinuityJoinParameters(
        join_radius_fraction=0.2, join_z_fraction=0.3
    ),
    outlier_scale_factor=0.05,
)

valid_params = ValidationParameters(
    accuracy_threshold=0.90,
)


pipe = Pipeline(
    [
        PointcloudPhase(
            get_params,
            frib_params,
            det_params,
            pad_params,
        ),
        ClusterValidationPhase(cluster_params, valid_params),
    ],
    [False, True],
    workspace_path,
    trace_path,
)


def main():
    start_pipeline(pipe, run_min, run_max, n_processes)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
