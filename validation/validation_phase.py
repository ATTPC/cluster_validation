from spyral import PhaseLike, PhaseResult, ClusterParameters
from spyral.core.point_cloud import PointCloud
from spyral.core.status_message import StatusMessage
from spyral.core.clusterize import form_clusters, join_clusters, cleanup_clusters
from spyral.core.spy_log import spyral_warn, spyral_info, spyral_error

from multiprocessing import SimpleQueue
from numpy.random import Generator
from dataclasses import dataclass
from pathlib import Path
import h5py as h5
import numpy as np
import polars as pl

from spyral.core.run_stacks import form_run_string


@dataclass
class ValidationParameters:
    accuracy_threshold: float


@dataclass
class ValidationResult:
    event: int
    truth_labels: list[int]
    n_truth: int
    n_predicted: int
    n_validated: int
    matches: list[tuple[int, int]]
    total_acc: list[float]
    inclusive_acc: list[float]
    exclusive_acc: list[float]


def validate_clusters(
    pred_labels, truth_labels, valid_params: ValidationParameters, event: int
) -> ValidationResult:
    upred = np.unique(pred_labels)
    utruth = np.unique(truth_labels)
    used_truth = set()
    matches = []
    total_acc = []
    inc_acc = []
    ex_acc = []
    n_pred = len(upred)
    for pred in upred:
        if pred == -1:
            n_pred -= 1  # Do not include noise as predicted label
            continue
        pmask = pred_labels == pred
        for truth in utruth:
            if truth in used_truth:
                continue

            tmask = truth_labels == truth
            tacc = np.count_nonzero(pmask == tmask) / float(len(tmask))
            if tacc > valid_params.accuracy_threshold:
                matches.append((pred, truth))
                total_acc.append(tacc)
                incacc = np.count_nonzero(pmask[tmask]) / float(np.count_nonzero(tmask))
                exmask = ~tmask
                exacc = 1.0
                nex = np.count_nonzero(exmask)
                if nex > 0:  # Some events only one true cluster
                    exacc = np.count_nonzero(~pmask[exmask]) / float(nex)
                inc_acc.append(incacc)
                ex_acc.append(exacc)
                used_truth.add(truth)
                break
    return ValidationResult(
        event,
        utruth.tolist(),
        len(utruth),
        n_pred,
        len(matches),
        matches,
        total_acc,
        inc_acc,
        ex_acc,
    )


class ClusterValidationPhase(PhaseLike):
    def __init__(
        self, cluster_params: ClusterParameters, valid_params: ValidationParameters
    ):
        super().__init__("ClusterValidation")

        self.cluster_params = cluster_params
        self.valid_params = valid_params

    def create_assets(self, workspace_path: Path) -> bool:
        return True

    def construct_artifact(
        self, payload: PhaseResult, workspace_path: Path
    ) -> PhaseResult:
        return PhaseResult(
            artifacts={
                "cluster": self.get_artifact_path(workspace_path)
                / f"{form_run_string(payload.run_number)}.h5",
                "validation": self.get_artifact_path(workspace_path)
                / f"{form_run_string(payload.run_number)}.parquet",
            },
            successful=True,
            run_number=payload.run_number,
        )

    def run(
        self,
        payload: PhaseResult,
        workspace_path: Path,
        msg_queue: SimpleQueue,
        rng: Generator,
    ) -> PhaseResult:
        # Check that point clouds exist
        point_path = payload.artifacts["pointcloud"]
        if not point_path.exists() or not payload.successful:
            spyral_warn(
                __name__,
                f"Point cloud data does not exist for run {payload.run_number} at phase 2. Skipping.",
            )
            return PhaseResult.invalid_result(payload.run_number)

        result = self.construct_artifact(payload, workspace_path)

        point_file = h5.File(point_path, "r")
        cluster_file = h5.File(result.artifacts["cluster"], "w")

        cloud_group: h5.Group = point_file["cloud"]  # type: ignore
        if not isinstance(cloud_group, h5.Group):
            spyral_error(
                __name__, f"Point cloud group not present in run {payload.run_number}!"
            )
            return PhaseResult.invalid_result(payload.run_number)

        min_event: int = cloud_group.attrs["min_event"]  # type: ignore
        max_event: int = cloud_group.attrs["max_event"]  # type: ignore
        cluster_group: h5.Group = cluster_file.create_group("cluster")
        cluster_group.attrs["min_event"] = min_event
        cluster_group.attrs["max_event"] = max_event

        nevents = max_event - min_event + 1
        total: int
        flush_val: int
        if nevents < 100:
            total = nevents
            flush_val = 0
        else:
            flush_percent = 0.01
            flush_val = int(flush_percent * (max_event - min_event))
            total = 100

        count = 0

        msg = StatusMessage(
            self.name, 1, total, payload.run_number
        )  # we always increment by 1

        validation = []

        # Process the data
        for idx in range(min_event, max_event + 1):
            count += 1
            if count > flush_val:
                count = 0
                msg_queue.put(msg)

            cloud_data: h5.Dataset | None = None
            cloud_name = f"cloud_{idx}"
            if cloud_name not in cloud_group:
                continue
            else:
                cloud_data = cloud_group[cloud_name]  # type: ignore

            if cloud_data is None:
                continue

            cloud = PointCloud(idx, cloud_data[:].copy())
            if np.any(np.diff(cloud.data[:, 2]) < 0.0):
                spyral_warn(
                    __name__,
                    f"Clustering for event {cloud.event_number} failed because point cloud was not sorted in z",
                )
                continue

            label_data: h5.Dataset | None = None
            label_name = f"labels_{idx}"
            if label_name not in cloud_group:
                spyral_error(
                    __name__,
                    "Cannot validate clusters without truth labels! Make sure data is from attpc_engine v0.8.0 or greater",
                )
                result.successful = False
                return result
            else:
                label_data = cloud_group[label_name]  # type: ignore

            if label_data is None:
                continue

            clusters, pred_labels = form_clusters(cloud, self.cluster_params)
            joined, pred_labels = join_clusters(
                clusters, self.cluster_params, pred_labels
            )
            cleaned, pred_labels = cleanup_clusters(
                joined, self.cluster_params, pred_labels
            )
            res = validate_clusters(
                pred_labels, label_data[:].copy(), self.valid_params, idx
            )
            validation.append(vars(res))

            # Each event can contain many clusters
            cluster_event_group = cluster_group.create_group(f"event_{idx}")
            cluster_event_group.attrs["nclusters"] = len(cleaned)
            cluster_event_group.attrs["orig_run"] = cloud_data.attrs["orig_run"]
            cluster_event_group.attrs["orig_event"] = cloud_data.attrs["orig_event"]
            cluster_event_group.attrs["ic_amplitude"] = cloud_data.attrs["ic_amplitude"]
            cluster_event_group.attrs["ic_centroid"] = cloud_data.attrs["ic_centroid"]
            cluster_event_group.attrs["ic_integral"] = cloud_data.attrs["ic_integral"]
            cluster_event_group.attrs["ic_multiplicity"] = cloud_data.attrs[
                "ic_multiplicity"
            ]
            for cidx, cluster in enumerate(cleaned):
                local_group = cluster_event_group.create_group(f"cluster_{cidx}")
                local_group.attrs["label"] = cluster.label
                local_group.create_dataset("cloud", data=cluster.data)
        df = pl.DataFrame(validation)
        df.write_parquet(result.artifacts["validation"])
        spyral_info(
            __name__, f"Phase Validate Clusters complete for run {payload.run_number}"
        )
        return result
