from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1

_MODEL_ID_FIELDS = (
    "schema_version",
    "target",
    "model_kind",
    "num_layers",
    "num_experts",
    "top_k",
    "buckets",
    "feature_names",
    "feature_mean",
    "feature_scale",
    "coefficients",
    "intercept",
    "minimum_ms",
    "fingerprint",
    "unknown_row_expert_route_priors",
    "proxy_workload_model",
    "protocol_adjustment",
)


def compute_model_id(artifact: Mapping[str, object]) -> str:
    payload = {
        key: artifact[key]
        for key in _MODEL_ID_FIELDS
        if key in artifact
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class VerifyCpuWorkload:
    bucket: int
    logical_tokens: int
    layer_route_counts: tuple[tuple[float, ...], ...]

    @classmethod
    def from_mapping(
        cls,
        *,
        bucket: int,
        logical_tokens: int,
        layer_route_counts: Mapping[str | int, Sequence[float]],
        num_layers: int,
        num_experts: int,
    ) -> "VerifyCpuWorkload":
        rows: list[tuple[float, ...]] = []
        for layer_idx in range(int(num_layers)):
            raw = layer_route_counts.get(str(layer_idx))
            if raw is None:
                raw = layer_route_counts.get(layer_idx, ())
            values = tuple(max(0.0, float(value)) for value in raw)
            if not values:
                values = (0.0,) * int(num_experts)
            if len(values) != int(num_experts):
                raise ValueError(
                    f"layer {layer_idx} has {len(values)} expert counts; "
                    f"expected {num_experts}"
                )
            rows.append(values)
        return cls(
            bucket=int(bucket),
            logical_tokens=int(logical_tokens),
            layer_route_counts=tuple(rows),
        )

    @property
    def padding_tokens(self) -> int:
        return max(0, int(self.bucket) - int(self.logical_tokens))


@dataclass(frozen=True)
class VerifyCostPrediction:
    total_ms: float
    fixed_ms: float
    exposed_cpu_ms: float
    error_p90_ms: float
    estimated_cpu_routes: float
    estimated_cpu_experts: float


@dataclass(frozen=True)
class VerifyWorkloadEstimate:
    workload: VerifyCpuWorkload
    known_rows: int
    unknown_rows: int
    known_cpu_routes: float
    prior_cpu_routes: float


@dataclass(frozen=True)
class VerifyProxySummary:
    """Vectorized proxy state consumed without materializing Python matrices."""

    bucket: int
    logical_tokens: int
    layer_route_counts: np.ndarray
    known_rows: int
    unknown_rows: int
    known_cpu_routes: float
    prior_cpu_routes: float
    proxy_cpu_routes: float
    proxy_cpu_experts: float

    def to_estimate(self) -> VerifyWorkloadEstimate:
        return VerifyWorkloadEstimate(
            workload=VerifyCpuWorkload(
                bucket=int(self.bucket),
                logical_tokens=int(self.logical_tokens),
                layer_route_counts=tuple(
                    tuple(float(value) for value in row)
                    for row in self.layer_route_counts
                ),
            ),
            known_rows=int(self.known_rows),
            unknown_rows=int(self.unknown_rows),
            known_cpu_routes=float(self.known_cpu_routes),
            prior_cpu_routes=float(self.prior_cpu_routes),
        )


def _shape_statistics(workload: VerifyCpuWorkload) -> dict[str, float]:
    layer_routes: list[float] = []
    total_routes = 0.0
    total_experts = 0.0
    route_sq_sum = 0.0
    max_route_per_expert_sum = 0.0
    singleton_experts = 0.0
    multi_route_experts = 0.0
    nonempty_layers = 0.0
    per_layer_stats: dict[str, float] = {}

    for layer_idx, row in enumerate(workload.layer_route_counts):
        routes = float(sum(row))
        active_by_expert = [min(1.0, max(0.0, float(value))) for value in row]
        multi_by_expert = [
            min(1.0, max(0.0, float(value) - 1.0)) for value in row
        ]
        experts = float(sum(active_by_expert))
        stats_for_layer = {
            f"layer_{layer_idx}_cpu_routes": routes,
            f"layer_{layer_idx}_cpu_experts": experts,
            f"layer_{layer_idx}_route_sq_sum": float(
                sum(value * value for value in row)
            ),
        }
        layer_routes.append(routes)
        per_layer_stats.update(stats_for_layer)
        total_routes += routes
        total_experts += experts
        route_sq_sum += float(sum(value * value for value in row))
        max_route_per_expert_sum += float(max(row, default=0.0))
        multi_route_experts += float(sum(multi_by_expert))
        singleton_experts += float(
            sum(active_by_expert) - sum(multi_by_expert)
        )
        nonempty_layers += float(routes > 0.0)

    max_layer_routes = max(layer_routes, default=0.0)
    if layer_routes:
        mean_layer_routes = total_routes / float(len(layer_routes))
        std_layer_routes = math.sqrt(
            sum((value - mean_layer_routes) ** 2 for value in layer_routes)
            / float(len(layer_routes))
        )
    else:
        std_layer_routes = 0.0
    return {
        "logical_tokens": float(workload.logical_tokens),
        "padding_tokens": float(workload.padding_tokens),
        "cpu_routes": total_routes,
        "cpu_experts": total_experts,
        "route_sq_sum": route_sq_sum,
        "max_route_per_expert_sum": max_route_per_expert_sum,
        "singleton_experts": singleton_experts,
        "multi_route_experts": multi_route_experts,
        "nonempty_layers": nonempty_layers,
        "max_layer_routes": max_layer_routes,
        "std_layer_routes": std_layer_routes,
        "cpu_routes_x_bucket": total_routes * float(workload.bucket),
        "cpu_experts_x_bucket": total_experts * float(workload.bucket),
        **(per_layer_stats if workload.layer_route_counts else {}),
    }


def feature_values(
    workload: VerifyCpuWorkload,
    feature_names: Iterable[str],
) -> list[float]:
    stats = _shape_statistics(workload)
    values: list[float] = []
    for name in feature_names:
        if name.startswith("bucket_"):
            bucket = int(name.split("_", 1)[1])
            values.append(float(int(workload.bucket) == bucket))
            continue
        if name not in stats:
            raise KeyError(f"unknown verify cost feature: {name}")
        values.append(float(stats[name]))
    return values


def proxy_feature_values(
    estimate: VerifyWorkloadEstimate,
    feature_names: Iterable[str],
    *,
    cached_expert_count: int,
    ready_direct_active_experts: int,
) -> list[float]:
    workload_stats = _shape_statistics(estimate.workload)
    stats = {
        "logical_tokens": float(estimate.workload.logical_tokens),
        "padding_tokens": float(estimate.workload.padding_tokens),
        "known_rows": float(estimate.known_rows),
        "unknown_rows": float(estimate.unknown_rows),
        "known_row_fraction": float(estimate.known_rows)
        / float(max(1, estimate.workload.bucket)),
        "known_cpu_routes": float(estimate.known_cpu_routes),
        "prior_cpu_routes": float(estimate.prior_cpu_routes),
        "proxy_cpu_routes": float(workload_stats["cpu_routes"]),
        "proxy_cpu_experts": float(workload_stats["cpu_experts"]),
        "proxy_cpu_routes_per_row": float(workload_stats["cpu_routes"])
        / float(max(1, estimate.workload.bucket)),
        "proxy_cpu_experts_per_row": float(workload_stats["cpu_experts"])
        / float(max(1, estimate.workload.bucket)),
        "cached_expert_count": float(cached_expert_count),
        "ready_direct_active_experts": float(ready_direct_active_experts),
    }
    stats["proxy_cpu_routes_sq"] = stats["proxy_cpu_routes"] ** 2
    stats["proxy_cpu_experts_sq"] = stats["proxy_cpu_experts"] ** 2
    stats["proxy_routes_x_experts"] = (
        stats["proxy_cpu_routes"] * stats["proxy_cpu_experts"]
    )
    stats["known_routes_sq"] = stats["known_cpu_routes"] ** 2
    for layer_idx, row in enumerate(estimate.workload.layer_route_counts):
        stats[f"layer_{layer_idx}_proxy_cpu_routes"] = float(sum(row))
        stats[f"layer_{layer_idx}_proxy_cpu_experts"] = float(
            sum(min(1.0, max(0.0, float(value))) for value in row)
        )
    values = []
    for name in feature_names:
        if name.startswith("bucket_") and name.removeprefix("bucket_").isdigit():
            bucket = int(name.split("_", 1)[1])
            values.append(float(estimate.workload.bucket == bucket))
            continue
        if name.startswith("b") and "_x_" in name:
            raw_bucket, base_name = name[1:].split("_x_", 1)
            if raw_bucket.isdigit() and base_name in stats:
                values.append(
                    float(estimate.workload.bucket == int(raw_bucket))
                    * float(stats[base_name])
                )
                continue
        if name not in stats:
            raise KeyError(f"unknown verify proxy feature: {name}")
        values.append(float(stats[name]))
    return values


def proxy_summary_feature_values(
    summary: VerifyProxySummary,
    feature_names: Iterable[str],
    *,
    cached_expert_count: int,
    ready_direct_active_experts: int,
) -> list[float]:
    """Build proxy features from NumPy aggregates without Python row scans."""
    feature_names = tuple(feature_names)
    bucket = int(summary.bucket)
    needs_layer_stats = any(name.startswith("layer_") for name in feature_names)
    layer_routes = layer_experts = None
    if needs_layer_stats:
        layer_routes = np.sum(
            summary.layer_route_counts, axis=1, dtype=np.float64
        )
        layer_experts = np.minimum(
            1.0,
            np.maximum(0.0, summary.layer_route_counts),
        ).sum(axis=1, dtype=np.float64)
    stats = {
        "logical_tokens": float(summary.logical_tokens),
        "padding_tokens": float(max(0, bucket - int(summary.logical_tokens))),
        "known_rows": float(summary.known_rows),
        "unknown_rows": float(summary.unknown_rows),
        "known_row_fraction": float(summary.known_rows) / float(max(1, bucket)),
        "known_cpu_routes": float(summary.known_cpu_routes),
        "prior_cpu_routes": float(summary.prior_cpu_routes),
        "proxy_cpu_routes": float(summary.proxy_cpu_routes),
        "proxy_cpu_experts": float(summary.proxy_cpu_experts),
        "proxy_cpu_routes_per_row": float(summary.proxy_cpu_routes)
        / float(max(1, bucket)),
        "proxy_cpu_experts_per_row": float(summary.proxy_cpu_experts)
        / float(max(1, bucket)),
        "cached_expert_count": float(cached_expert_count),
        "ready_direct_active_experts": float(ready_direct_active_experts),
    }
    stats["proxy_cpu_routes_sq"] = stats["proxy_cpu_routes"] ** 2
    stats["proxy_cpu_experts_sq"] = stats["proxy_cpu_experts"] ** 2
    stats["proxy_routes_x_experts"] = (
        stats["proxy_cpu_routes"] * stats["proxy_cpu_experts"]
    )
    stats["known_routes_sq"] = stats["known_cpu_routes"] ** 2

    values: list[float] = []
    for name in feature_names:
        if name.startswith("bucket_") and name.removeprefix("bucket_").isdigit():
            values.append(float(bucket == int(name.removeprefix("bucket_"))))
            continue
        if name.startswith("layer_"):
            parts = name.split("_")
            if len(parts) == 5 and parts[1].isdigit():
                layer_idx = int(parts[1])
                if layer_routes is None or layer_experts is None:
                    raise KeyError(f"unknown verify proxy feature: {name}")
                if not 0 <= layer_idx < len(layer_routes):
                    raise KeyError(f"unknown verify proxy feature: {name}")
                suffix = "_".join(parts[2:])
                if suffix == "proxy_cpu_routes":
                    values.append(float(layer_routes[layer_idx]))
                    continue
                if suffix == "proxy_cpu_experts":
                    values.append(float(layer_experts[layer_idx]))
                    continue
        if name.startswith("b") and "_x_" in name:
            raw_bucket, base_name = name[1:].split("_x_", 1)
            if raw_bucket.isdigit() and base_name in stats:
                values.append(
                    float(bucket == int(raw_bucket)) * float(stats[base_name])
                )
                continue
        if name not in stats:
            raise KeyError(f"unknown verify proxy feature: {name}")
        values.append(float(stats[name]))
    return values


class VerifyTimeCostModel:
    def __init__(self, artifact: Mapping[str, object]) -> None:
        schema_version = int(artifact.get("schema_version", 0) or 0)
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported verify cost model schema {schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        self.artifact = dict(artifact)
        self.num_layers = int(artifact["num_layers"])
        self.num_experts = int(artifact["num_experts"])
        self.top_k = int(artifact["top_k"])
        self.buckets = tuple(int(value) for value in artifact["buckets"])
        self.feature_names = tuple(str(value) for value in artifact["feature_names"])
        self.feature_mean = tuple(float(value) for value in artifact["feature_mean"])
        self.feature_scale = tuple(float(value) for value in artifact["feature_scale"])
        self.coefficients = tuple(float(value) for value in artifact["coefficients"])
        self.intercept = float(artifact["intercept"])
        self.minimum_ms = float(artifact.get("minimum_ms", 0.0) or 0.0)
        raw_protocol_adjustment = artifact.get("protocol_adjustment")
        self.protocol_adjustment = (
            dict(raw_protocol_adjustment)
            if isinstance(raw_protocol_adjustment, Mapping)
            else None
        )
        self._bucket_adjustment_ms: dict[int, float] = {}
        self._bucket_affine: dict[int, tuple[float, float]] = {}
        if self.protocol_adjustment is not None:
            adjustment_kind = self.protocol_adjustment.get("kind")
            if adjustment_kind not in {"bucket_offset_ms", "bucket_affine"}:
                raise ValueError("unsupported verify cost protocol adjustment")
            if adjustment_kind == "bucket_offset_ms":
                raw_offsets = self.protocol_adjustment.get("bucket_offsets_ms")
                if not isinstance(raw_offsets, Mapping):
                    raise ValueError("protocol adjustment lacks bucket offsets")
                self._bucket_adjustment_ms = {
                    int(bucket): float(offset)
                    for bucket, offset in raw_offsets.items()
                }
                if set(self._bucket_adjustment_ms) != set(self.buckets):
                    raise ValueError(
                        "protocol adjustment buckets do not match model buckets"
                    )
                if not all(
                    math.isfinite(value)
                    for value in self._bucket_adjustment_ms.values()
                ):
                    raise ValueError("protocol adjustment offsets must be finite")
            else:
                raw_affine = self.protocol_adjustment.get("bucket_affine")
                if not isinstance(raw_affine, Mapping):
                    raise ValueError("protocol adjustment lacks bucket affine rows")
                for bucket, parameters in raw_affine.items():
                    if not isinstance(parameters, Mapping):
                        raise ValueError("invalid bucket affine parameters")
                    slope = float(parameters["slope"])
                    intercept = float(parameters["intercept_ms"])
                    if not math.isfinite(slope) or slope <= 0.0:
                        raise ValueError("bucket affine slopes must be positive")
                    if not math.isfinite(intercept):
                        raise ValueError("bucket affine intercepts must be finite")
                    self._bucket_affine[int(bucket)] = (slope, intercept)
                if set(self._bucket_affine) != set(self.buckets):
                    raise ValueError(
                        "protocol adjustment buckets do not match model buckets"
                    )
        proxy_workload_model = artifact.get("proxy_workload_model")
        self.proxy_workload_model = (
            dict(proxy_workload_model)
            if isinstance(proxy_workload_model, Mapping)
            else None
        )
        aggregate_features = {
            "logical_tokens",
            "padding_tokens",
            "cpu_routes",
            "cpu_experts",
            "cpu_routes_x_bucket",
            "cpu_experts_x_bucket",
        }
        self._unsupported_aggregate_features = tuple(
            name
            for name in self.feature_names
            if not name.startswith("bucket_") and name not in aggregate_features
        )
        self._proxy_feature_names: tuple[str, ...] = ()
        self._proxy_feature_mean = np.empty(0, dtype=np.float64)
        self._proxy_feature_scale = np.empty(0, dtype=np.float64)
        self._proxy_route_coefficients = np.empty(0, dtype=np.float64)
        self._proxy_expert_coefficients = np.empty(0, dtype=np.float64)
        if self.proxy_workload_model is not None:
            proxy_model = self.proxy_workload_model
            self._proxy_feature_names = tuple(
                str(value) for value in proxy_model["feature_names"]
            )
            self._proxy_feature_mean = np.asarray(
                proxy_model["feature_mean"], dtype=np.float64
            )
            self._proxy_feature_scale = np.asarray(
                proxy_model["feature_scale"], dtype=np.float64
            )
            self._proxy_route_coefficients = np.asarray(
                proxy_model["route_coefficients"], dtype=np.float64
            )
            self._proxy_expert_coefficients = np.asarray(
                proxy_model["expert_coefficients"], dtype=np.float64
            )
            proxy_widths = {
                len(self._proxy_feature_names),
                self._proxy_feature_mean.size,
                self._proxy_feature_scale.size,
                self._proxy_route_coefficients.size,
                self._proxy_expert_coefficients.size,
            }
            if len(proxy_widths) != 1:
                raise ValueError("verify proxy workload model vector sizes do not match")
            if np.any(self._proxy_feature_scale <= 0.0):
                raise ValueError("verify proxy workload feature scales must be positive")
        self.model_id = compute_model_id(artifact)
        recorded_model_id = str(artifact.get("model_id", "") or "")
        if recorded_model_id and recorded_model_id != self.model_id:
            raise ValueError(
                "verify cost model identity mismatch: "
                f"recorded={recorded_model_id} computed={self.model_id}"
            )
        metrics = artifact.get("validation_metrics", {})
        self.error_p90_ms = float(
            metrics.get("p90_abs_error_ms", 0.0) if isinstance(metrics, Mapping) else 0.0
        )
        width = len(self.feature_names)
        if not (
            len(self.feature_mean)
            == len(self.feature_scale)
            == len(self.coefficients)
            == width
        ):
            raise ValueError("verify cost model feature vector sizes do not match")
        if any(scale <= 0.0 for scale in self.feature_scale):
            raise ValueError("verify cost model feature scales must be positive")

    @classmethod
    def load(cls, path: str | Path) -> "VerifyTimeCostModel":
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(artifact, dict):
            raise ValueError("verify cost model artifact must be a JSON object")
        return cls(artifact)

    def _raw_prediction(self, workload: VerifyCpuWorkload) -> float:
        if len(workload.layer_route_counts) != self.num_layers:
            raise ValueError(
                f"workload has {len(workload.layer_route_counts)} layers; "
                f"expected {self.num_layers}"
            )
        if int(workload.bucket) not in self.buckets:
            raise ValueError(
                f"verify bucket {workload.bucket} is not calibrated; "
                f"known buckets={list(self.buckets)}"
            )
        raw = feature_values(workload, self.feature_names)
        prediction = self.intercept
        for value, mean, scale, coefficient in zip(
            raw,
            self.feature_mean,
            self.feature_scale,
            self.coefficients,
            strict=True,
        ):
            prediction += ((value - mean) / scale) * coefficient
        return self._apply_protocol_adjustment(workload.bucket, prediction)

    def _apply_protocol_adjustment(self, bucket: int, prediction: float) -> float:
        prediction = float(prediction)
        if int(bucket) in self._bucket_affine:
            slope, intercept = self._bucket_affine[int(bucket)]
            prediction = slope * prediction + intercept
        else:
            prediction += self._bucket_adjustment_ms.get(int(bucket), 0.0)
        return max(self.minimum_ms, prediction)

    def validate_protocol(
        self,
        *,
        acceptance_strategy: str,
        temperature: float | None = None,
    ) -> None:
        if self.protocol_adjustment is None:
            return
        normalized = str(acceptance_strategy).strip().lower()
        if normalized in {"sampling", "spec_sampling"}:
            normalized = "standard_sampling"
        expected = str(
            self.protocol_adjustment.get("acceptance_strategy", "")
        ).strip().lower()
        if normalized != expected:
            raise ValueError(
                "verify cost protocol adjustment requires acceptance strategy "
                f"{expected!r}; got {normalized!r}"
            )
        expected_temperature = self.protocol_adjustment.get("temperature")
        if temperature is not None and expected_temperature is not None:
            if abs(float(temperature) - float(expected_temperature)) > 1e-6:
                raise ValueError(
                    "verify cost protocol adjustment requires temperature "
                    f"{float(expected_temperature)}; got {float(temperature)}"
                )

    def _prediction_with_global_cpu_counts(
        self,
        workload: VerifyCpuWorkload,
        *,
        cpu_routes: float,
        cpu_experts: float,
    ) -> float:
        stats = {
            "logical_tokens": float(workload.logical_tokens),
            "padding_tokens": float(workload.padding_tokens),
            "cpu_routes": float(cpu_routes),
            "cpu_experts": float(cpu_experts),
            "cpu_routes_x_bucket": float(cpu_routes) * float(workload.bucket),
            "cpu_experts_x_bucket": float(cpu_experts) * float(workload.bucket),
        }
        prediction = self.intercept
        for name, mean, scale, coefficient in zip(
            self.feature_names,
            self.feature_mean,
            self.feature_scale,
            self.coefficients,
            strict=True,
        ):
            if name.startswith("bucket_"):
                value = float(int(name.split("_", 1)[1]) == int(workload.bucket))
            else:
                try:
                    value = stats[name]
                except KeyError as exc:
                    raise ValueError(
                        "aggregate CPU-count prediction is incompatible with "
                        f"model feature: {name}"
                    ) from exc
            prediction += ((value - mean) / scale) * coefficient
        return self._apply_protocol_adjustment(workload.bucket, prediction)

    def predict(self, workload: VerifyCpuWorkload) -> VerifyCostPrediction:
        total_ms = self._raw_prediction(workload)
        empty = VerifyCpuWorkload(
            bucket=workload.bucket,
            logical_tokens=workload.logical_tokens,
            layer_route_counts=tuple(
                (0.0,) * self.num_experts for _ in range(self.num_layers)
            ),
        )
        fixed_ms = self._raw_prediction(empty)
        stats = _shape_statistics(workload)
        return VerifyCostPrediction(
            total_ms=total_ms,
            fixed_ms=fixed_ms,
            exposed_cpu_ms=max(0.0, total_ms - fixed_ms),
            error_p90_ms=self.error_p90_ms,
            estimated_cpu_routes=float(stats["cpu_routes"]),
            estimated_cpu_experts=float(stats["cpu_experts"]),
        )

    def predict_cpu_counts(
        self,
        *,
        bucket: int,
        logical_tokens: int,
        cpu_routes: float,
        cpu_experts: float,
    ) -> VerifyCostPrediction:
        """Predict latency from aggregate CPU work supplied by a causal model."""
        if self._unsupported_aggregate_features:
            raise ValueError(
                "aggregate CPU-count prediction is incompatible with model "
                f"features: {list(self._unsupported_aggregate_features)}"
            )
        bucket = int(bucket)
        logical_tokens = int(logical_tokens)
        cpu_routes = float(cpu_routes)
        cpu_experts = float(cpu_experts)
        if not 0 < logical_tokens <= bucket:
            raise ValueError(
                f"invalid verify logical/bucket tokens {logical_tokens}/{bucket}"
            )
        maximum_routes = float(bucket * self.top_k * self.num_layers)
        maximum_experts = min(
            float(self.num_layers * self.num_experts),
            maximum_routes,
        )
        if not 0.0 <= cpu_routes <= maximum_routes + 1e-6:
            raise ValueError(
                f"CPU routes {cpu_routes} outside [0, {maximum_routes}]"
            )
        if not 0.0 <= cpu_experts <= min(cpu_routes, maximum_experts) + 1e-6:
            raise ValueError(
                f"CPU experts {cpu_experts} outside physical route bounds"
            )
        empty = VerifyCpuWorkload(
            bucket=bucket,
            logical_tokens=logical_tokens,
            layer_route_counts=(),
        )
        total_ms = self._prediction_with_global_cpu_counts(
            empty,
            cpu_routes=cpu_routes,
            cpu_experts=cpu_experts,
        )
        fixed_ms = self._prediction_with_global_cpu_counts(
            empty,
            cpu_routes=0.0,
            cpu_experts=0.0,
        )
        return VerifyCostPrediction(
            total_ms=total_ms,
            fixed_ms=fixed_ms,
            exposed_cpu_ms=max(0.0, total_ms - fixed_ms),
            error_p90_ms=self.error_p90_ms,
            estimated_cpu_routes=cpu_routes,
            estimated_cpu_experts=cpu_experts,
        )

    def predict_proxy(
        self,
        estimate: VerifyWorkloadEstimate,
        *,
        cached_expert_count: int,
        ready_direct_active_experts: int,
    ) -> VerifyCostPrediction:
        if self.proxy_workload_model is None:
            return self.predict(estimate.workload)
        values = proxy_feature_values(
            estimate,
            self._proxy_feature_names,
            cached_expert_count=int(cached_expert_count),
            ready_direct_active_experts=int(ready_direct_active_experts),
        )
        return self._predict_proxy_values(
            values,
            bucket=estimate.workload.bucket,
            logical_tokens=estimate.workload.logical_tokens,
        )

    def predict_proxy_summary(
        self,
        summary: VerifyProxySummary,
        *,
        cached_expert_count: int,
        ready_direct_active_experts: int,
    ) -> VerifyCostPrediction:
        if self.proxy_workload_model is None:
            return self.predict(summary.to_estimate().workload)
        values = proxy_summary_feature_values(
            summary,
            self._proxy_feature_names,
            cached_expert_count=int(cached_expert_count),
            ready_direct_active_experts=int(ready_direct_active_experts),
        )
        return self._predict_proxy_values(
            values,
            bucket=summary.bucket,
            logical_tokens=summary.logical_tokens,
        )

    def _predict_proxy_values(
        self,
        values: Sequence[float],
        *,
        bucket: int,
        logical_tokens: int,
    ) -> VerifyCostPrediction:
        if self.proxy_workload_model is None:
            raise ValueError("verify proxy workload model is not configured")
        model = self.proxy_workload_model
        if len(values) != len(self._proxy_feature_names):
            raise ValueError("verify proxy workload model vector sizes do not match")
        normalized = (
            np.asarray(values, dtype=np.float64) - self._proxy_feature_mean
        ) / self._proxy_feature_scale
        cpu_routes = float(model["route_intercept"]) + float(
            normalized @ self._proxy_route_coefficients
        )
        cpu_experts = float(model["expert_intercept"]) + float(
            normalized @ self._proxy_expert_coefficients
        )
        max_routes = float(
            int(bucket) * self.top_k * self.num_layers
        )
        cpu_routes = min(max_routes, max(0.0, cpu_routes))
        cpu_experts = min(
            float(self.num_layers * self.num_experts),
            cpu_routes,
            max(0.0, cpu_experts),
        )
        count_prediction = self.predict_cpu_counts(
            bucket=int(bucket),
            logical_tokens=int(logical_tokens),
            cpu_routes=cpu_routes,
            cpu_experts=cpu_experts,
        )
        validation = model.get("validation_metrics", {})
        error_p90_ms = (
            float(validation.get("p90_abs_error_ms", self.error_p90_ms))
            if isinstance(validation, Mapping)
            else self.error_p90_ms
        )
        return VerifyCostPrediction(
            total_ms=count_prediction.total_ms,
            fixed_ms=count_prediction.fixed_ms,
            exposed_cpu_ms=count_prediction.exposed_cpu_ms,
            error_p90_ms=error_p90_ms,
            estimated_cpu_routes=cpu_routes,
            estimated_cpu_experts=cpu_experts,
        )

    def validate_fingerprint(self, runtime: Mapping[str, object]) -> None:
        expected = self.artifact.get("fingerprint", {})
        if not isinstance(expected, Mapping):
            return
        mismatches = []
        for key, expected_value in expected.items():
            if key not in runtime or runtime[key] != expected_value:
                mismatches.append(
                    f"{key}: expected={expected_value!r} actual={runtime.get(key)!r}"
                )
        if mismatches:
            raise ValueError(
                "verify cost model fingerprint mismatch: " + "; ".join(mismatches)
            )


class DraftRouteCostProxy:
    """Estimate verify CPU work from draft routes and the published GPU cache."""

    def __init__(self, model: VerifyTimeCostModel) -> None:
        self.model = model
        raw_priors = model.artifact.get("unknown_row_expert_route_priors", {})
        if not isinstance(raw_priors, Mapping):
            raise ValueError("verify cost artifact lacks unknown-row route priors")
        self._priors: dict[int, np.ndarray] = {}
        for raw_bucket, raw_rows in raw_priors.items():
            bucket = int(raw_bucket)
            if not isinstance(raw_rows, Sequence) or len(raw_rows) != model.num_layers:
                raise ValueError(f"invalid unknown-row priors for bucket {bucket}")
            rows = []
            for layer_idx, raw_row in enumerate(raw_rows):
                if not isinstance(raw_row, Sequence) or len(raw_row) != model.num_experts:
                    raise ValueError(
                        f"invalid unknown-row prior width for bucket {bucket}, "
                        f"layer {layer_idx}"
                    )
                row = [max(0.0, float(value)) for value in raw_row]
                route_sum = float(sum(row))
                if not math.isclose(route_sum, float(model.top_k), abs_tol=0.05):
                    raise ValueError(
                        f"unknown-row prior for bucket {bucket}, layer {layer_idx} "
                        f"sums to {route_sum}, expected top_k={model.top_k}"
                    )
                rows.append(row)
            self._priors[bucket] = np.asarray(rows, dtype=np.float32)
        self._known_hist = np.zeros(
            (model.num_layers, model.num_experts), dtype=np.float32
        )
        self._uncached_mask = np.ones_like(self._known_hist, dtype=bool)
        self._layer_offsets = (
            np.arange(model.num_layers, dtype=np.int64)[:, None, None]
            * int(model.num_experts)
        )
        self.reset()

    def reset(self) -> None:
        self._known_hist.fill(0.0)
        self._known_rows = 0

    @property
    def known_rows(self) -> int:
        return int(self._known_rows)

    def observe(self, original_routes: Sequence[Sequence[Sequence[int]]]) -> None:
        routes = np.asarray(original_routes)
        if routes.ndim != 3 or routes.shape[0] != self.model.num_layers:
            raise ValueError(
                f"draft routes have shape {routes.shape}; expected "
                f"[{self.model.num_layers}, batch, {self.model.top_k}]"
            )
        if routes.shape[2] != self.model.top_k:
            raise ValueError(
                f"draft route width is {routes.shape[2]}; expected {self.model.top_k}"
            )
        batch_size = int(routes.shape[1])
        if not batch_size:
            return
        if not np.all(np.isfinite(routes)) or not np.all(routes == np.floor(routes)):
            raise ValueError("draft route ids must be finite integers")
        route_ids = routes.astype(np.int64, copy=False)
        if np.any(route_ids < 0) or np.any(route_ids >= self.model.num_experts):
            raise ValueError("draft route id outside calibrated expert range")
        flat_indices = (route_ids + self._layer_offsets).reshape(-1)
        histogram = np.bincount(
            flat_indices,
            minlength=self.model.num_layers * self.model.num_experts,
        ).reshape(self.model.num_layers, self.model.num_experts)
        self._known_hist += histogram
        self._known_rows += batch_size

    def build_uncached_mask(
        self,
        cached_experts: Mapping[int, Iterable[int]],
        additional_cached_experts: Iterable[tuple[int, int]] = (),
    ) -> np.ndarray:
        """Return a reusable mask whose true entries execute on the CPU."""
        mask = self._uncached_mask
        mask.fill(True)
        for raw_layer_idx, raw_experts in cached_experts.items():
            layer_idx = int(raw_layer_idx)
            if not 0 <= layer_idx < self.model.num_layers:
                raise ValueError(f"cached layer id outside model range: {layer_idx}")
            if not raw_experts:
                continue
            expert_ids = np.fromiter(
                (int(value) for value in raw_experts),
                dtype=np.int64,
                count=len(raw_experts),
            )
            if np.any(expert_ids < 0) or np.any(expert_ids >= self.model.num_experts):
                raise ValueError("cached expert id outside model range")
            mask[layer_idx, expert_ids] = False
        for raw_layer_idx, raw_expert_idx in additional_cached_experts:
            layer_idx = int(raw_layer_idx)
            expert_idx = int(raw_expert_idx)
            if not 0 <= layer_idx < self.model.num_layers:
                raise ValueError(f"cached layer id outside model range: {layer_idx}")
            if not 0 <= expert_idx < self.model.num_experts:
                raise ValueError("cached expert id outside model range")
            mask[layer_idx, expert_idx] = False
        return mask

    def build_uncached_mask_from_host_masks(
        self,
        cached_masks: Iterable[tuple[int, np.ndarray]],
        additional_cached_experts: Iterable[tuple[int, int]] = (),
    ) -> np.ndarray:
        """Build the CPU-execution mask from cache-owned NumPy mask rows."""
        mask = self._uncached_mask
        mask.fill(True)
        for raw_layer_idx, cached_mask in cached_masks:
            layer_idx = int(raw_layer_idx)
            if not 0 <= layer_idx < self.model.num_layers:
                raise ValueError(f"cached layer id outside model range: {layer_idx}")
            row = np.asarray(cached_mask, dtype=bool)
            if row.shape != (self.model.num_experts,):
                raise ValueError(
                    f"cached expert mask has shape {row.shape}; expected "
                    f"({self.model.num_experts},)"
                )
            np.logical_not(row, out=mask[layer_idx])
        for raw_layer_idx, raw_expert_idx in additional_cached_experts:
            layer_idx = int(raw_layer_idx)
            expert_idx = int(raw_expert_idx)
            if not 0 <= layer_idx < self.model.num_layers:
                raise ValueError(f"cached layer id outside model range: {layer_idx}")
            if not 0 <= expert_idx < self.model.num_experts:
                raise ValueError("cached expert id outside model range")
            mask[layer_idx, expert_idx] = False
        return mask

    def estimate_summary(
        self,
        *,
        bucket: int,
        logical_tokens: int,
        cached_experts: Mapping[int, set[int] | frozenset[int]] | None = None,
        uncached_mask: np.ndarray | None = None,
    ) -> VerifyProxySummary:
        bucket = int(bucket)
        logical_tokens = int(logical_tokens)
        if bucket not in self._priors:
            raise ValueError(f"verify bucket {bucket} has no unknown-row route prior")
        if logical_tokens < self._known_rows:
            raise ValueError(
                f"logical verify rows {logical_tokens} are fewer than observed draft rows "
                f"{self._known_rows}"
            )
        if bucket < logical_tokens:
            raise ValueError(
                f"verify bucket {bucket} is smaller than logical rows {logical_tokens}"
            )
        if uncached_mask is None:
            if cached_experts is None:
                raise ValueError("cached experts or an uncached mask is required")
            uncached_mask = self.build_uncached_mask(cached_experts)
        if uncached_mask.shape != self._known_hist.shape:
            raise ValueError(
                f"uncached mask shape {uncached_mask.shape} does not match "
                f"{self._known_hist.shape}"
            )

        unknown_rows = bucket - self._known_rows
        known_cpu_routes = float(
            np.sum(self._known_hist, where=uncached_mask, dtype=np.float64)
        )
        prior_counts = self._priors[bucket] * float(unknown_rows)
        prior_cpu_routes = float(
            np.sum(prior_counts, where=uncached_mask, dtype=np.float64)
        )
        counts = (self._known_hist + prior_counts) * uncached_mask
        return VerifyProxySummary(
            bucket=bucket,
            logical_tokens=logical_tokens,
            layer_route_counts=counts,
            known_rows=self._known_rows,
            unknown_rows=unknown_rows,
            known_cpu_routes=known_cpu_routes,
            prior_cpu_routes=prior_cpu_routes,
            proxy_cpu_routes=float(np.sum(counts, dtype=np.float64)),
            proxy_cpu_experts=float(
                np.minimum(1.0, np.maximum(0.0, counts)).sum(dtype=np.float64)
            ),
        )

    def estimate(
        self,
        *,
        bucket: int,
        logical_tokens: int,
        cached_experts: Mapping[int, set[int] | frozenset[int]],
    ) -> VerifyWorkloadEstimate:
        return self.estimate_summary(
            bucket=bucket,
            logical_tokens=logical_tokens,
            cached_experts=cached_experts,
        ).to_estimate()
