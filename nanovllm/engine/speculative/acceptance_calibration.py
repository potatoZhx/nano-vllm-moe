from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


SCHEMA_VERSION = 1
ARTIFACT_KIND = "acceptance_alpha_calibration"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def acceptance_predictor_identity(path: str | Path) -> dict[str, object]:
    target = Path(path).expanduser().resolve()
    if target.is_dir():
        config_path = target / "config.json"
        weight_path = target / "best_model.pth"
    else:
        weight_path = target
        config_path = target.parent / "config.json"
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(
            "acceptance predictor identity requires config.json and best_model.pth"
        )
    files = {
        "config.json": _sha256(config_path),
        "best_model.pth": _sha256(weight_path),
    }
    encoded = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return {
        "model_id": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def compute_calibration_id(artifact: Mapping[str, object]) -> str:
    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"calibration_id", "created_utc"}
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


class AcceptanceAlphaCalibration:
    def __init__(self, artifact: Mapping[str, object]) -> None:
        if int(artifact.get("schema_version", 0) or 0) != SCHEMA_VERSION:
            raise ValueError("unsupported acceptance alpha calibration schema")
        if str(artifact.get("artifact_kind", "")) != ARTIFACT_KIND:
            raise ValueError("invalid acceptance alpha calibration artifact kind")
        self.artifact = dict(artifact)
        self.calibration_id = compute_calibration_id(artifact)
        recorded_id = str(artifact.get("calibration_id", "") or "")
        if recorded_id and recorded_id != self.calibration_id:
            raise ValueError(
                "acceptance alpha calibration identity mismatch: "
                f"recorded={recorded_id} computed={self.calibration_id}"
            )
        predictor = artifact.get("acceptance_predictor", {})
        if not isinstance(predictor, Mapping):
            raise ValueError("acceptance alpha calibration lacks predictor identity")
        self.predictor_model_id = str(predictor.get("model_id", "") or "")
        if not self.predictor_model_id:
            raise ValueError("acceptance alpha calibration lacks predictor model id")
        transform = artifact.get("transform", {})
        if not isinstance(transform, Mapping):
            raise ValueError("acceptance alpha calibration lacks transform")
        self.selected = str(transform.get("selected", "identity"))
        if self.selected not in {"identity", "affine"}:
            raise ValueError(f"unknown acceptance alpha transform: {self.selected}")
        self.slope = float(transform.get("slope", 1.0))
        self.intercept = float(transform.get("intercept", 0.0))
        context = artifact.get("calibration_context", {})
        if not isinstance(context, Mapping):
            raise ValueError("invalid acceptance alpha calibration context")
        self.acceptance_strategy = str(
            context.get("acceptance_strategy", "") or ""
        ).strip().lower()

    def validate_acceptance_strategy(self, acceptance_strategy: str) -> None:
        actual = str(acceptance_strategy).strip().lower()
        if actual in {"sampling", "spec_sampling"}:
            actual = "standard_sampling"
        expected = self.acceptance_strategy
        if expected in {"sampling", "spec_sampling"}:
            expected = "standard_sampling"
        if expected and expected != actual:
            raise ValueError(
                "acceptance alpha calibration strategy mismatch: "
                f"expected={expected} actual={actual}"
            )
        if not expected and actual == "standard_sampling":
            raise ValueError(
                "standard_sampling requires a strategy-bound acceptance "
                "alpha calibration"
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        acceptance_predictor_path: str | Path,
    ) -> "AcceptanceAlphaCalibration":
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(artifact, dict):
            raise ValueError("acceptance alpha calibration artifact must be an object")
        calibration = cls(artifact)
        actual = acceptance_predictor_identity(acceptance_predictor_path)
        if str(actual["model_id"]) != calibration.predictor_model_id:
            raise ValueError(
                "acceptance alpha calibration predictor mismatch: "
                f"expected={calibration.predictor_model_id} "
                f"actual={actual['model_id']}"
            )
        return calibration

    def calibrate(self, alpha: float) -> float:
        value = float(alpha)
        if self.selected == "affine":
            value = self.slope * value + self.intercept
        return min(1.0, max(0.0, value))
