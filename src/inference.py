from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


class SantanderPredictor:
    """Load exported artifacts and run probability/class inference."""

    def __init__(self, artifact_dir: str | Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.artifact_dir = Path(artifact_dir) if artifact_dir else project_root / "artifacts"

        model_path = self.artifact_dir / "model.pkl"
        features_path = self.artifact_dir / "selected_features.json"
        metadata_path = self.artifact_dir / "model_metadata.json"

        missing = [p.name for p in (model_path, features_path, metadata_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing model artifacts: " + ", ".join(missing) +
                ". Run the notebook's artifact-export cell first."
            )

        self.model = joblib.load(model_path)
        self.selected_features = json.loads(features_path.read_text(encoding="utf-8"))
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.threshold = float(self.metadata["decision_threshold_f1"])

    @staticmethod
    def _apply_preprocessing(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        if "var3" in frame.columns:
            frame["var3"] = frame["var3"].replace(-999999, np.nan)
        return frame

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = self._apply_preprocessing(frame)
        # Reindex enforces the exact feature order used at training time.
        return frame.reindex(columns=self.selected_features)

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = self._prepare(frame)
        probability = self.model.predict_proba(X)[:, 1]
        prediction = (probability >= self.threshold).astype(int)
        return pd.DataFrame({
            "probability": probability,
            "prediction": prediction,
        }, index=frame.index)

    def predict_one(self, features: Mapping[str, Any]) -> dict[str, Any]:
        result = self.predict_frame(pd.DataFrame([dict(features)])).iloc[0]
        return {
            "probability": float(result["probability"]),
            "prediction": int(result["prediction"]),
            "threshold": self.threshold,
        }
