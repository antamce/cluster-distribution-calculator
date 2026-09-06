from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .models import Calibration


def default_preset_path() -> Path:
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("APPDATA", Path.home() / ".config"))
    return root / "Synpo" / "calibrations.json"


class CalibrationStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_preset_path()

    def load(self) -> dict[str, Calibration]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return {
                name: Calibration.from_dict(value)
                for name, value in raw.get("presets", {}).items()
            }
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read calibration presets: {exc}") from exc

    def save_preset(self, calibration: Calibration) -> None:
        calibration.validate()
        presets = self.load()
        presets[calibration.preset_name] = calibration
        payload = {
            "schema_version": 1,
            "presets": {name: value.to_dict() for name, value in sorted(presets.items())},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)
