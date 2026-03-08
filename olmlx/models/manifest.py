import dataclasses
import json
import hashlib
import types
import typing
from dataclasses import dataclass, asdict
from pathlib import Path


def _is_str_hint(hint) -> bool:
    """Check if a type hint is str or Optional[str] / str | None."""
    if hint is str:
        return True
    # str | None (Python 3.10+ union)
    if isinstance(hint, types.UnionType):
        return str in hint.__args__ and all(
            a in (str, type(None)) for a in hint.__args__
        )
    # Optional[str]
    origin = getattr(hint, "__origin__", None)
    if origin is typing.Union:
        return str in hint.__args__ and all(
            a in (str, type(None)) for a in hint.__args__
        )
    return False


@dataclass
class ModelManifest:
    name: str
    hf_path: str
    size: int = 0
    modified_at: str = ""
    digest: str = ""
    format: str = "mlx"
    family: str = ""
    parameter_size: str = ""
    quantization_level: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ModelManifest":
        with open(path) as f:
            data = json.load(f)
        # Coerce None to field defaults for str fields; raise on null required fields
        hints = typing.get_type_hints(cls)
        for k, field in cls.__dataclass_fields__.items():
            if k in data and data[k] is None and _is_str_hint(hints.get(k)):
                if field.default is dataclasses.MISSING:
                    raise ValueError(f"Required field '{k}' is null in manifest {path}")
                data[k] = field.default
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @staticmethod
    def compute_digest(name: str) -> str:
        return "sha256:" + hashlib.sha256(name.encode()).hexdigest()[:12]
