__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Defer heavy mlx_lm top-level imports.
#
# mlx_lm/__init__.py imports generate.py which imports transformers.
# transformers v5 scans ~2000 .py files at import time, taking minutes.
# By pre-seeding sys.modules with a lightweight stub, Python skips
# mlx_lm/__init__.py when we import submodules like mlx_lm.models.cache.
# The real mlx_lm package is loaded on demand via ensure_mlx_lm().
#
# Note: the stub is only installed if mlx_lm hasn't been imported yet.
# If something imports mlx_lm before olmlx, the stub is never installed
# and ensure_mlx_lm() is a no-op.
# ---------------------------------------------------------------------------
import importlib
import importlib.util
import sys
import threading
import types

if "mlx_lm" not in sys.modules:
    _mlx_lm_stub = types.ModuleType("mlx_lm")
    _spec = importlib.util.find_spec("mlx_lm")
    if _spec is not None and _spec.submodule_search_locations:
        _mlx_lm_stub.__path__ = list(_spec.submodule_search_locations)
        _mlx_lm_stub.__package__ = "mlx_lm"
        _mlx_lm_stub.__spec__ = _spec
        _mlx_lm_stub.__mlx_stub__ = True  # marker so we can detect it
        sys.modules["mlx_lm"] = _mlx_lm_stub
    del _spec

_mlx_lm_lock = threading.Lock()


def ensure_mlx_lm():
    """Force-load the real mlx_lm package (triggers transformers import).

    Call this before using mlx_lm.load(), mlx_lm.generate(), etc.
    Safe to call from multiple threads — only the first call does work.
    """
    mod = sys.modules.get("mlx_lm")
    if mod is None or not getattr(mod, "__mlx_stub__", False):
        return  # Already loaded or never stubbed
    with _mlx_lm_lock:
        # Re-check under lock
        mod = sys.modules.get("mlx_lm")
        if mod is None or not getattr(mod, "__mlx_stub__", False):
            return
        del sys.modules["mlx_lm"]
        importlib.import_module("mlx_lm")
