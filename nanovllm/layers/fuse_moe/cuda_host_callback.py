from __future__ import annotations

import ctypes
import ctypes.util
import threading
from collections.abc import Callable


_HostFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_registry: dict[int, Callable[[], None]] = {}
_registry_lock = threading.Lock()
_next_callback_id = 1
_cudart = None


def _load_cudart():
    global _cudart
    if _cudart is not None:
        return _cudart

    libname = ctypes.util.find_library("cudart")
    candidates = [libname, "libcudart.so", "libcudart.so.12", "libcudart.so.11.0"]
    last_error: OSError | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            lib = ctypes.CDLL(candidate)
            lib.cudaLaunchHostFunc.argtypes = [ctypes.c_void_p, _HostFn, ctypes.c_void_p]
            lib.cudaLaunchHostFunc.restype = ctypes.c_int
            _cudart = lib
            return lib
        except OSError as exc:
            last_error = exc

    raise RuntimeError(f"Unable to load CUDA runtime for host callbacks: {last_error}")


@_HostFn
def _trampoline(arg: ctypes.c_void_p) -> None:
    callback_id = int(arg or 0)
    fn = _registry.get(callback_id)
    if fn is None:
        return
    fn()


def register_host_callback(fn: Callable[[], None]) -> int:
    """Register a Python callable for CUDA host callback use.

    Callback ids are intentionally kept alive for the process lifetime because
    CUDA graphs retain the raw user-data pointer and may replay long after the
    Python call that captured the graph has returned.
    """
    global _next_callback_id
    with _registry_lock:
        callback_id = _next_callback_id
        _next_callback_id += 1
        _registry[callback_id] = fn
        return callback_id


def launch_host_callback(cuda_stream: int, callback_id: int) -> None:
    lib = _load_cudart()
    err = int(
        lib.cudaLaunchHostFunc(
            ctypes.c_void_p(int(cuda_stream)),
            _trampoline,
            ctypes.c_void_p(int(callback_id)),
        )
    )
    if err != 0:
        raise RuntimeError(f"cudaLaunchHostFunc failed with CUDA error code {err}")
