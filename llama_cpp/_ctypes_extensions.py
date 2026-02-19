from __future__ import annotations

import sys
import os
import ctypes
import functools
import pathlib

from typing import (
    Any,
    Callable,
    List,
    Union,
    Optional,
    TYPE_CHECKING,
    TypeVar,
    Generic,
)
from typing_extensions import TypeAlias


# Load the library
def load_shared_library(lib_base_name: str, base_paths: Union[pathlib.Path, list[pathlib.Path]]):
    if isinstance(base_paths, pathlib.Path):
        base_paths = [base_paths]

    lib_names = []

    if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        lib_names = [f"lib{lib_base_name}.so"]
    elif sys.platform == "darwin":
        lib_names = [
            f"lib{lib_base_name}.dylib",
            f"lib{lib_base_name}.so",
        ]
    elif sys.platform == "win32":
        lib_names = [
            f"{lib_base_name}.dll",
            f"lib{lib_base_name}.dll",
        ]
    else:
        raise RuntimeError("Unsupported platform")

    cdll_args = dict()  # type: ignore

    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32":
        for base_path in base_paths:
            os.add_dll_directory(str(base_path))
        os.environ["PATH"] = str(base_path) + os.pathsep + os.environ["PATH"]

    if sys.platform == "win32" and sys.version_info >= (3, 9):
        for base_path in base_paths:
            os.add_dll_directory(str(base_path))
        if "CUDA_PATH" in os.environ:
            cuda_path = os.environ["CUDA_PATH"]
            sub_dirs_to_add = [
                "bin",
                os.path.join("bin", "x64"),  # CUDA 13.0+
                "lib",
                os.path.join("lib", "x64")
            ]
            for sub_dir in sub_dirs_to_add:
                full_path = os.path.join(cuda_path, sub_dir)
                if os.path.exists(full_path):
                    os.add_dll_directory(full_path)

        if "HIP_PATH" in os.environ:
            os.add_dll_directory(os.path.join(os.environ["HIP_PATH"], "bin"))
            os.add_dll_directory(os.path.join(os.environ["HIP_PATH"], "lib"))

        if "VULKAN_SDK" in os.environ:
            os.add_dll_directory(os.path.join(os.environ["VULKAN_SDK"], "Bin"))
            os.add_dll_directory(os.path.join(os.environ["VULKAN_SDK"], "Lib"))

        cdll_args["winmode"] = ctypes.RTLD_GLOBAL

    errors = []

    # Try to load the shared library, handling potential errors
    for base_path in base_paths:
        for lib_name in lib_names:
            lib_path = base_path / lib_name
            if lib_path.exists():
                try:
                    return ctypes.CDLL(str(lib_path), **cdll_args)
                except Exception as e:
                    errors.append(f"{lib_path}: {e}")

    raise RuntimeError(
        f"Failed to load '{lib_base_name}' from {base_paths}\n"
        + "\n".join(errors)
    )


# ctypes sane type hint helpers
#
# - Generic Pointer and Array types
# - PointerOrRef type with a type hinted byref function
#
# NOTE: Only use these for static type checking not for runtime checks
# no good will come of that

if TYPE_CHECKING:
    CtypesCData = TypeVar("CtypesCData", bound=ctypes._CData)  # type: ignore

    CtypesArray: TypeAlias = ctypes.Array[CtypesCData]  # type: ignore

    CtypesPointer: TypeAlias = ctypes._Pointer[CtypesCData]  # type: ignore

    CtypesVoidPointer: TypeAlias = ctypes.c_void_p

    class CtypesRef(Generic[CtypesCData]):
        pass

    CtypesPointerOrRef: TypeAlias = Union[
        CtypesPointer[CtypesCData], CtypesRef[CtypesCData]
    ]

    CtypesFuncPointer: TypeAlias = ctypes._FuncPointer  # type: ignore

F = TypeVar("F", bound=Callable[..., Any])


def ctypes_function_for_shared_library(lib: ctypes.CDLL):
    """Decorator for defining ctypes functions with type hints"""

    def ctypes_function(
        name: str, argtypes: List[Any], restype: Any, enabled: bool = True
    ):
        def decorator(f: F) -> F:
            if enabled:
                func = getattr(lib, name)
                func.argtypes = argtypes
                func.restype = restype
                functools.wraps(f)(func)
                return func
            else:
                return f

        return decorator

    return ctypes_function


def _byref(obj: CtypesCData, offset: Optional[int] = None) -> CtypesRef[CtypesCData]:
    """Type-annotated version of ctypes.byref"""
    ...


byref = _byref if TYPE_CHECKING else ctypes.byref
