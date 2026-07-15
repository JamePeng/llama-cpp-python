from __future__ import annotations

import sys
import os
import ctypes
import functools
import pathlib
from ctypes.util import find_library
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Union,
    Optional,
    TYPE_CHECKING,
    TypeVar,
    Generic,
)
from typing_extensions import TypeAlias

def _format_library_dir_contents(base_paths: list[pathlib.Path]) -> str:
    """Format directory contents for diagnostics after library loading fails."""
    sections = []

    for base_path in base_paths:
        p = pathlib.Path(base_path)

        if not p.exists():
            sections.append(f"{p}: <not found>")
            continue

        if not p.is_dir():
            sections.append(f"{p}: <not a directory>")
            continue

        try:
            # Only list files when reporting a final loading failure.
            files = sorted(x.name for x in p.iterdir())
        except Exception as e:
            sections.append(f"{p}: <failed to list: {e}>")
            continue

        if files:
            sections.append(
                f"{p}:\n"
                + "\n".join(f"  - {name}" for name in files)
            )
        else:
            sections.append(f"{p}: <empty>")

    return "\n".join(sections)

# Load the library
def load_shared_library(lib_base_name: str, base_paths: Union[pathlib.Path, list[pathlib.Path]]):
    if isinstance(base_paths, pathlib.Path):
        base_paths = [base_paths]

    lib_names = []

    if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        lib_names = [f"lib{lib_base_name}.so"]

        base_paths.extend([
            "/usr/local/lib",
            "/usr/lib",
            "/usr/lib64",
        ])

    elif sys.platform == "darwin":
        lib_names = [
            f"lib{lib_base_name}.dylib",
            f"lib{lib_base_name}.so",
        ]

        base_paths.extend([
            "/usr/local/lib",
            "/opt/homebrew/lib",
            "/usr/lib",
        ])

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

        # Add CUDA runtime DLL directories if CUDA is available.
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

        # Add HIP runtime DLL directories when HIP backend is available.
        if "HIP_PATH" in os.environ:
            os.add_dll_directory(os.path.join(os.environ["HIP_PATH"], "bin"))
            os.add_dll_directory(os.path.join(os.environ["HIP_PATH"], "lib"))

        # Add Vulkan SDK DLL directories when Vulkan backend is enabled.
        if "VULKAN_SDK" in os.environ:
            os.add_dll_directory(os.path.join(os.environ["VULKAN_SDK"], "Bin"))
            os.add_dll_directory(os.path.join(os.environ["VULKAN_SDK"], "Lib"))

        # Add package-provided library directories.
        #
        # The paths are added in reverse order intentionally.
        # This ensures that the first entry in base_paths gets prepended
        # to PATH last, making it the highest priority search location.
        #
        # Example:
        #   base_paths = [
        #       package/lib,
        #       package/bin,
        #   ]
        #
        # After reversed iteration:
        #   PATH = package/lib;package/bin;...
        for base_path in reversed(base_paths):
            p = pathlib.Path(base_path)
            if p.exists() and p.is_dir():
                os.add_dll_directory(str(p))
                os.environ["PATH"] = str(p) + os.pathsep + os.environ["PATH"]

        cdll_args["winmode"] = ctypes.RTLD_GLOBAL

    errors = []

    # First, try to find an available library through the system
    lib_path = find_library(lib_base_name)
    if lib_path:
        try:
            lib = ctypes.CDLL(lib_path, **cdll_args)
            print(f"[llama-cpp-python].find_library: loaded library from {lib_path}")
            return lib
        except Exception as e:
            errors.append(f"{lib_path}: {e}")

    # Then fallback to manually checking the list of paths.
    for base_path in base_paths:
        for lib_name in lib_names:
            lib_path = pathlib.Path(base_path) / lib_name

            if lib_path.exists():
                try:
                    lib = ctypes.CDLL(str(lib_path), **cdll_args)
                    print(f"[llama-cpp-python].provided_path: loaded library from {lib_path}")
                    return lib
                except Exception as e:
                    errors.append(f"{lib_path}: {e}")

    # Include directory contents only in the failure path to avoid extra work during successful imports.
    raise RuntimeError(
        f"Failed to load '{lib_base_name}' from {base_paths}\n"
        + "\n".join(errors)
        + "\nLibrary search path contents:\n"
        + _format_library_dir_contents(base_paths)
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
    """Create a decorator used to bind typed Python declarations to C symbols.

    The returned decorator accepts either a single exported symbol name or an
    iterable of ABI-compatible aliases. When aliases are provided, they are
    checked in order and the first available symbol is selected.
    """

    def ctypes_function(
        name: Union[str, Iterable[str]],
        argtypes: List[Any],
        restype: Any,
        enabled: bool = True,
    ):
        """Bind a Python declaration to one of the requested C symbols.

        Args:
            name: A symbol name or an ordered iterable of compatible aliases.
            argtypes: The ctypes argument types assigned to the C function.
            restype: The ctypes return type assigned to the C function.
            enabled: Return the original Python declaration when disabled.

        Raises:
            ValueError: If no symbol names are provided.
            AttributeError: If none of the requested symbols exist in the
                shared library.
        """
        symbol_names = (name,) if isinstance(name, str) else tuple(name)

        if not symbol_names:
            raise ValueError("At least one shared library symbol name is required")

        def decorator(f: F) -> F:
            if not enabled:
                return f

            for symbol_name in symbol_names:
                try:
                    func = getattr(lib, symbol_name)
                except AttributeError:
                    continue

                func.argtypes = argtypes
                func.restype = restype
                functools.update_wrapper(func, f)

                # Preserve the actual exported symbol selected at runtime for
                # diagnostics, especially when ABI aliases are being used.
                func.__ctypes_symbol_name__ = symbol_name
                return func

            raise AttributeError(
                "None of the shared library symbols were found: "
                + ", ".join(symbol_names)
            )

        return decorator

    return ctypes_function


def _byref(obj: CtypesCData, offset: Optional[int] = None) -> CtypesRef[CtypesCData]:
    """Type-annotated version of ctypes.byref"""
    ...


byref = _byref if TYPE_CHECKING else ctypes.byref
