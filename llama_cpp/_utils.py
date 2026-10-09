import os
import sys

from typing import Any, Dict


def normalize_embedding(vector, mode=2):
    """Normalize an embedding using llama.cpp modes; return a list.

    vector (sequence[float]): Input embedding; zero vectors stay unchanged.
    mode (bool/int): True selects L2 (2), False selects no normalization (-1).
        <0: None; preserve the original values.
         0: MaxInt16; scale max(abs(x)) to 32760, retaining floating-point values.
         1: L1; divide by sum(abs(x)) so absolute values sum to one.
         2: L2; divide by sqrt(sum(x**2)) to produce a unit-length vector.
        >2: Lp; divide by sum(abs(x)**p)**(1/p), with p = mode.
    """
    import numpy as np

    if isinstance(mode, bool):
        mode = 2 if mode else -1
    if not isinstance(mode, int):
        raise TypeError("normalize must be a bool or int")
    values = list(vector)
    if mode < 0:
        return values
    array = np.asarray(values, dtype=np.float32)
    if mode == 0:
        norm = float(np.max(np.abs(array))) if array.size else 0.0
    elif mode == 1:
        norm = float(np.sum(np.abs(array)))
    elif mode == 2:
        norm = float(np.linalg.norm(array))
    else:
        norm = float(np.sum(np.abs(array) ** mode) ** (1.0 / mode))
    return values if norm == 0 else ((array / norm) * (32760.0 if mode == 0 else 1.0)).tolist()

# Avoid "LookupError: unknown encoding: ascii" when open() called in a destructor
outnull_file = open(os.devnull, "w")
errnull_file = open(os.devnull, "w")

STDOUT_FILENO = 1
STDERR_FILENO = 2


class suppress_stdout_stderr(object):
    # NOTE: these must be "saved" here to avoid exceptions when using
    #       this context manager inside of a __del__ method
    sys = sys
    os = os

    def __init__(self, disable: bool = True):
        self.disable = disable

    # Oddly enough this works better than the contextlib version
    def __enter__(self):
        if self.disable:
            return self

        self.old_stdout_fileno_undup = STDOUT_FILENO
        self.old_stderr_fileno_undup = STDERR_FILENO

        self.old_stdout_fileno = self.os.dup(self.old_stdout_fileno_undup)
        self.old_stderr_fileno = self.os.dup(self.old_stderr_fileno_undup)

        self.old_stdout = self.sys.stdout
        self.old_stderr = self.sys.stderr

        self.os.dup2(outnull_file.fileno(), self.old_stdout_fileno_undup)
        self.os.dup2(errnull_file.fileno(), self.old_stderr_fileno_undup)

        self.sys.stdout = outnull_file
        self.sys.stderr = errnull_file
        return self

    def __exit__(self, *_):
        if self.disable:
            return

        # Check if sys.stdout and sys.stderr have fileno method
        self.sys.stdout = self.old_stdout
        self.sys.stderr = self.old_stderr

        self.os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        self.os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        self.os.close(self.old_stdout_fileno)
        self.os.close(self.old_stderr_fileno)


class MetaSingleton(type):
    """
    Metaclass for implementing the Singleton pattern.
    """

    _instances: Dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Singleton(object, metaclass=MetaSingleton):
    """
    Base class for implementing the Singleton pattern.
    """

    def __init__(self):
        super(Singleton, self).__init__()
