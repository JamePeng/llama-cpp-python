import importlib


def test_import_mtmd_cpp():
    module = importlib.import_module("llama_cpp.mtmd_cpp")

    assert module is not None
