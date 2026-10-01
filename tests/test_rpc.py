import socket
import struct

import pytest

from llama_cpp import _ggml, _rpc


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        (["host-a:50052", "host-b:50053"], ["host-a:50052", "host-b:50053"]),
        ("host-a:50052, host-b:50053", ["host-a:50052", "host-b:50053"]),
    ],
)
def test_normalize_rpc_servers(value, expected):
    assert _rpc.normalize_rpc_servers(value) == expected


@pytest.mark.parametrize("value", ["", "host", "host:0", "host:65536", "host:abc", "host:1,", "bad host:1", "host\x00:1", "例子:1"])
def test_normalize_rpc_servers_rejects_bad_endpoints(value):
    with pytest.raises(ValueError):
        _rpc.normalize_rpc_servers(value)


def test_registration_excludes_other_models_rpc_devices(monkeypatch):
    registrations = {"server-a:50052": 10, "server-b:50052": 20}
    names = {10: b"RPC[server-a:50052]", 20: b"RPC[server-b:50052]", 30: b"CUDA"}
    owners = {101: 10, 201: 20, 301: 30, 401: 30}
    registered = []

    monkeypatch.setattr(_rpc, "_registered_servers", {})
    monkeypatch.setattr(_rpc, "_probe_rpc_server", lambda endpoint: 1)
    monkeypatch.setattr(_ggml, "ggml_backend_reg_by_name", lambda name: 1 if name == b"RPC" else None)
    monkeypatch.setattr(_ggml, "ggml_backend_reg_get_proc_address", lambda reg, name: 99)
    monkeypatch.setattr(_rpc.ctypes, "CFUNCTYPE", lambda *args: lambda ptr: lambda endpoint: registrations[endpoint.decode()])
    monkeypatch.setattr(_ggml, "ggml_backend_register", registered.append)
    monkeypatch.setattr(_ggml, "ggml_backend_reg_dev_count", lambda reg: 1)
    monkeypatch.setattr(_ggml, "ggml_backend_reg_dev_get", lambda reg, index: {10: 101, 20: 201}[reg])
    monkeypatch.setattr(_ggml, "ggml_backend_dev_count", lambda: 4)
    monkeypatch.setattr(_ggml, "ggml_backend_dev_get", lambda index: [101, 201, 301, 401][index])
    monkeypatch.setattr(_ggml, "ggml_backend_dev_backend_reg", lambda dev: owners[dev])
    monkeypatch.setattr(_ggml, "ggml_backend_reg_name", lambda reg: names[reg])
    monkeypatch.setattr(_ggml, "ggml_backend_dev_name", lambda dev: {301: b"CUDA0"}[dev])
    monkeypatch.setattr(_ggml, "ggml_backend_dev_type", lambda dev: 0 if dev == 401 else 1)
    monkeypatch.setattr(_ggml, "ggml_backend_dev_get_props", lambda dev, props: None)
    with pytest.raises(ValueError, match="unknown or unavailable"):
        _rpc.register_rpc_devices(["server-b:50052"], local_devices=["unknown"])
    assert registered == []

    with pytest.raises(ValueError, match="tensor_split must have"):
        _rpc.register_rpc_devices(["server-b:50052"], expected_devices=1)
    assert registered == []

    monkeypatch.setattr(_rpc, "_RPC_MAX_REGISTERED_SERVERS", 0)
    with pytest.raises(RuntimeError, match="too many RPC endpoints"):
        _rpc.register_rpc_devices(["server-b:50052"])
    assert registered == []
    monkeypatch.setattr(_rpc, "_RPC_MAX_REGISTERED_SERVERS", 16)

    assert _rpc.register_rpc_devices(["server-b:50052"]) == [201, 301]
    assert registered == [20]
    assert _rpc.register_rpc_devices(["server-b:50052"], local_devices=[]) == [201]
    assert registered == [20]


def test_probe_checks_vendor_handshake_and_device_count(monkeypatch):
    def response(payload):
        return struct.pack("=Q", len(payload)) + payload

    class FakeSocket:
        def __init__(self, major=7, device_count=2):
            self.sent = []
            self.data = bytearray(
                response(bytes((major, 0, 0, 0)) + bytes(24))
                + response(struct.pack("=I", device_count))
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def settimeout(self, timeout):
            pass

        def sendall(self, data):
            self.sent.append(data)

        def recv(self, size):
            chunk = self.data[:size]
            del self.data[:size]
            return bytes(chunk)

    sock = FakeSocket()
    monkeypatch.setattr(socket, "gethostbyname", lambda host: "127.0.0.1")
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: sock)
    assert _rpc._probe_rpc_server("host:50052") == 2
    assert sock.sent[0][0] == 14
    assert sock.sent[1][0] == 15

    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: FakeSocket(major=8))
    with pytest.raises(RuntimeError, match="incompatible protocol"):
        _rpc._probe_rpc_server("host:50052")


def test_row_split_rejected_before_rpc_registration():
    from llama_cpp import Llama, llama_split_mode

    with pytest.raises(ValueError, match="do not support LLAMA_SPLIT_MODE_ROW"):
        Llama(
            model_path="unused.gguf",
            rpc_servers="host:50052",
            split_mode=llama_split_mode.LLAMA_SPLIT_MODE_ROW,
            verbose=False,
        )


def test_device_array_outlives_native_model_cleanup():
    from llama_cpp import Llama

    model = Llama.__new__(Llama)
    devices = object()
    model._c_rpc_devices = devices
    model.model_params = object()

    class NativeResource:
        def close(self):
            assert model._c_rpc_devices is devices
            assert model.model_params is not None

    model._stack = NativeResource()
    model.close()
    assert model._c_rpc_devices is None
