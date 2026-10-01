"""Registration of remote ggml devices before loading a llama model."""

import ctypes
import socket
import struct
import threading
from typing import List, Optional, Sequence, Union

from . import _ggml


# ggml-rpc.h protocol version and ggml-rpc.cpp command IDs at the bundled vendor revision.
_RPC_PROTO_MAJOR = 7
_RPC_PROTO_MINOR = 0
_RPC_CMD_HELLO = 14
_RPC_CMD_DEVICE_COUNT = 15
_RPC_CONN_CAPS_SIZE = 24
_RPC_TIMEOUT_SECONDS = 5.0
_RPC_MAX_REGISTERED_SERVERS = 16  # GGML_RPC_MAX_SERVERS in ggml-rpc.h

# The vendor registry owns these registrations for the lifetime of the process.
# Reuse them; unloading a registration while a model may reference its devices is unsafe.
_registered_servers: dict[str, int] = {}
_registration_lock = threading.RLock()


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    """Read one complete RPC field, raising if the peer closes it early."""
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("RPC server closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


def _rpc_request(sock: socket.socket, command: int, payload: bytes, response_size: int) -> bytes:
    """Send a framed vendor RPC request and read its fixed-size response.

    The wire format uses a one-byte command followed by a native-endian
    uint64 payload length. The response starts with its own uint64 length.
    """
    sock.sendall(bytes((command,)) + struct.pack("=Q", len(payload)) + payload)
    actual_size = struct.unpack("=Q", _recv_exact(sock, 8))[0]
    if actual_size != response_size:
        raise ConnectionError(f"unexpected RPC response size: {actual_size}")
    return _recv_exact(sock, response_size)


def _probe_rpc_server(endpoint: str) -> int:
    """Return the remote device count after checking the vendor handshake.

    This catches ordinary connection and protocol errors before calling the
    native registration function, which may abort on such errors. A server
    can still disconnect between this probe and native registration.
    """
    host, port_text = endpoint.rsplit(":", 1)
    try:
        # vendor transport.cpp connects with AF_INET/gethostbyname, so probe IPv4 too.
        ipv4_host = socket.gethostbyname(host)
        with socket.create_connection((ipv4_host, int(port_text)), timeout=_RPC_TIMEOUT_SECONDS) as sock:
            sock.settimeout(_RPC_TIMEOUT_SECONDS)
            # All-zero capabilities request TCP, including from RDMA-capable servers.
            hello = _rpc_request(sock, _RPC_CMD_HELLO, bytes(_RPC_CONN_CAPS_SIZE), 4 + _RPC_CONN_CAPS_SIZE)
            major, minor, patch = hello[:3]
            if major != _RPC_PROTO_MAJOR or minor > _RPC_PROTO_MINOR:
                raise RuntimeError(
                    f"RPC server {endpoint!r} has incompatible protocol {major}.{minor}.{patch}; "
                    f"client supports {_RPC_PROTO_MAJOR}.{_RPC_PROTO_MINOR}"
                )
            count = struct.unpack("=I", _rpc_request(sock, _RPC_CMD_DEVICE_COUNT, b"", 4))[0]
            if count == 0:
                raise RuntimeError(f"RPC server {endpoint!r} exposed no devices")
            return count
    except OSError as exc:
        raise ConnectionError(f"cannot connect to RPC server {endpoint!r}: {exc}") from exc


def normalize_rpc_servers(value: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Validate and normalize RPC endpoints while preserving their order.

    Accept a comma-separated string or a sequence of ``host:port`` strings.
    IPv6 literals are excluded because the bundled vendor transport uses
    IPv4 host resolution.
    """
    if value is None:
        return []
    if isinstance(value, str):
        servers = value.split(",")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        servers = list(value)
    else:
        raise TypeError("rpc_servers must be a comma-separated string or a sequence of strings")

    result: List[str] = []
    for server in servers:
        if not isinstance(server, str):
            raise TypeError("each RPC server must be a string")
        endpoint = server.strip()
        if endpoint.count(":") != 1:
            raise ValueError(f"invalid RPC server {endpoint!r}: expected host:port")
        host, port_text = endpoint.split(":", 1)
        if (
            not host
            or not host.isascii()
            or any(char.isspace() or char == "\x00" for char in host)
            or not port_text.isascii()
            or not port_text.isdecimal()
        ):
            raise ValueError(f"invalid RPC server {endpoint!r}: expected host:port")
        port = int(port_text)
        if not 1 <= port <= 65535:
            raise ValueError(f"invalid RPC server {endpoint!r}: port must be between 1 and 65535")
        if endpoint in result:
            raise ValueError(f"duplicate RPC server: {endpoint}")
        result.append(endpoint)
    return result


def register_rpc_devices(
    servers: Sequence[str],
    local_devices: Optional[Sequence[str]] = None,
    tensor_parallel: bool = False,
    max_devices: Optional[int] = None,
    expected_devices: Optional[int] = None,
) -> List[int]:
    """Return device pointers ordered as remote devices, then local GPUs.

    The caller must first initialize the ggml backend registry, including
    loading any dynamic backend plugins. ``local_devices`` selects local GPU
    names in order; ``None`` uses the default local GPUs and an empty sequence
    selects only remote devices. ``tensor_parallel`` filters CPU buffer devices
    from the local selection. ``max_devices`` enforces the native device limit,
    and ``expected_devices`` checks a caller-provided tensor split length.

    The ggml registry owns registrations for the process lifetime. The caller
    must pass these pointers as a null-terminated array in llama_model_params
    and keep that array alive until native model cleanup completes. Passing an
    explicit list also excludes RPC devices registered for other models.
    """
    servers = normalize_rpc_servers(servers)
    local_gpus: List[int] = []
    local_igpus: List[int] = []
    local_gpu_ids: set[bytes] = set()
    last_igpu_reg: Optional[int] = None
    # Scan the global registry, excluding remote devices from earlier models.
    for index in range(_ggml.ggml_backend_dev_count()):
        device = _ggml.ggml_backend_dev_get(index)
        reg_name = _ggml.ggml_backend_reg_name(
            _ggml.ggml_backend_dev_backend_reg(device)
        )
        if reg_name == b"RPC" or reg_name.startswith(b"RPC["):
            continue
        if tensor_parallel and (
            _ggml.ggml_backend_dev_buffer_type(device)
            == _ggml.ggml_backend_cpu_buffer_type()
        ):
            continue
        device_type = _ggml.ggml_backend_dev_type(device)
        if device_type == _ggml.GGMLBackendDevType.GGML_BACKEND_DEVICE_TYPE_GPU:
            props = _ggml.GGMLBackendDevProps()
            _ggml.ggml_backend_dev_get_props(device, ctypes.byref(props))
            if props.device_id:
                if props.device_id in local_gpu_ids:
                    continue
                local_gpu_ids.add(props.device_id)
            local_gpus.append(device)
        elif device_type == _ggml.GGMLBackendDevType.GGML_BACKEND_DEVICE_TYPE_IGPU:
            owner = _ggml.ggml_backend_dev_backend_reg(device)
            # Match llama.cpp's integrated GPU deduplication policy.
            if not local_igpus or owner == last_igpu_reg:
                local_igpus.append(device)
                last_igpu_reg = owner

    available = local_gpus + (local_igpus if not local_gpus else [])
    if local_devices is not None:
        if isinstance(local_devices, (str, bytes)):
            raise TypeError("local_devices must be a sequence of device names")
        by_name = {
            _ggml.ggml_backend_dev_name(dev).decode("utf-8"): dev
            for dev in local_gpus + local_igpus
        }
        selected: List[int] = []
        for name in local_devices:
            if not isinstance(name, str):
                raise TypeError("each local device name must be a string")
            if name not in by_name:
                raise ValueError(f"unknown or unavailable local GPU device: {name!r}")
            if by_name[name] in selected:
                raise ValueError(f"duplicate local GPU device: {name!r}")
            selected.append(by_name[name])
        available = selected

    with _registration_lock:
        rpc_reg = _ggml.ggml_backend_reg_by_name(b"RPC")
        if not rpc_reg:
            raise RuntimeError(
                "RPC backend is unavailable; load ggml backends before RPC "
                "registration, or rebuild with -DGGML_RPC=ON"
            )

        # Dynamic backends expose add_server through the registry, not ggml.dll.
        address = _ggml.ggml_backend_reg_get_proc_address(
            rpc_reg, b"ggml_backend_rpc_add_server"
        )
        if not address:
            raise RuntimeError("RPC backend does not expose ggml_backend_rpc_add_server")
        add_server = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)(address)

        # Probe every server before registering any new endpoint, avoiding partial
        # process-global registration for ordinary connection and version failures.
        expected_counts = {endpoint: _probe_rpc_server(endpoint) for endpoint in servers}
        selected_count = sum(expected_counts.values()) + len(available)
        if max_devices is not None and selected_count > max_devices:
            raise ValueError(
                f"RPC and local GPU devices exceed LLAMA_MAX_DEVICES={max_devices}"
            )
        if expected_devices is not None and selected_count != expected_devices:
            raise ValueError(
                f"tensor_split must have {selected_count} values in RPC mode"
            )
        # A registration may already exist even when this module's cache is empty.
        new_endpoints = [
            endpoint
            for endpoint in servers
            if endpoint not in _registered_servers
            and not _ggml.ggml_backend_reg_by_name(f"RPC[{endpoint}]".encode("utf-8"))
        ]
        if len(_registered_servers) + len(new_endpoints) > _RPC_MAX_REGISTERED_SERVERS:
            raise RuntimeError(
                "too many RPC endpoints registered in this process; "
                "restart Python to use different servers"
            )

        remote_devices: List[int] = []
        for endpoint in servers:
            reg = _registered_servers.get(endpoint)
            if reg is None:
                reg = _ggml.ggml_backend_reg_by_name(f"RPC[{endpoint}]".encode("utf-8"))
            if not reg:
                reg = add_server(endpoint.encode("utf-8"))
                if not reg:
                    raise RuntimeError(f"RPC server {endpoint!r} exposed no devices")
                _ggml.ggml_backend_register(reg)
            _registered_servers[endpoint] = reg
            count = _ggml.ggml_backend_reg_dev_count(reg)
            if count != expected_counts[endpoint]:
                raise RuntimeError(
                    f"RPC server {endpoint!r} device count changed from "
                    f"{expected_counts[endpoint]} to {count}; restart the Python process"
                )
            for index in range(count):
                remote_devices.append(_ggml.ggml_backend_reg_dev_get(reg, index))

    return remote_devices + available
