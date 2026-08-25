from llama_cpp import llama_cpp as llama_cpp_lib
from llama_cpp.llama_cache import HybridCheckpointCache


def test_hybrid_checkpoint_preserves_native_positions_and_removes_suffix(monkeypatch):
    state = bytes([1, 2, 3, 4])
    memory = object()
    removals = []

    monkeypatch.setattr(llama_cpp_lib, "llama_state_seq_get_size_ext", lambda ctx, seq_id, flags: len(state))

    def get_data(ctx, buffer, size, seq_id, flags):
        buffer[:size] = state
        return size

    monkeypatch.setattr(llama_cpp_lib, "llama_state_seq_get_data_ext", get_data)
    monkeypatch.setattr(llama_cpp_lib, "llama_state_seq_set_data_ext", lambda ctx, buffer, size, seq_id, flags: size)
    monkeypatch.setattr(llama_cpp_lib, "llama_get_memory", lambda ctx: memory)
    monkeypatch.setattr(llama_cpp_lib, "llama_memory_seq_pos_min", lambda mem, seq_id: 11)
    monkeypatch.setattr(llama_cpp_lib, "llama_memory_seq_pos_max", lambda mem, seq_id: 42)

    def memory_seq_rm(mem, seq_id, p0, p1):
        removals.append((mem, seq_id, p0, p1))
        return True

    monkeypatch.setattr(llama_cpp_lib, "llama_memory_seq_rm", memory_seq_rm)

    cache = HybridCheckpointCache(ctx=object())
    assert cache.save_checkpoint(current_pos=17, tokens=list(range(17)), seq_id=3)

    checkpoint = cache.checkpoints[0]
    assert checkpoint.pos == 17
    assert checkpoint.pos_min == 11
    assert checkpoint.pos_max == 42

    assert cache.restore_checkpoint(checkpoint, seq_id=3)
    assert removals == [(memory, 3, 43, -1)]


def test_hybrid_checkpoint_restore_reports_suffix_removal_failure(monkeypatch):
    state = bytes([1, 2])
    monkeypatch.setattr(llama_cpp_lib, "llama_state_seq_get_size_ext", lambda ctx, seq_id, flags: len(state))

    def get_data(ctx, buffer, size, seq_id, flags):
        buffer[:size] = state
        return size

    monkeypatch.setattr(llama_cpp_lib, "llama_state_seq_get_data_ext", get_data)
    monkeypatch.setattr(llama_cpp_lib, "llama_state_seq_set_data_ext", lambda ctx, buffer, size, seq_id, flags: size)
    monkeypatch.setattr(llama_cpp_lib, "llama_get_memory", lambda ctx: object())
    monkeypatch.setattr(llama_cpp_lib, "llama_memory_seq_pos_min", lambda mem, seq_id: 0)
    monkeypatch.setattr(llama_cpp_lib, "llama_memory_seq_pos_max", lambda mem, seq_id: 8)
    monkeypatch.setattr(llama_cpp_lib, "llama_memory_seq_rm", lambda mem, seq_id, p0, p1: False)

    cache = HybridCheckpointCache(ctx=object())
    assert cache.save_checkpoint(current_pos=9, tokens=list(range(9)))
    assert not cache.restore_checkpoint(cache.checkpoints[0])
