"""Common-prefix selection and lazy snapshot reads across cache backends."""

import random
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from llama_cpp.llama import LlamaState
from llama_cpp.llama_cache import BaseLlamaCache, LlamaDiskCache, LlamaRAMCache, LlamaTrieCache


def snapshot(tokens):
    return LlamaState(
        input_ids=np.array(tokens, dtype=np.intc),
        scores=np.zeros((1, 3), dtype=np.single),
        n_tokens=len(tokens), llama_state=b"state", llama_state_size=5, seed=0,
    )


@pytest.fixture(params=[LlamaRAMCache, LlamaTrieCache, LlamaDiskCache])
def cache(request, tmp_path):
    cls = request.param
    instance = cls(str(tmp_path)) if cls is LlamaDiskCache else cls()
    yield instance
    if cls is LlamaDiskCache:
        instance.cache.close()


def test_cache_backends_select_longest_common_prefix(cache):
    cache[[1, 2, 3, 4]] = snapshot([1, 2, 3, 4])
    cache[[1, 8]] = snapshot([1, 8])
    for query, expected in (
        ([1, 2, 3, 4], [1, 2, 3, 4]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4]),
        ([1, 2, 3], [1, 2, 3, 4]),
        ([1, 2, 9], [1, 2, 3, 4]),
        ([1, 8, 9], [1, 8]),
        ([9], None),
        ([], None),
    ):
        assert (query in cache) is (expected is not None), query
        if expected is None:
            with pytest.raises(KeyError):
                cache[query]
        else:
            assert cache[query].input_ids.tolist() == expected, query


def test_equal_prefix_selection_is_stable_across_reads_and_replacement(cache):
    cache[[1, 2, 3]] = snapshot([1, 2, 3])
    cache[[1, 2, 4]] = snapshot([1, 2, 4])
    cache[[1, 2, 3]]  # LRU changes must not change tie selection.
    cache[[1, 2, 3]] = snapshot([1, 2, 3])
    assert cache[[1, 2, 9]].input_ids.tolist() == [1, 2, 3]
    with pytest.raises(KeyError):
        cache.get_matching_state([1, 2, 9], min_prefix_length=2)
    assert cache.get_matching_state([1, 2, 9], min_prefix_length=1).input_ids.tolist() == [1, 2, 3]


def test_zero_length_prefix_is_not_a_hit(cache):
    cache[[]] = snapshot([])
    assert [] not in cache and [1] not in cache


def test_long_prefix_selection_and_minimum_threshold(cache):
    tokens = list(range(1024))
    cache[tokens] = snapshot(tokens)
    for prefix_length in (1, 31, 32, 33, 287, 288, 289, 543, 544, 1023, 1024):
        # Lists and NumPy token arrays must select the same saved state.
        query = tokens[:prefix_length] + [2048]
        for key in (query, np.asarray(query, dtype=np.intc)):
            state = cache.get_matching_state(key, min_prefix_length=prefix_length - 1)
            assert state.input_ids.tolist() == tokens, prefix_length
            with pytest.raises(KeyError):
                cache.get_matching_state(key, min_prefix_length=prefix_length)
        # Ending the query inside a saved history is also a common-prefix hit.
        assert cache[tokens[:prefix_length]].input_ids.tolist() == tokens, prefix_length


def test_disk_rejects_unhelpful_candidates_before_deserializing(tmp_path, monkeypatch):
    cache = LlamaDiskCache(str(tmp_path))
    cache[[1, 2, 3]] = snapshot([1, 2, 3])
    reads = []
    original = type(cache.cache).__getitem__

    def read(instance, key):
        reads.append(key)
        return original(instance, key)

    monkeypatch.setattr(type(cache.cache), "__getitem__", read)
    try:
        for query, minimum in (([1, 2, 9], 2), ([1, 2], 2), ([9], 0)):
            with pytest.raises(KeyError):
                cache.get_matching_state(query, minimum)
        assert [1, 2, 9] in cache
        assert reads == []
        assert cache.get_matching_state([1, 2, 9], 1).n_tokens == 3
        assert reads == [(1, 2, 3)]
    finally:
        cache.cache.close()


@pytest.mark.parametrize("cls", [LlamaRAMCache, LlamaTrieCache])
def test_oversized_insert_and_replacement_preserve_existing_cache(cls):
    state = snapshot([1, 2])
    instance = cls(capacity_bytes=state.nbytes * 2)
    instance[[1, 2]] = state
    instance[[3, 4]] = snapshot([3, 4])
    oversized = snapshot(list(range(100)))
    instance[[1, 2]] = oversized
    instance[[5]] = oversized
    assert instance.cache_size == state.nbytes * 2
    assert instance[[1, 2]] is state
    assert [3, 4] in instance and [5] not in instance


def test_trie_matches_ram_through_branch_eviction_and_reinsertion():
    rng = random.Random(42)
    ram, trie = LlamaRAMCache(capacity_bytes=5), LlamaTrieCache(capacity_bytes=5)
    for step in range(200):
        key = tuple(rng.randrange(4) for _ in range(rng.randrange(1, 8)))
        state = SimpleNamespace(nbytes=1, llama_state_size=1, marker=step)
        ram[key] = trie[key] = state
        for _ in range(3):
            query = tuple(rng.randrange(4) for _ in range(rng.randrange(8)))
            assert (query in trie) == (query in ram)
            if query in ram:
                assert trie[query] is ram[query]
        assert trie.cache_size == ram.cache_size
        assert list(trie.lru_tracker) == list(ram.cache_state)


@pytest.mark.parametrize("keep_oldest", [False, True])
def test_wide_trie_eviction_reinsertion_and_candidate_heap_stay_consistent(keep_oldest):
    ram, trie = LlamaRAMCache(capacity_bytes=32), LlamaTrieCache(capacity_bytes=32)
    anchor = (1, 0)
    state = SimpleNamespace(nbytes=1, llama_state_size=1, marker=-1)
    ram[anchor] = trie[anchor] = state
    rng = random.Random(7)
    for step in range(512):
        key = (1, rng.randrange(48), rng.randrange(4))
        state = SimpleNamespace(nbytes=1, llama_state_size=1, marker=step)
        ram[key] = trie[key] = state
        if keep_oldest:
            assert trie[anchor] is ram[anchor]
        for query in ((1, 99), key[:2], (1, rng.randrange(48), 99)):
            assert trie[query] is ram[query]
        assert list(trie.lru_tracker) == list(ram.cache_state)
        assert trie.cache_size == ram.cache_size
        nodes = [trie.root]
        while nodes:
            node = nodes.pop()
            nodes.extend(node.children.values())
            if node._child_entries is not None:
                assert len(node._child_entries) <= 2 * len(node.children)

    # Shrinking below the heap threshold must retain correct prefix selection.
    trie.capacity_bytes = ram.capacity_bytes = 4
    state = SimpleNamespace(nbytes=1, llama_state_size=1, marker=512)
    ram[(1, 100)] = trie[(1, 100)] = state
    assert trie[(1, 99)] is ram[(1, 99)]
    assert list(trie.lru_tracker) == list(ram.cache_state)


@pytest.mark.parametrize("short_first", [False, True])
def test_evicted_prefix_merges_paths_without_changing_surviving_state(short_first):
    cache = LlamaTrieCache(capacity_bytes=2)
    prefix = tuple(range(128))
    longer = prefix + tuple(range(128, 512))
    short_state = SimpleNamespace(nbytes=1, llama_state_size=1, marker="short")
    long_state = SimpleNamespace(nbytes=1, llama_state_size=1, marker="long")
    entries = [(prefix, short_state), (longer, long_state)]
    for key, state in (entries if short_first else entries[::-1]):
        cache[key] = state
    assert cache[longer] is long_state  # Keep the descendant, evict the prefix.
    cache[(999,)] = SimpleNamespace(nbytes=1, llama_state_size=1)
    assert list(cache.lru_tracker) == [longer, (999,)]
    for query in (prefix, prefix + (999,), longer, longer + (999,)):
        assert cache[query] is long_state
    assert len(cache.root.children[0].tokens) == len(longer)


@pytest.mark.parametrize("evict_branch", [False, True])
def test_split_edge_merges_after_leaf_eviction_and_releases_snapshot(evict_branch):
    prefix = list(range(256))
    branch = prefix[:64] + [999] + prefix[65:]
    original = snapshot(prefix)
    alternate = snapshot(branch)
    cache = LlamaTrieCache(capacity_bytes=original.nbytes * 2)
    cache[prefix] = original
    cache[branch] = alternate
    removed = alternate if evict_branch else original
    retained = original if evict_branch else alternate
    removed_ref = weakref.ref(removed)
    retained_key = prefix if evict_branch else branch
    cache[retained_key]
    del removed, original, alternate
    cache[[2000] * 256] = snapshot([2000] * 256)
    assert removed_ref() is None
    assert cache[prefix[:64]] is retained
    assert cache[retained_key] is retained
    assert len(cache.root.children[0].tokens) == 256


@pytest.mark.parametrize("seed", [2, 7, 19])
def test_long_compressed_paths_match_ram_through_splits_and_evictions(seed):
    rng = random.Random(seed)
    ram, trie = LlamaRAMCache(capacity_bytes=24), LlamaTrieCache(capacity_bytes=24)
    shared = tuple(range(512))
    for step in range(400):
        key = shared + (rng.randrange(48) + 1000,) + tuple(range(rng.randrange(128)))
        if step % 5 == 0:
            key = key[:rng.choice([0, 1, 31, 32, 128, 512])]
        state = SimpleNamespace(nbytes=rng.randrange(1, 4), llama_state_size=1, marker=step)
        ram[key] = trie[key] = state
        queries = [key, key[:32], shared + (rng.randrange(48) + 1000, 9999)]
        for query in queries:
            assert (query in trie) == (query in ram)
            if query in ram:
                assert trie[query] is ram[query]
        assert list(trie.lru_tracker) == list(ram.cache_state)
        assert trie.cache_size == ram.cache_size
        # Cache churn must not leave empty branches or per-token chains behind.
        pending, node_count = [trie.root], 0
        while pending:
            node = pending.pop()
            node_count += 1
            if node is not trie.root and node.state is None:
                assert len(node.children) >= 2
            for token, child in node.children.items():
                assert child.tokens and child.tokens[0] == token
                pending.append(child)
        assert node_count <= 2 * len(trie.lru_tracker) + 1


def test_nested_cached_prefixes_survive_replacement_and_bulk_eviction():
    cache = LlamaTrieCache(capacity_bytes=128)
    for length in range(1, 129):
        cache[(1,) * length] = SimpleNamespace(nbytes=1, llama_state_size=1, marker=length)
    replacement = SimpleNamespace(nbytes=1, llama_state_size=1, marker="replacement")
    cache[(1,) * 64] = replacement
    assert cache[(1,) * 64] is replacement
    assert cache[(1,) * 65].marker == 65
    assert cache[(1,) * 128 + (2,)].marker == 128
    large = SimpleNamespace(nbytes=128, llama_state_size=128, marker="large")
    cache[(2,) * 256] = large
    assert list(cache.lru_tracker) == [(2,) * 256]
    assert cache.cache_size == 128
    assert (1,) not in cache
    assert cache[(2,) * 32] is large
    assert len(cache.root.children) == 1
    assert len(cache.root.children[2].tokens) == 256


def test_successive_appends_remain_valid_when_ancestors_are_evicted_and_merged():
    ram, trie = LlamaRAMCache(capacity_bytes=2), LlamaTrieCache(capacity_bytes=2)
    key = tuple(range(64))
    for step in range(100):
        key += (step + 1000,)
        state = SimpleNamespace(nbytes=1, llama_state_size=1, marker=step)
        ram[key] = trie[key] = state
        assert trie[key] is ram[key]
        assert trie[key[:64]] is ram[key[:64]]
        assert trie[key + (9999,)] is ram[key + (9999,)]
        assert list(trie.lru_tracker) == list(ram.cache_state)
        assert trie.cache_size == ram.cache_size


def test_custom_dictionary_cache_keeps_default_prefix_interface():
    class CustomCache(BaseLlamaCache):
        cache_size = 0

        def __getitem__(self, key):
            return snapshot([1, 2])

        def __contains__(self, key):
            return True

        def __setitem__(self, key, value):
            pass

    instance = CustomCache()
    assert instance.get_matching_state([1, 2, 3], 1).n_tokens == 2
    with pytest.raises(KeyError):
        instance.get_matching_state([1, 2, 3], 2)
