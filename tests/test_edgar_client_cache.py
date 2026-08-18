"""Regression tests for EdgarClient's disk cache and its expiry behavior.

No network involved — EdgarClient._get_raw is monkeypatched so these run fast and
deterministically. This exists because the cache-staleness bug (cache entries never
expired, so a stale ticker's filings/facts could hide silently forever) was found
and fixed by hand with no test backing it; a later refactor could reintroduce it
without anything here catching that.
"""

import json

import pytest

from tools.edgar_client import EdgarClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EDGAR_USER_AGENT", "Test Suite test@example.com")
    return EdgarClient(cache_dir=tmp_path)


def _patch_raw(monkeypatch, client, payloads):
    """Make client._get_raw return successive JSON payloads and count calls."""
    calls = {"count": 0}

    def fake_get_raw(url):
        calls["count"] += 1
        return json.dumps(payloads[min(calls["count"] - 1, len(payloads) - 1)])

    monkeypatch.setattr(client, "_get_raw", fake_get_raw)
    return calls


def test_first_call_hits_network_and_writes_wrapped_cache(client, monkeypatch):
    calls = _patch_raw(monkeypatch, client, [{"v": 1}])

    result = client._get_json("http://x", cache_key="k", max_age_seconds=3600)

    assert result == {"v": 1}
    assert calls["count"] == 1

    cached = json.loads((client.cache_dir / "k.json").read_text())
    assert cached["data"] == {"v": 1}
    assert "fetched_at" in cached


def test_fresh_cache_is_reused_without_hitting_network(client, monkeypatch):
    calls = _patch_raw(monkeypatch, client, [{"v": 1}, {"v": 2}])

    first = client._get_json("http://x", cache_key="k", max_age_seconds=3600)
    second = client._get_json("http://x", cache_key="k", max_age_seconds=3600)

    assert first == second == {"v": 1}
    assert calls["count"] == 1  # second call was served from cache


def test_expired_cache_triggers_a_refetch(client, monkeypatch):
    calls = _patch_raw(monkeypatch, client, [{"v": 1}, {"v": 2}])

    client._get_json("http://x", cache_key="k", max_age_seconds=3600)

    # Backdate the cache entry past its max age, simulating time passing.
    cache_path = client.cache_dir / "k.json"
    cached = json.loads(cache_path.read_text())
    cached["fetched_at"] -= 7200
    cache_path.write_text(json.dumps(cached))

    result = client._get_json("http://x", cache_key="k", max_age_seconds=3600)

    assert result == {"v": 2}  # got the fresh payload, not the stale one
    assert calls["count"] == 2


def test_max_age_zero_always_refetches(client, monkeypatch):
    calls = _patch_raw(monkeypatch, client, [{"v": 1}, {"v": 2}])

    client._get_json("http://x", cache_key="k", max_age_seconds=0)
    client._get_json("http://x", cache_key="k", max_age_seconds=0)

    assert calls["count"] == 2


def test_legacy_unwrapped_cache_file_is_treated_as_expired(client, monkeypatch):
    """Cache files written before the fetched_at wrapper existed (or corrupted ones
    missing it) should be refetched rather than crash or serve garbage."""
    calls = _patch_raw(monkeypatch, client, [{"v": "fresh"}])
    (client.cache_dir / "k.json").write_text(json.dumps({"some": "old-format-data"}))

    result = client._get_json("http://x", cache_key="k", max_age_seconds=3600)

    assert result == {"v": "fresh"}
    assert calls["count"] == 1


def test_get_document_caches_with_no_expiry(client, monkeypatch):
    """Filing documents are immutable once accessioned — no fetched_at/max_age
    machinery should apply, and a cached doc should never be refetched."""
    calls = {"count": 0}

    def fake_get_raw(url):
        calls["count"] += 1
        return "<html>doc</html>"

    monkeypatch.setattr(client, "_get_raw", fake_get_raw)

    first = client.get_document("http://x/doc.htm", cache_key="doc1")
    second = client.get_document("http://x/doc.htm", cache_key="doc1")

    assert first == second == "<html>doc</html>"
    assert calls["count"] == 1


def test_cache_disabled_never_reads_or_writes_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("EDGAR_USER_AGENT", "Test Suite test@example.com")
    client = EdgarClient(cache_dir=tmp_path, use_cache=False)
    calls = _patch_raw(monkeypatch, client, [{"v": 1}, {"v": 2}])

    client._get_json("http://x", cache_key="k", max_age_seconds=3600)
    client._get_json("http://x", cache_key="k", max_age_seconds=3600)

    assert calls["count"] == 2
    assert not (tmp_path / "k.json").exists()
