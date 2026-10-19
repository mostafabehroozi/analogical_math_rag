import io
import sys
import threading
import types

from src.context_logger import guard_ipykernel_thread_output


class FakeOutStream(io.StringIO):
    """The part of ipykernel 6.29.5's OutStream that routes a thread's output.

    `parent_header` is ipykernel/iostream.py lines 505-522 without the asyncio
    ContextVar, which a plain thread never has set.
    """

    def __init__(self):
        super().__init__()
        self._thread_to_parent = {}
        self._thread_to_parent_header = {}
        self._parent_header_global = {"cell": "running"}

    @property
    def parent_header(self):
        try:
            identity = threading.current_thread().ident
            while identity in self._thread_to_parent:
                identity = self._thread_to_parent[identity]
            return self._thread_to_parent_header[identity]
        except KeyError:
            return self._parent_header_global


def test_guard_releases_a_thread_looping_on_a_thread_map_cycle(monkeypatch):
    stream = FakeOutStream()
    result = {}

    def print_from_thread():
        # Stale entries of exited threads whose idents were reused.
        me = threading.get_ident()
        stream._thread_to_parent[me] = 1
        stream._thread_to_parent[1] = me
        result["header"] = stream.parent_header

    thread = threading.Thread(target=print_from_thread, daemon=True)
    thread.start()
    thread.join(0.2)
    assert thread.is_alive(), "ipykernel's lookup should loop on the cycle"

    monkeypatch.setattr(sys, "stdout", stream)
    assert guard_ipykernel_thread_output() == 1
    thread.join(5)
    assert not thread.is_alive()
    assert result["header"] == {"cell": "running"}


def test_guarded_maps_ignore_writes_and_the_guard_applies_once(monkeypatch):
    stream = FakeOutStream()
    monkeypatch.setattr(sys, "stdout", stream)
    assert guard_ipykernel_thread_output() == 1

    # ipykernel's thread-start hook keeps writing to both maps.
    me = threading.get_ident()
    stream._thread_to_parent[me] = 1
    stream._thread_to_parent[1] = me
    stream._thread_to_parent_header[me] = {"cell": "earlier"}

    assert stream._thread_to_parent == {}
    assert stream._thread_to_parent_header == {}
    assert stream.parent_header == {"cell": "running"}
    assert guard_ipykernel_thread_output() == 0


def test_guard_reaches_the_kernel_streams_behind_a_wrapped_sys_stdout(monkeypatch):
    stdout, stderr = FakeOutStream(), FakeOutStream()
    shell = types.SimpleNamespace(kernel=types.SimpleNamespace(_stdout=stdout, _stderr=stderr))
    monkeypatch.setitem(sys.modules, "IPython", types.SimpleNamespace(get_ipython=lambda: shell))
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    assert guard_ipykernel_thread_output() == 2
    stdout._thread_to_parent[1] = 2
    stderr._thread_to_parent[1] = 2
    assert stdout._thread_to_parent == {} and stderr._thread_to_parent == {}


def test_guard_leaves_streams_without_thread_maps_alone(monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.delitem(sys.modules, "IPython", raising=False)

    assert guard_ipykernel_thread_output() == 0
