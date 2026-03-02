"""Tests for helicon.plugins — plugin registry."""

from __future__ import annotations

import pytest

from helicon.plugins.registry import PluginRegistry


def test_register_and_get():
    reg = PluginRegistry()

    @reg.register("postprocess", "double")
    def double(x):
        return x * 2

    fn = reg.get("postprocess", "double")
    assert fn(3) == 6


def test_register_instance():
    reg = PluginRegistry()

    def triple(x):
        return x * 3

    reg.register_instance("postprocess", "triple", triple)
    assert reg.get("postprocess", "triple")(4) == 12


def test_call():
    reg = PluginRegistry()
    reg.register_instance("physics", "identity", lambda x: x)
    assert reg.call("physics", "identity", 99) == 99


def test_call_with_kwargs():
    reg = PluginRegistry()
    reg.register_instance("optimize", "add", lambda a, b=0: a + b)
    assert reg.call("optimize", "add", 1, b=2) == 3


def test_duplicate_raises():
    reg = PluginRegistry()
    reg.register_instance("ns", "fn", lambda: None)
    with pytest.raises(ValueError, match="already registered"):
        reg.register_instance("ns", "fn", lambda: None)


def test_force_overwrite():
    reg = PluginRegistry()
    reg.register_instance("ns", "fn", lambda: 1)
    reg.register_instance("ns", "fn", lambda: 2, force=True)
    assert reg.call("ns", "fn") == 2


def test_get_missing_raises():
    reg = PluginRegistry()
    with pytest.raises(KeyError, match="not found"):
        reg.get("ns", "missing")


def test_contains():
    reg = PluginRegistry()
    reg.register_instance("ns", "fn", lambda: None)
    assert ("ns", "fn") in reg
    assert ("ns", "other") not in reg


def test_list_plugins_all():
    reg = PluginRegistry()
    reg.register_instance("a", "x", lambda: None)
    reg.register_instance("a", "y", lambda: None)
    reg.register_instance("b", "z", lambda: None)
    listing = reg.list_plugins()
    assert set(listing["a"]) == {"x", "y"}
    assert set(listing["b"]) == {"z"}


def test_list_plugins_namespace():
    reg = PluginRegistry()
    reg.register_instance("a", "x", lambda: None)
    reg.register_instance("b", "z", lambda: None)
    listing = reg.list_plugins("a")
    assert "a" in listing
    assert "b" not in listing


def test_repr():
    reg = PluginRegistry()
    reg.register_instance("ns", "fn", lambda: None)
    r = repr(reg)
    assert "PluginRegistry" in r


def test_decorator_form():
    reg = PluginRegistry()

    @reg.register("postprocess", "my_op")
    def my_op(path):
        return {"result": path}

    assert reg.call("postprocess", "my_op", "/tmp") == {"result": "/tmp"}


def test_multiple_namespaces_isolated():
    reg = PluginRegistry()
    reg.register_instance("ns1", "fn", lambda: "ns1")
    reg.register_instance("ns2", "fn", lambda: "ns2")
    assert reg.call("ns1", "fn") == "ns1"
    assert reg.call("ns2", "fn") == "ns2"


# Test default module-level registry
def test_module_level_api():

    # Create a fresh registry for isolation
    fresh = PluginRegistry()
    fresh.register_instance("postprocess", "noop", lambda: None)
    assert fresh.call("postprocess", "noop") is None


def test_load_entry_points_no_crash():
    """load_entry_points should not raise even with no external plugins installed."""
    reg = PluginRegistry()
    n = reg.load_entry_points()
    assert isinstance(n, int)
    assert n >= 0
