"""Helicon plugin architecture.

Third-party postprocessing modules and custom physics operators can be
registered here without modifying Helicon core.

Usage::

    import helicon.plugins as plugins

    # Register a custom postprocessing operator
    @plugins.register("postprocess", "my_metric")
    def my_metric(output_dir):
        return 42.0

    # Discover and call it
    fn = plugins.get("postprocess", "my_metric")
    result = fn("some/output/dir")

    # Or call directly
    result = plugins.call("postprocess", "my_metric", "some/output/dir")
"""

from helicon.plugins.registry import (
    PluginRegistry,
    call,
    get,
    list_plugins,
    register,
    register_instance,
)

__all__ = [
    "PluginRegistry",
    "call",
    "get",
    "list_plugins",
    "register",
    "register_instance",
]
