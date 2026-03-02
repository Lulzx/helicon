"""Plugin registry for Helicon.

Provides a global plugin registry with namespace-based organisation.
Plugins can be:
  - registered programmatically via :func:`register` / :func:`register_instance`
  - discovered via Python entry points (``helicon.plugins`` group)

Namespaces
----------
``"postprocess"``
    Custom postprocessing operators that accept an ``output_dir`` path and
    return a scalar or dict of additional metrics.

``"physics"``
    Custom physics sub-models (e.g. anomalous transport closures) that are
    called by the hybrid solver.

``"optimize"``
    Custom objective or constraint functions for the optimisation pipeline.

Any other namespace string is also valid — the registry is open.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level default registry instance
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY: PluginRegistry | None = None


def _default() -> PluginRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = PluginRegistry()
        _DEFAULT_REGISTRY.load_entry_points()
    return _DEFAULT_REGISTRY


# ---------------------------------------------------------------------------
# PluginRegistry
# ---------------------------------------------------------------------------


class PluginRegistry:
    """Namespace-keyed registry of callable plugins.

    Parameters
    ----------
    entry_point_group : str
        Python entry-point group name to auto-discover plugins from.
    """

    def __init__(self, entry_point_group: str = "helicon.plugins") -> None:
        self._store: dict[str, dict[str, Callable]] = defaultdict(dict)
        self._entry_point_group = entry_point_group

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, namespace: str, name: str, *, force: bool = False) -> Callable:
        """Decorator that registers a callable under *namespace/name*.

        Parameters
        ----------
        namespace : str
            Plugin namespace (e.g. ``"postprocess"``).
        name : str
            Unique plugin name within the namespace.
        force : bool
            Allow overwriting an existing registration.

        Returns
        -------
        Callable
            A decorator that registers the wrapped function.

        Example
        -------
        >>> registry = PluginRegistry()
        >>> @registry.register("postprocess", "my_fn")
        ... def my_fn(output_dir): ...
        """

        def decorator(fn: Callable) -> Callable:
            if name in self._store[namespace] and not force:
                raise ValueError(
                    f"Plugin '{namespace}/{name}' already registered. "
                    "Pass force=True to overwrite."
                )
            self._store[namespace][name] = fn
            logger.debug("Registered plugin %s/%s → %s", namespace, name, fn)
            return fn

        return decorator

    def register_instance(
        self, namespace: str, name: str, fn: Callable, *, force: bool = False
    ) -> None:
        """Register a callable directly (non-decorator form).

        Parameters
        ----------
        namespace : str
            Plugin namespace.
        name : str
            Plugin name.
        fn : callable
            The plugin callable.
        force : bool
            Allow overwriting an existing registration.
        """
        if name in self._store[namespace] and not force:
            raise ValueError(
                f"Plugin '{namespace}/{name}' already registered. "
                "Pass force=True to overwrite."
            )
        self._store[namespace][name] = fn
        logger.debug("Registered plugin %s/%s → %s", namespace, name, fn)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def load_entry_points(self) -> int:
        """Load plugins registered via Python entry points.

        Looks for the group ``helicon.plugins``.  Each entry point should
        expose a callable that accepts *(registry)* and registers its plugins.

        Returns
        -------
        int
            Number of entry-point plugins loaded.
        """
        n = 0
        try:
            eps = importlib.metadata.entry_points(group=self._entry_point_group)
        except Exception:
            return 0
        for ep in eps:
            try:
                loader = ep.load()
                loader(self)
                n += 1
                logger.debug("Loaded entry-point plugin: %s", ep.name)
            except Exception as exc:
                logger.warning("Failed to load entry-point plugin %s: %s", ep.name, exc)
        return n

    # ------------------------------------------------------------------
    # Lookup & call
    # ------------------------------------------------------------------

    def get(self, namespace: str, name: str) -> Callable:
        """Retrieve a registered plugin callable.

        Parameters
        ----------
        namespace : str
        name : str

        Returns
        -------
        Callable

        Raises
        ------
        KeyError
            If the plugin is not found.
        """
        try:
            return self._store[namespace][name]
        except KeyError:
            available = list(self._store[namespace].keys())
            raise KeyError(
                f"Plugin '{namespace}/{name}' not found. "
                f"Available in '{namespace}': {available}"
            ) from None

    def call(self, namespace: str, name: str, *args: Any, **kwargs: Any) -> Any:
        """Look up and immediately call a plugin.

        Parameters
        ----------
        namespace : str
        name : str
        *args, **kwargs
            Forwarded to the plugin callable.
        """
        return self.get(namespace, name)(*args, **kwargs)

    def list_plugins(self, namespace: str | None = None) -> dict[str, list[str]]:
        """Return a mapping of namespace → [plugin names].

        Parameters
        ----------
        namespace : str, optional
            If given, only return that namespace.
        """
        if namespace is not None:
            return {namespace: list(self._store.get(namespace, {}).keys())}
        return {ns: list(names.keys()) for ns, names in self._store.items()}

    def __contains__(self, item: tuple[str, str]) -> bool:
        """Check ``(namespace, name) in registry``."""
        namespace, name = item
        return name in self._store.get(namespace, {})

    def __repr__(self) -> str:
        counts = {ns: len(names) for ns, names in self._store.items()}
        return f"PluginRegistry({counts})"


# ---------------------------------------------------------------------------
# Module-level convenience functions (operate on the default registry)
# ---------------------------------------------------------------------------


def register(namespace: str, name: str, *, force: bool = False) -> Callable:
    """Decorator that registers *fn* in the default registry."""
    return _default().register(namespace, name, force=force)


def register_instance(namespace: str, name: str, fn: Callable, *, force: bool = False) -> None:
    """Register a callable in the default registry."""
    _default().register_instance(namespace, name, fn, force=force)


def get(namespace: str, name: str) -> Callable:
    """Retrieve a plugin from the default registry."""
    return _default().get(namespace, name)


def call(namespace: str, name: str, *args: Any, **kwargs: Any) -> Any:
    """Look up and call a plugin from the default registry."""
    return _default().call(namespace, name, *args, **kwargs)


def list_plugins(namespace: str | None = None) -> dict[str, list[str]]:
    """List plugins in the default registry."""
    return _default().list_plugins(namespace)
