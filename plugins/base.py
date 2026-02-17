"""Plugin base class and registry (Phase 5C.4).

Provides:
  - ``PluginBase``: abstract base all plugins extend
  - ``PluginType``: enum for channel vs tool plugins
  - ``PluginRegistry``: singleton tracking loaded plugins

Different from ``evolution/skills.py`` — skills are auto-generated
conversation patterns.  Plugins are developer-authored extensions
for new channels or tools.
"""
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class PluginType(str, Enum):
    """Plugin category."""
    CHANNEL = "channel"
    TOOL = "tool"


class PluginBase(ABC):
    """Abstract base class for all Molly plugins.

    Subclass this and implement ``register()`` to wire the plugin into
    Molly's runtime.  Set ``PLUGIN_CLASS = YourPlugin`` at module level
    so the loader can discover it.
    """

    # Required metadata — subclasses must set these
    name: str = ""
    plugin_type: PluginType = PluginType.TOOL
    version: str = "0.1.0"
    description: str = ""

    @abstractmethod
    def register(self, registry: PluginRegistry) -> None:
        """Register this plugin with the given registry.

        For channel plugins: register a Channel subclass with
        ``channels.base.registry``.

        For tool plugins: register MCP tool definitions.
        """
        ...

    def on_load(self) -> None:
        """Optional hook called after the plugin is loaded.

        Override for initialization that needs to happen once at startup.
        """
        pass

    def on_unload(self) -> None:
        """Optional hook called when the plugin is being unloaded.

        Override for cleanup (close connections, flush state, etc.).
        """
        pass

    def get_info(self) -> dict[str, Any]:
        """Return plugin metadata as a dict."""
        return {
            "name": self.name,
            "type": self.plugin_type.value,
            "version": self.version,
            "description": self.description,
        }

    def __repr__(self) -> str:
        return f"<Plugin {self.name} ({self.plugin_type.value}) v{self.version}>"


class PluginRegistry:
    """Thread-safe registry of loaded plugins.

    The global instance is ``plugins.base.plugin_registry``.
    """

    def __init__(self):
        self._plugins: dict[str, PluginBase] = {}
        self._lock = threading.Lock()

    def register(self, plugin: PluginBase) -> None:
        """Register a plugin instance."""
        if not plugin.name:
            raise ValueError("Plugin must have a name")
        with self._lock:
            if plugin.name in self._plugins:
                log.warning("Plugin '%s' already registered, replacing", plugin.name)
            self._plugins[plugin.name] = plugin
        log.info("Plugin registered: %s", plugin)

    def unregister(self, name: str) -> PluginBase | None:
        """Remove a plugin from the registry."""
        with self._lock:
            plugin = self._plugins.pop(name, None)
        if plugin:
            try:
                plugin.on_unload()
            except Exception:
                log.warning("Plugin '%s' on_unload failed", name, exc_info=True)
            log.info("Plugin unregistered: %s", name)
        return plugin

    def get(self, name: str) -> PluginBase | None:
        """Get a plugin by name."""
        with self._lock:
            return self._plugins.get(name)

    def list_plugins(self) -> list[dict[str, Any]]:
        """Return info for all registered plugins."""
        with self._lock:
            plugins = sorted(self._plugins.values(), key=lambda p: p.name)
        return [p.get_info() for p in plugins]

    def list_by_type(self, plugin_type: PluginType) -> list[PluginBase]:
        """Return all plugins of a given type."""
        with self._lock:
            return [
                p for p in self._plugins.values()
                if p.plugin_type == plugin_type
            ]

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._plugins

    def __len__(self) -> int:
        with self._lock:
            return len(self._plugins)

    def __repr__(self) -> str:
        with self._lock:
            count = len(self._plugins)
        return f"<PluginRegistry [{count} plugins]>"


# Global plugin registry
plugin_registry = PluginRegistry()
