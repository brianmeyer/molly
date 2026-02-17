"""Molly plugin architecture (Phase 5C.4).

Lightweight discovery-based plugin system.  Two types:
  - Channel plugins: extend ``channels.base.Channel``
  - Tool plugins: provide new MCP tool definitions

Discovery: scan ``config.PLUGIN_DIR`` for Python modules with a
``PLUGIN_CLASS`` attribute.  Each plugin implements ``register(registry)``.
Restart to load.
"""
from plugins.base import PluginBase, PluginRegistry, PluginType
from plugins.loader import load_plugins, discover_plugins

__all__ = [
    "PluginBase",
    "PluginRegistry",
    "PluginType",
    "load_plugins",
    "discover_plugins",
]
