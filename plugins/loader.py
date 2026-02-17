"""Plugin discovery and loading (Phase 5C.4).

Scans ``config.PLUGIN_DIR`` for Python modules that export a
``PLUGIN_CLASS`` attribute.  Each module is imported and its
``PLUGIN_CLASS`` is instantiated and registered.

Usage::

    from plugins.loader import load_plugins
    loaded = load_plugins()  # returns list of loaded plugin names

Discovery rules:
  1. Only ``.py`` files in the plugin directory (non-recursive)
  2. Files starting with ``_`` are skipped
  3. Module must have a top-level ``PLUGIN_CLASS`` attribute
  4. ``PLUGIN_CLASS`` must be a subclass of ``PluginBase``
  5. Plugin must have a non-empty ``name`` attribute
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

import config
from plugins.base import PluginBase, PluginRegistry, plugin_registry

log = logging.getLogger(__name__)


def discover_plugins(plugin_dir: Path | None = None) -> list[dict[str, Any]]:
    """Discover available plugins without loading them.

    Returns metadata dicts for each discovered plugin module.
    """
    target_dir = plugin_dir or config.PLUGIN_DIR
    if not target_dir.exists() or not target_dir.is_dir():
        return []

    discovered: list[dict[str, Any]] = []
    for py_file in sorted(target_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        module_name = py_file.stem
        discovered.append({
            "module": module_name,
            "path": str(py_file),
        })

    return discovered


def load_plugins(
    plugin_dir: Path | None = None,
    registry: PluginRegistry | None = None,
) -> list[str]:
    """Load and register all plugins from the plugin directory.

    Parameters
    ----------
    plugin_dir : Path | None
        Directory to scan.  Defaults to ``config.PLUGIN_DIR``.
    registry : PluginRegistry | None
        Registry to register plugins in.  Defaults to the global
        ``plugin_registry``.

    Returns
    -------
    list[str]
        Names of successfully loaded plugins.
    """
    if not config.PLUGIN_ENABLED:
        log.debug("Plugin system disabled (MOLLY_PLUGIN_ENABLED=False)")
        return []

    target_dir = plugin_dir or config.PLUGIN_DIR
    target_registry = registry if registry is not None else plugin_registry

    if not target_dir.exists():
        log.debug("Plugin directory does not exist: %s", target_dir)
        return []

    if not target_dir.is_dir():
        log.warning("Plugin path is not a directory: %s", target_dir)
        return []

    loaded_names: list[str] = []
    errors: list[str] = []

    for py_file in sorted(target_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        module_name = f"molly_plugin_{py_file.stem}"

        try:
            plugin_instance = _load_plugin_module(py_file, module_name)
            if plugin_instance is None:
                continue

            # Call on_load hook BEFORE registering (fail-safe: don't
            # register broken plugins)
            try:
                plugin_instance.on_load()
            except Exception:
                log.warning(
                    "Plugin '%s' on_load() failed — skipping registration",
                    plugin_instance.name,
                    exc_info=True,
                )
                # Clean up sys.modules to avoid poisoning future imports
                sys.modules.pop(module_name, None)
                continue

            # Register the plugin (only after on_load succeeds)
            target_registry.register(plugin_instance)

            # Let the plugin register itself with Molly's subsystems
            try:
                plugin_instance.register(target_registry)
            except Exception:
                log.warning(
                    "Plugin '%s' register() failed — unregistering",
                    plugin_instance.name,
                    exc_info=True,
                )
                target_registry.unregister(plugin_instance.name)
                sys.modules.pop(module_name, None)
                continue

            loaded_names.append(plugin_instance.name)
            log.info("Plugin loaded: %s from %s", plugin_instance.name, py_file.name)

        except Exception as exc:
            errors.append(f"{py_file.name}: {exc}")
            log.error("Failed to load plugin from %s", py_file, exc_info=True)

    if loaded_names:
        log.info("Loaded %d plugin(s): %s", len(loaded_names), ", ".join(loaded_names))
    if errors:
        log.warning("Failed to load %d plugin(s): %s", len(errors), "; ".join(errors))

    return loaded_names


def _load_plugin_module(py_file: Path, module_name: str) -> PluginBase | None:
    """Import a single plugin module and return its PLUGIN_CLASS instance.

    Returns None if the module doesn't define PLUGIN_CLASS or if
    PLUGIN_CLASS is not a valid PluginBase subclass.
    """
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, str(py_file))
    if spec is None or spec.loader is None:
        log.debug("Skipping %s: invalid module spec", py_file)
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        # Clean up on failure
        sys.modules.pop(module_name, None)
        raise

    # Check for PLUGIN_CLASS attribute
    plugin_class = getattr(module, "PLUGIN_CLASS", None)
    if plugin_class is None:
        log.debug("Skipping %s: no PLUGIN_CLASS attribute", py_file.name)
        sys.modules.pop(module_name, None)
        return None

    # Validate it's a PluginBase subclass
    if isinstance(plugin_class, type):
        if not issubclass(plugin_class, PluginBase):
            log.warning(
                "Skipping %s: PLUGIN_CLASS %s is not a PluginBase subclass",
                py_file.name,
                plugin_class.__name__,
            )
            sys.modules.pop(module_name, None)
            return None
        # Instantiate the plugin class
        instance = plugin_class()
    elif isinstance(plugin_class, PluginBase):
        # Already an instance
        instance = plugin_class
    else:
        log.warning(
            "Skipping %s: PLUGIN_CLASS is neither a class nor an instance (%s)",
            py_file.name,
            type(plugin_class).__name__,
        )
        sys.modules.pop(module_name, None)
        return None

    # Validate name
    if not instance.name:
        log.warning("Skipping %s: plugin has no name", py_file.name)
        sys.modules.pop(module_name, None)
        return None

    return instance


def unload_all(registry: PluginRegistry | None = None) -> int:
    """Unload all plugins from the registry.

    Returns the number of plugins unloaded.
    """
    target_registry = registry or plugin_registry
    names = [info["name"] for info in target_registry.list_plugins()]
    for name in names:
        target_registry.unregister(name)
    return len(names)
