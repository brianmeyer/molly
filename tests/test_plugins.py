"""Tests for Phase 5C.4 Plugin Architecture.

Tests cover:
  - PluginBase ABC
  - PluginRegistry operations
  - PluginType enum
  - Plugin discovery
  - Plugin loading
  - Config values
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import config
from plugins.base import PluginBase, PluginRegistry, PluginType


class TestPluginType(unittest.TestCase):
    """Test plugin type enum."""

    def test_channel_type(self):
        self.assertEqual(PluginType.CHANNEL, "channel")

    def test_tool_type(self):
        self.assertEqual(PluginType.TOOL, "tool")


class TestPluginBase(unittest.TestCase):
    """Test PluginBase abstract class."""

    def test_must_implement_register(self):
        with self.assertRaises(TypeError):
            PluginBase()  # type: ignore[abstract]

    def test_concrete_plugin(self):
        class TestPlugin(PluginBase):
            name = "test"
            plugin_type = PluginType.TOOL
            version = "1.0.0"
            description = "Test plugin"

            def register(self, registry):
                pass

        plugin = TestPlugin()
        self.assertEqual(plugin.name, "test")
        self.assertEqual(plugin.plugin_type, PluginType.TOOL)
        self.assertEqual(plugin.version, "1.0.0")

    def test_get_info(self):
        class TestPlugin(PluginBase):
            name = "info-test"
            plugin_type = PluginType.CHANNEL
            version = "2.0.0"
            description = "Info test"

            def register(self, registry):
                pass

        plugin = TestPlugin()
        info = plugin.get_info()
        self.assertEqual(info["name"], "info-test")
        self.assertEqual(info["type"], "channel")
        self.assertEqual(info["version"], "2.0.0")
        self.assertEqual(info["description"], "Info test")

    def test_repr(self):
        class TestPlugin(PluginBase):
            name = "repr-test"
            plugin_type = PluginType.TOOL
            version = "0.1.0"

            def register(self, registry):
                pass

        plugin = TestPlugin()
        self.assertIn("repr-test", repr(plugin))
        self.assertIn("tool", repr(plugin))


class TestPluginRegistry(unittest.TestCase):
    """Test plugin registry operations."""

    def setUp(self):
        self.registry = PluginRegistry()

    def _make_plugin(self, name: str, ptype: PluginType = PluginType.TOOL):
        class DynPlugin(PluginBase):
            plugin_type = ptype
            version = "1.0.0"

            def register(self, registry):
                pass

        plugin = DynPlugin()
        plugin.name = name
        return plugin

    def test_register_and_get(self):
        plugin = self._make_plugin("test1")
        self.registry.register(plugin)
        self.assertIn("test1", self.registry)
        self.assertEqual(self.registry.get("test1"), plugin)

    def test_get_unknown(self):
        self.assertIsNone(self.registry.get("nonexistent"))

    def test_len(self):
        self.assertEqual(len(self.registry), 0)
        self.registry.register(self._make_plugin("p1"))
        self.assertEqual(len(self.registry), 1)
        self.registry.register(self._make_plugin("p2"))
        self.assertEqual(len(self.registry), 2)

    def test_unregister(self):
        plugin = self._make_plugin("removeme")
        self.registry.register(plugin)
        self.assertIn("removeme", self.registry)
        removed = self.registry.unregister("removeme")
        self.assertEqual(removed, plugin)
        self.assertNotIn("removeme", self.registry)

    def test_unregister_nonexistent(self):
        result = self.registry.unregister("ghost")
        self.assertIsNone(result)

    def test_list_plugins(self):
        self.registry.register(self._make_plugin("beta"))
        self.registry.register(self._make_plugin("alpha"))
        plugins = self.registry.list_plugins()
        self.assertEqual(len(plugins), 2)
        names = [p["name"] for p in plugins]
        self.assertEqual(names, ["alpha", "beta"])  # sorted

    def test_list_by_type(self):
        self.registry.register(self._make_plugin("ch1", PluginType.CHANNEL))
        self.registry.register(self._make_plugin("tool1", PluginType.TOOL))
        self.registry.register(self._make_plugin("ch2", PluginType.CHANNEL))

        channels = self.registry.list_by_type(PluginType.CHANNEL)
        tools = self.registry.list_by_type(PluginType.TOOL)
        self.assertEqual(len(channels), 2)
        self.assertEqual(len(tools), 1)

    def test_register_no_name(self):
        class NoNamePlugin(PluginBase):
            name = ""
            def register(self, registry):
                pass

        with self.assertRaises(ValueError):
            self.registry.register(NoNamePlugin())

    def test_repr(self):
        self.assertIn("0 plugins", repr(self.registry))
        self.registry.register(self._make_plugin("x"))
        self.assertIn("1 plugins", repr(self.registry))


class TestPluginDiscovery(unittest.TestCase):
    """Test plugin discovery."""

    def test_discover_empty_dir(self):
        from plugins.loader import discover_plugins
        with tempfile.TemporaryDirectory() as tmpdir:
            result = discover_plugins(Path(tmpdir))
            self.assertEqual(result, [])

    def test_discover_with_plugins(self):
        from plugins.loader import discover_plugins
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "my_plugin.py").write_text("PLUGIN_CLASS = None")
            (Path(tmpdir) / "_private.py").write_text("PLUGIN_CLASS = None")
            result = discover_plugins(Path(tmpdir))
            # _private.py should be skipped
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["module"], "my_plugin")

    def test_discover_nonexistent_dir(self):
        from plugins.loader import discover_plugins
        result = discover_plugins(Path("/nonexistent/path"))
        self.assertEqual(result, [])


class TestPluginLoading(unittest.TestCase):
    """Test plugin loading from files."""

    def test_load_valid_plugin(self):
        from plugins.loader import load_plugins
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_code = '''
from plugins.base import PluginBase, PluginType

class MyPlugin(PluginBase):
    name = "test-loader"
    plugin_type = PluginType.TOOL
    version = "1.0.0"
    description = "Test plugin for loading"

    def register(self, registry):
        pass

PLUGIN_CLASS = MyPlugin
'''
            (Path(tmpdir) / "test_plugin.py").write_text(plugin_code)
            registry = PluginRegistry()
            loaded = load_plugins(Path(tmpdir), registry)
            self.assertEqual(loaded, ["test-loader"])
            self.assertIn("test-loader", registry)

    def test_load_no_plugin_class(self):
        from plugins.loader import load_plugins
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "no_plugin.py").write_text("x = 42")
            registry = PluginRegistry()
            loaded = load_plugins(Path(tmpdir), registry)
            self.assertEqual(loaded, [])

    def test_load_disabled(self):
        from plugins.loader import load_plugins
        saved = config.PLUGIN_ENABLED
        config.PLUGIN_ENABLED = False
        try:
            loaded = load_plugins()
            self.assertEqual(loaded, [])
        finally:
            config.PLUGIN_ENABLED = saved

    def test_unload_all(self):
        from plugins.loader import unload_all
        registry = PluginRegistry()

        class Dummy(PluginBase):
            name = "dummy"
            def register(self, r): pass

        registry.register(Dummy())
        self.assertEqual(len(registry), 1)
        count = unload_all(registry)
        self.assertEqual(count, 1)
        self.assertEqual(len(registry), 0)


class TestPluginConfig(unittest.TestCase):
    """Test plugin config values."""

    def test_plugin_enabled_default(self):
        self.assertTrue(config.PLUGIN_ENABLED)

    def test_plugin_dir(self):
        self.assertIsInstance(config.PLUGIN_DIR, Path)


if __name__ == "__main__":
    unittest.main()
