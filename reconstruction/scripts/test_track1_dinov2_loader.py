"""Cached DINOv2 must load without GitHub discovery or silent fallback."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

source = Path(__file__).resolve().parents[1] / 'modules/v2d_cari4d/lib/cari4d/learning/models/dinov2_loader.py'
spec = importlib.util.spec_from_file_location('dinov2_loader', source)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


class Dinov2LoaderTests(unittest.TestCase):
    def test_cached_source_uses_local_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached = Path(tmp) / 'facebookresearch_dinov2_main'
            cached.mkdir(); (cached / 'hubconf.py').touch()
            hub = Mock(); hub.get_dir.return_value = tmp
            self.assertIs(loader.load_dinov2('dinov2_vitb14', hub=hub), hub.load.return_value)
            hub.load.assert_called_once_with(str(cached), 'dinov2_vitb14', source='local')

    def test_missing_cache_uses_explicit_main_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            hub = Mock(); hub.get_dir.return_value = tmp
            loader.load_dinov2('dinov2_vits14', hub=hub)
            hub.load.assert_called_once_with('facebookresearch/dinov2:main', 'dinov2_vits14')

    def test_broken_cached_source_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached = Path(tmp) / 'facebookresearch_dinov2_main'
            cached.mkdir(); (cached / 'hubconf.py').touch()
            hub = Mock(); hub.get_dir.return_value = tmp
            hub.load.side_effect = RuntimeError('invalid local model')
            with self.assertRaisesRegex(RuntimeError, 'invalid local model'):
                loader.load_dinov2('dinov2_vitb14', hub=hub)
            self.assertEqual(hub.load.call_count, 1)


if __name__ == '__main__':
    unittest.main()
