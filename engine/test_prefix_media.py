import tempfile
import unittest
import uuid
from types import SimpleNamespace

import torch

from engine.prefix_disk import PrefixDisk
from engine.prefix_media import image_fingerprints, media_prefix
from engine.test_prefix_cache import _fixture
from engine.test_prefix_disk import payload
from engine import test_prefix_disk


def image(start=2, pixel=1):
    return SimpleNamespace(start=start, n_vit_h=1, n_vit_w=1,
                           types=torch.tensor([0, 1]), patches=torch.full((1, 3, 2, 2), float(pixel)))


class ImagePrefixTests(unittest.TestCase):
    def test_pixels_layout_and_type_validation(self):
        a = image_fingerprints([image()])
        self.assertEqual(a, image_fingerprints([image()]))
        self.assertNotEqual(a, image_fingerprints([image(pixel=2)]))
        im = image()
        im.n_vit_w = 2
        self.assertNotEqual(a, image_fingerprints([im]))
        self.assertEqual(a, image_fingerprints([image()], torch.tensor([-1, -1, 0, 1, -1])))
        with self.assertRaises(ValueError):
            image_fingerprints([image()], torch.tensor([-1, -1, -1, -1]))
        with self.assertRaises(ValueError):
            image_fingerprints([image(), image(start=3)])
        self.assertEqual(media_prefix(a, 2), ())
        self.assertIsNone(media_prefix(a, 3))
        self.assertEqual(media_prefix(a, 4), a)

    def test_memory_reuse_changed_image_and_text_only_transition(self):
        e = _fixture()
        ids = [1, 2, 3, 4, 5, 6]
        e._prefix_media = image_fingerprints([image()])
        e._prefix_snapshots[2] = e._snapshot_prefix(ids[:2], 2)
        self.assertIsNone(e._snapshot_prefix(ids[:3], 3))
        e._save_prefix(ids, 6)
        self.assertEqual(e._restore_prefix(ids + [7]), 6)
        e._prefix_media = image_fingerprints([image(pixel=2)])
        self.assertEqual(e._restore_prefix(ids + [7]), 2)
        e._save_prefix(ids, 6)
        e._prefix_media = ()
        self.assertEqual(e._restore_prefix(ids + [7]), 2)

    def test_new_image_after_cached_prefix_does_not_invalidate_it(self):
        e = _fixture()
        ids = [1, 2, 3, 4, 5, 6]
        e._prefix_media = image_fingerprints([image()])
        e._save_prefix(ids, 6)
        e._prefix_media = image_fingerprints([image(), image(start=6, pixel=2)])
        self.assertEqual(e._restore_prefix(ids + [7, 8]), 6)

    def test_disk_media_identity_restart_and_partial_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            disk = PrefixDisk(root, 'test', 1000000)
            p = payload()
            media = image_fingerprints([image()])
            p['media'] = media
            for n, snap in p['snapshots'].items():
                snap['media'] = media_prefix(media, n)
            bid = uuid.uuid4().hex
            disk.save(bid, p)
            disk = PrefixDisk(root, 'test', 1000000)
            ids = list(p['ids']) + [7]
            self.assertEqual(disk.candidates(ids, media=media), [(bid, 6), (bid, 4), (bid, 2)])
            self.assertIsNotNone(disk.load((bid, 6), ids, media))
            changed = image_fingerprints([image(pixel=2)])
            self.assertEqual(disk.candidates(ids, media=changed), [(bid, 2)])
            self.assertIsNone(disk.load((bid, 6), ids, changed))
            self.assertEqual(disk.candidates(ids), [(bid, 2)])
            self.assertIsNone(disk.load((bid, 6), ids))

    def test_adapter_rejects_changed_media_before_gpu_restore(self):
        fixture = test_prefix_disk.PrefixDiskTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        p = payload()
        media = image_fingerprints([image()])
        p['media'] = media
        for n, snap in p['snapshots'].items():
            snap['media'] = media_prefix(media, n)
        fixture.save(p)
        adapter = fixture.adapter()
        adapter.engine._prefix_media = media
        self.assertEqual(adapter.restore(list(p['ids']), 0), 6)
        adapter.engine._prefix_media = image_fingerprints([image(pixel=2)])
        self.assertEqual(adapter.restore(list(p['ids']), 0), 2)


if __name__ == '__main__':
    unittest.main()
