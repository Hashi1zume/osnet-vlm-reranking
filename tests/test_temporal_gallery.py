import numpy as np

from reid_pipeline.temporal.gallery import TemporalIdentityGallery


def test_gallery_registers_and_matches_reappearing_identity():
    gallery = TemporalIdentityGallery(similarity_threshold=0.8)
    gallery.register("global-1", np.array([1.0, 0.0]), frame=1)
    identity, score = gallery.match(np.array([0.98, 0.02]), frame=20)
    assert identity == "global-1"
    assert score > 0.8


def test_gallery_rejects_dissimilar_identity():
    gallery = TemporalIdentityGallery(similarity_threshold=0.8)
    gallery.register("global-1", np.array([1.0, 0.0]), frame=1)
    identity, score = gallery.match(np.array([0.0, 1.0]), frame=20)
    assert identity is None
    assert score < 0.8


def test_gallery_updates_prototype_without_losing_history():
    gallery = TemporalIdentityGallery(similarity_threshold=0.5, max_samples=2)
    gallery.register("global-1", np.array([1.0, 0.0]), frame=1)
    gallery.update("global-1", np.array([0.8, 0.2]), frame=2)
    assert len(gallery.samples("global-1")) == 2
