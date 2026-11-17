import json

from reid_pipeline.retrieval.osnet_retriever import OSNetRetriever, retrieve_top_k
from reid_pipeline.reranking.vlm_reranker import VLMReranker, parse_vlm_json, fuse_scores


def test_retrieve_top_k_returns_descending_candidates():
    result = retrieve_top_k([1.0, 0.0], {"a": [0.8, 0.6], "b": [0.0, 1.0], "c": [0.9, 0.1]}, k=2)
    assert [candidate.identity for candidate in result] == ["c", "a"]


def test_retriever_uses_configured_candidate_budget():
    result = OSNetRetriever(default_k=1).retrieve([1.0, 0.0], {"a": [1.0, 0.0], "b": [0.0, 1.0]})
    assert len(result) == 1


def test_parse_vlm_json_rejects_low_confidence_and_malformed_payloads():
    assert parse_vlm_json("not json") is None
    assert parse_vlm_json(json.dumps({"same_person": True, "confidence": 0.7})) is None


def test_parse_vlm_json_accepts_structured_judgment():
    result = parse_vlm_json(json.dumps({"same_person": True, "confidence": 0.9, "matched_attributes": ["shoes"]}))
    assert result.same_person is True
    assert result.confidence == 0.9


def test_fuse_scores_preserves_osnet_when_vlm_missing():
    assert fuse_scores(0.8, 0.5, None) == 0.675


def test_vlm_reranker_keeps_osnet_order_when_judgment_is_unavailable():
    class UnavailableVLM:
        def generate(self, images, prompt):
            return "invalid"

    reranker = VLMReranker(UnavailableVLM())
    result = reranker.rerank([], [("a", 0.9, 0.1), ("b", 0.8, 0.9)])
    assert [item[0] for item in result] == ["a", "b"]


def test_vlm_reranker_fuses_high_confidence_judgment():
    class AgreeingVLM:
        def generate(self, images, prompt):
            return '{"same_person": true, "confidence": 0.9}'

    result = VLMReranker(AgreeingVLM()).rerank([], [("a", 0.8, 0.5)])
    assert result[0][2] == 0.765


def test_vlm_reranker_sends_query_and_candidate_images():
    class ImageAwareVLM:
        def __init__(self):
            self.image_count = 0

        def generate(self, images, prompt):
            self.image_count = len(images)
            return '{"same_person": true, "confidence": 0.9}'

    client = ImageAwareVLM()
    result = VLMReranker(client).rerank([b"query"], [("a", 0.8, 0.5, [b"candidate"])])
    assert client.image_count == 2
    assert result[0][0] == "a"
