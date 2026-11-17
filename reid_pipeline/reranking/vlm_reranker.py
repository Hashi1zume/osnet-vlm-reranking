from dataclasses import dataclass
import json


@dataclass(frozen=True)
class VLMJudgment:
    same_person: bool
    confidence: float
    matched_attributes: tuple[str, ...] = ()
    conflicting_attributes: tuple[str, ...] = ()
    unknown_attributes: tuple[str, ...] = ()


def parse_vlm_json(raw: str, minimum_confidence: float = 0.8):
    try:
        payload = json.loads(raw)
        confidence = float(payload["confidence"])
        same_person = payload["same_person"]
        if not isinstance(same_person, bool) or confidence < minimum_confidence:
            return None
        return VLMJudgment(
            same_person=same_person,
            confidence=confidence,
            matched_attributes=tuple(payload.get("matched_attributes", [])),
            conflicting_attributes=tuple(payload.get("conflicting_attributes", [])),
            unknown_attributes=tuple(payload.get("unknown_attributes", [])),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None


def fuse_scores(osnet_score: float, attribute_score: float, judgment: VLMJudgment | None) -> float:
    vlm_score = judgment.confidence if judgment is not None and judgment.same_person else 0.0
    return round(0.75 * osnet_score + 0.15 * attribute_score + 0.10 * vlm_score, 10)


class VLMReranker:
    """Applies VLM evidence only after OSNet has reduced the search space."""

    def __init__(self, vlm_client, minimum_confidence: float = 0.8):
        self.vlm_client = vlm_client
        self.minimum_confidence = minimum_confidence
        self.last_judgments = {}

    def rerank(self, query_images: list[bytes], candidates):
        reranked = []
        for candidate in candidates:
            identity, osnet_score, attribute_score = candidate[:3]
            candidate_images = candidate[3] if len(candidate) > 3 else []
            prompt = (
                "Compare the query images with the candidate person. "
                "Return JSON with same_person, confidence, matched_attributes, "
                "conflicting_attributes, and unknown_attributes."
            )
            raw = self.vlm_client.generate(query_images + candidate_images, prompt)
            judgment = parse_vlm_json(raw, self.minimum_confidence)
            self.last_judgments[identity] = judgment
            # A failed VLM call must degrade to the proven retrieval path, not invent evidence.
            final_score = osnet_score if judgment is None else fuse_scores(osnet_score, attribute_score, judgment)
            reranked.append((identity, osnet_score, final_score))
        return sorted(reranked, key=lambda item: item[2], reverse=True)
