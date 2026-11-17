from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalMetrics:
    rank1: float
    mAP: float


def evaluate_retrieval(query_ids, gallery_ids, scores) -> RetrievalMetrics:
    if not query_ids or not gallery_ids:
        return RetrievalMetrics(0.0, 0.0)
    rank1_hits = 0
    average_precisions = []
    for query_id, row in zip(query_ids, scores):
        if not row:
            average_precisions.append(0.0)
            continue
        order = sorted(range(len(row)), key=lambda index: row[index], reverse=True)
        relevant = [gallery_ids[index] == query_id for index in order]
        if relevant and relevant[0]:
            rank1_hits += 1
        total_relevant = sum(relevant)
        if not total_relevant:
            average_precisions.append(0.0)
            continue
        hits = 0
        precision_sum = 0.0
        for rank, is_relevant in enumerate(relevant, start=1):
            if is_relevant:
                hits += 1
                precision_sum += hits / rank
        average_precisions.append(precision_sum / total_relevant)
    count = len(query_ids)
    return RetrievalMetrics(rank1=rank1_hits / count, mAP=sum(average_precisions) / count)
