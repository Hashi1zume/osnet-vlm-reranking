def collect_hard_cases(retrievals, judgments, final_sequences: set[str], minimum_confidence: float = 0.8):
    cases = []
    for retrieval, judgment in zip(retrievals, judgments):
        sequence = retrieval["sequence"]
        if sequence in final_sequences:
            raise ValueError(f"final sequence cannot produce adaptation cases: {sequence}")
        if float(judgment.get("confidence", 0.0)) < minimum_confidence:
            continue
        if retrieval.get("same_identity") is False and judgment.get("same_person") is False:
            cases.append({**retrieval, "judgment": judgment, "kind": "hard_negative"})
        elif retrieval.get("same_identity") is True and judgment.get("same_person") is True:
            cases.append({**retrieval, "judgment": judgment, "kind": "reliable_positive"})
    return cases
