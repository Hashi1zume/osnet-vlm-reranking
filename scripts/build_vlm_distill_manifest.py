"""Build a soft VLM relation manifest from audited embeddings."""
import argparse, json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit = {(r["frame"], r["assigned_identity"]): r for r in (json.loads(x) for x in args.audit.read_text().splitlines() if x.strip())}
    rows=[]
    for line in args.labels.read_text().splitlines():
        r=json.loads(line); q=audit.get((r["frame"],r["query_assigned"])); c=audit.get((r["frame"],r["candidate_assigned"]))
        if not q or not c: continue
        v=r["vlm"]; p=float(v["confidence"] if v["same_person"] else 1.0-v["confidence"])
        rows.append({"query_embedding":q["embedding"],"candidate_embedding":c["embedding"],"teacher_same_probability":p,"confidence":v["confidence"]})
    args.output.write_text("\n".join(json.dumps(x) for x in rows)+("\n" if rows else "")); print(len(rows))
if __name__ == "__main__": main()
