"""Extract unique query embeddings and GT identities from pair records."""
import argparse, json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    seen = set()
    rows = []
    for line in args.pairs.read_text().splitlines():
        record = json.loads(line)
        key = (record["gt_identity"], tuple(record["query_embedding"]))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"embedding": record["query_embedding"], "gt_identity": record["gt_identity"]})
    args.output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    print(len(rows))

if __name__ == "__main__":
    main()
