"""Export audited video crops for VLM-labeled pairs so OSNet can be fine-tuned."""
import argparse, json
from pathlib import Path
import cv2

def main():
    p=argparse.ArgumentParser(); p.add_argument('labels',type=Path); p.add_argument('audit',type=Path); p.add_argument('video',type=Path); p.add_argument('output',type=Path); a=p.parse_args()
    labels=[json.loads(x) for x in a.labels.read_text().splitlines() if x.strip()]
    audit={(r['frame'],r['assigned_identity']):r for r in (json.loads(x) for x in a.audit.read_text().splitlines() if x.strip())}
    needed={(r['frame'],r['query_assigned'],'q') for r in labels}|{(r['frame'],r['candidate_assigned'],'c') for r in labels}
    by_frame={}
    for frame,ident,kind in needed: by_frame.setdefault(frame,[]).append((ident,kind))
    cap=cv2.VideoCapture(str(a.video)); frame=0; a.output.mkdir(parents=True,exist_ok=True); rows=[]
    pair_files = {}
    while True:
        ok,img=cap.read()
        if not ok: break
        frame+=1
        for ident,kind in by_frame.get(frame,[]):
            r=audit.get((frame,ident))
            if not r: continue
            x1,y1,x2,y2=map(int,r['bbox']); crop=img[max(0,y1):max(y1,y2),max(0,x1):max(x1,x2)]
            if crop.size==0: continue
            pair_index = next((i for i, item in enumerate(labels) if item['frame'] == frame and item.get('query_assigned' if kind == 'q' else 'candidate_assigned') == ident), len(rows))
            name=f"{len(rows):06d}_{kind}.jpg"; cv2.imwrite(str(a.output/name),crop); rows.append({'file':name,'frame':frame,'identity':ident,'kind':kind,'pair_index':pair_index})
    cap.release(); (a.output/'index.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n'); print(len(rows))
if __name__=='__main__': main()
