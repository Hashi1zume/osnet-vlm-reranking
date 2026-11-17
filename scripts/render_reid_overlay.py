"""Render sampled video frames with the global Re-ID labels from a JSONL audit."""
import argparse, json, hashlib
from collections import defaultdict
from pathlib import Path
import cv2

def color(identity, video_scope):
    # Scope the palette by video so identical local ID numbers never imply cross-video identity.
    digest=hashlib.md5(f'{video_scope}:{identity}'.encode()).digest()
    return (70+digest[0]%170,70+digest[1]%170,70+digest[2]%170)

def main():
    p=argparse.ArgumentParser(); p.add_argument('video',type=Path); p.add_argument('audit',type=Path); p.add_argument('output',type=Path); p.add_argument('--frame-skip',type=int,default=1); p.add_argument('--id-scope',default=None); a=p.parse_args()
    video_scope=a.id_scope or a.video.stem
    by_frame=defaultdict(list)
    for line in a.audit.read_text().splitlines():
        r=json.loads(line); by_frame[int(r['frame'])].append(r)
    cap=cv2.VideoCapture(str(a.video)); fps=cap.get(cv2.CAP_PROP_FPS) or 25; w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    a.output.parent.mkdir(parents=True,exist_ok=True); out=cv2.VideoWriter(str(a.output),cv2.VideoWriter_fourcc(*'mp4v'),fps/a.frame_skip,(w,h)); idx=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        idx+=1
        # Match the runner's zero-based sampling phase so each audit row overlays its source frame.
        if (idx-1)%a.frame_skip: continue
        for r in by_frame.get(idx,[]):
            x1,y1,x2,y2=map(int,r['bbox']); ident=r.get('assigned_identity',f"track-{r['track_id']}"); col=color(ident,video_scope)
            cv2.rectangle(frame,(x1,y1),(x2,y2),col,3)
            label=f"{video_scope} · {ident}"
            (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.8,2); y=max(th+8,y1)
            cv2.rectangle(frame,(x1,y-th-8),(x1+tw+10,y),col,-1); cv2.putText(frame,label,(x1+5,y-5),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(frame,f"Frame {idx}  |  OSNet + Qwen2.5-VL identity gate",(20,38),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2,cv2.LINE_AA); out.write(frame)
    cap.release(); out.release(); print(a.output)
if __name__=='__main__': main()
