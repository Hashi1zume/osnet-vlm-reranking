"""Train a conservative low-rank student using soft VLM pair relations."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from .adapter import LowRankAdapter
from .vlm_distillation import pair_relation_distillation_loss

def main():
    p=argparse.ArgumentParser(); p.add_argument('manifest',type=Path); p.add_argument('output',type=Path); p.add_argument('--device',default='cpu'); a=p.parse_args()
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x.strip()]
    q=torch.tensor([x['query_embedding'] for x in rows],dtype=torch.float32); c=torch.tensor([x['candidate_embedding'] for x in rows],dtype=torch.float32); y=torch.tensor([x['teacher_same_probability'] for x in rows],dtype=torch.float32)
    device=torch.device(a.device); q,c,y=q.to(device),c.to(device),y.to(device); model=LowRankAdapter(512,16,16).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-4); history=[]
    for _ in range(30):
        loss=pair_relation_distillation_loss(model(F.normalize(q,dim=1)),model(F.normalize(c,dim=1)),y)+0.1*F.mse_loss(model(q),q)+0.1*F.mse_loss(model(c),c)
        opt.zero_grad(); loss.backward(); opt.step(); history.append(float(loss.detach().cpu()))
    a.output.parent.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':model.state_dict(),'config':{'dim':512,'rank':16,'alpha':16},'loss':history},a.output); print(json.dumps({'pairs':len(rows),'final_loss':history[-1]}))
if __name__ == '__main__': main()
