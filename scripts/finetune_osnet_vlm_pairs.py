"""Directly fine-tune OSNet on VLM-labeled image pairs."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch,torchreid
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

def main():
 p=argparse.ArgumentParser(); p.add_argument('labels',type=Path); p.add_argument('crops',type=Path); p.add_argument('checkpoint',type=Path); p.add_argument('output',type=Path); p.add_argument('--device',default='cuda'); a=p.parse_args()
 labels=[json.loads(x) for x in a.labels.read_text().splitlines() if x.strip()]
 index=[json.loads(x) for x in (a.crops/'index.jsonl').read_text().splitlines() if x.strip()]
 by={(x['pair_index'],x['kind']):a.crops/x['file'] for x in index}
 tf=transforms.Compose([transforms.Resize((256,128)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 pairs=[]
 for i,r in enumerate(labels):
  q,c=by.get((i,'q')),by.get((i,'c'))
  if q and c: pairs.append((q,c,float(r['vlm']['confidence'] if r['vlm']['same_person'] else 1-r['vlm']['confidence'])))
 device=torch.device(a.device); model=torchreid.models.build_model('osnet_x0_25',num_classes=1000,loss='triplet',pretrained=True); ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); sd={k.removeprefix('module.'):v for k,v in ck.get('state_dict',ck).items() if not k.removeprefix('module.').startswith('classifier.')}; model.load_state_dict(sd,strict=False); model.to(device).train(); opt=torch.optim.AdamW(model.parameters(),lr=1e-5)
 hist=[]
 for _ in range(5):
  losses=[]
  for qpath,cpath,teacher in pairs:
   q=tf(Image.open(qpath).convert('RGB')).unsqueeze(0).to(device); c=tf(Image.open(cpath).convert('RGB')).unsqueeze(0).to(device); _,features=model(torch.cat((q,c),dim=0)); qf,cf=features[:1],features[1:]; sim=(F.normalize(qf,dim=1)*F.normalize(cf,dim=1)).sum(); target=torch.tensor([teacher],device=device); loss=F.smooth_l1_loss((sim+1)/2,target); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
  hist.append(sum(losses)/max(1,len(losses)))
 a.output.parent.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':model.state_dict(),'loss':hist,'pairs':len(pairs)},a.output); print(json.dumps({'pairs':len(pairs),'loss':hist}))
if __name__=='__main__': main()
