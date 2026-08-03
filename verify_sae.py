import numpy as np, torch, vit_sae_conic as V
tok,Y,P = V.extract(2500)
X=tok.reshape(-1,tok.shape[-1]); Xc=X-X.mean(0,keepdims=True)
Xg=torch.tensor(Xc,device=V.DEVICE)
tu=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-8)
pi=np.repeat(np.arange(len(tok)),P)
rng=np.random.default_rng(0); Xtr=torch.tensor(Xc[rng.choice(len(Xc),150000,replace=False)],device=V.DEVICE)
for nonneg,name in [(True,'conic'),(False,'signed')]:
  coh,pur,mp=[],[],[]
  for s in range(3):
    m=V.train_sae(Xtr,1024,nonneg,16,seed=s); C=V.sae_codes(m,Xg,16)
    coh.append(V.coherence(C,tu)); pur.append(V.purity(C,pi,Y))
    mp.append(V.editability(C.reshape(len(tok),P,-1).max(1),Y)['probe_mAP'])
    print(f'  {name} seed{s}: coher {coh[-1]:.3f} purity {pur[-1]:.3f} mAP {mp[-1]:.3f}',flush=True)
  print(f'== {name}: coher {np.mean(coh):.3f}±{np.std(coh):.3f} purity {np.mean(pur):.3f}±{np.std(pur):.3f} mAP {np.mean(mp):.3f}±{np.std(mp):.3f}',flush=True)
