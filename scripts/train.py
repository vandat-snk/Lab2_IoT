"""Fine-tune a lightweight convolutional MobileNetV3 on CUDA, report held-out metrics."""
import argparse
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache/torch"))
LABELS = ["with_mask", "without_mask", "mask_weared_incorrect"]
NAMES = ["Đeo đúng", "Không đeo", "Đeo sai"]
NORMALIZE = transforms.Normalize([.485,.456,.406],[.229,.224,.225])


class Faces(Dataset):
    def __init__(self, rows, augment=False):
        self.rows = rows
        # Small crops fit in RAM; decode once instead of reopening thousands of
        # JPEGs through Windows antivirus on every epoch.
        self.images = []
        for row in rows:
            with Image.open(ROOT / row['path']) as im:
                self.images.append(im.convert('RGB').copy())
        ops = [transforms.Resize((160,160))]
        if augment:
            ops += [transforms.RandomHorizontalFlip(),transforms.RandomAffine(10,translate=(.04,.04),scale=(.9,1.1)),
                    transforms.ColorJitter(.2,.2,.15,.03)]
        self.transform = transforms.Compose(ops + [transforms.ToTensor(),NORMALIZE])
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows[idx]
        image = self.transform(self.images[idx])
        return image, LABELS.index(row["label"])


def make_model(pretrained=False):
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 3)
    return model


def evaluate(model, loader, device):
    model.eval()
    truth, pred, loss_sum = [], [], 0.
    with torch.inference_mode():
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            logits = model(x)
            loss_sum += nn.functional.cross_entropy(logits,y,reduction="sum").item()
            truth.extend(y.cpu().tolist()); pred.extend(logits.argmax(1).cpu().tolist())
    report = classification_report(truth,pred,labels=[0,1,2],target_names=LABELS,output_dict=True,zero_division=0)
    return loss_sum/len(truth),report,confusion_matrix(truth,pred,labels=[0,1,2]).tolist(),truth,pred


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=25)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--device',default='cuda')
    p.add_argument('--workers',type=int,default=0)
    args=p.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install the CUDA PyTorch build or explicitly pass --device cpu.')
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device=torch.device(args.device)
    rows=json.loads((ROOT/'data/processed/manifest.json').read_text())
    parts={s:[r for r in rows if r['split']==s] for s in ['train','val','test']}
    trainset=Faces(parts['train'],True)
    counts=np.bincount([LABELS.index(r['label']) for r in parts['train']],minlength=3)
    # Sampling balances the rare incorrect class without double weighting its loss.
    sampler=WeightedRandomSampler([1/counts[LABELS.index(r['label'])] for r in parts['train']],len(trainset),replacement=True,
                                  generator=torch.Generator().manual_seed(42))
    loaders={s:DataLoader(Faces(parts[s]),batch_size=args.batch_size,num_workers=args.workers) for s in ['val','test']}
    trainloader=DataLoader(trainset,batch_size=args.batch_size,sampler=sampler,num_workers=args.workers,pin_memory=device.type=='cuda')
    model=make_model(True).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-5)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    history=[]; best=-1; best_epoch=0; start=time.perf_counter()
    gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'
    print(f'Training on {gpu}, {sum(p.numel() for p in model.parameters()):,} parameters',flush=True)
    for epoch in range(1,args.epochs+1):
        t=time.perf_counter();model.train();total_loss=0;correct=0;seen=0
        for x,y in trainloader:
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                logits=model(x);loss=nn.functional.cross_entropy(logits,y,label_smoothing=.05)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            total_loss+=loss.item()*len(y);correct+=(logits.argmax(1)==y).sum().item();seen+=len(y)
        scheduler.step()
        vl,report,_,_,_=evaluate(model,loaders['val'],device)
        score=report['macro avg']['f1-score']
        row={'epoch':epoch,'train_loss':total_loss/seen,'train_accuracy_balanced_sampling':correct/seen,'val_loss':vl,
             'val_accuracy':report['accuracy'],'val_macro_f1':score,'epoch_seconds':time.perf_counter()-t}
        history.append(row)
        if score>best:
            best=score;best_epoch=epoch
            cpu_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            torch.save({'state_dict':cpu_state,'epoch':epoch,'labels':LABELS},ROOT/'artifacts/checkpoint.pt')
        (ROOT/'artifacts/history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(row),flush=True)
    checkpoint=torch.load(ROOT/'artifacts/checkpoint.pt',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    tl,test,cm,truth,pred=evaluate(model,loaders['test'],device)
    result={'architecture':'MobileNetV3-Small CNN, ImageNet transfer learning, 3-class head','device':gpu,
            'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'epochs':args.epochs,'best_epoch':best_epoch,
            'batch_size':args.batch_size,'seed':42,'selection':'validation macro F1 only; test evaluated once after selection',
            'parameter_count':sum(p.numel() for p in model.parameters()),'training_seconds':time.perf_counter()-start,
            'test_loss':tl,'classification_report':test,'confusion_matrix':cm,'class_order':LABELS,
            'caveat':'Face-crop classifier evaluation. Full camera pipeline additionally depends on face detection. Public dataset contains class imbalance and domain shift.'}
    (ROOT/'artifacts/evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (ROOT/'artifacts/test_predictions.json').write_text(json.dumps([{'path':r['path'],'truth':a,'predicted':b} for r,a,b in zip(parts['test'],truth,pred)],indent=2))
    print(json.dumps(result,indent=2),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    axs[0].plot([r['epoch'] for r in history],[r['train_loss'] for r in history],label='Train (balanced sampling)')
    axs[0].plot([r['epoch'] for r in history],[r['val_loss'] for r in history],label='Validation')
    axs[0].set(xlabel='Epoch',ylabel='Cross entropy',title='Learning curves');axs[0].legend()
    axs[1].imshow(cm,cmap='Blues');axs[1].set(xticks=range(3),yticks=range(3),xticklabels=['Correct','No mask','Incorrect'],yticklabels=['Correct','No mask','Incorrect'],xlabel='Predicted',ylabel='Ground truth',title='Held-out test confusion matrix')
    for i in range(3):
        for j in range(3):axs[1].text(j,i,str(cm[i][j]),ha='center',va='center',color='white' if cm[i][j]>max(map(max,cm))/2 else 'black')
    fig.tight_layout();fig.savefig(ROOT/'artifacts/training_results.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    main()
