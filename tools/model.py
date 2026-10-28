"""Pinned Anomalib adapter: overrides data routing, never network/loss."""
import torch
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule
from anomalib.data import ImageBatch, ImageItem
from anomalib.models import EfficientAd
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModelSize
from common import ROOT, preprocess, verified_records

class Images(Dataset):
    def __init__(self,split): self.rows=verified_records(split)
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        return ImageItem(image=torch.from_numpy(preprocess(ROOT/row['path'])[0]),gt_label=torch.tensor(row['label']),image_path=str(ROOT/row['path']))

class ExplicitData(LightningDataModule):
    train_batch_size=1
    def __init__(self):
        super().__init__()
        self.train=Images('train_normal');self.stats=Images('stats_normal')
    def loader(self,data,shuffle=False):
        return DataLoader(data,batch_size=1,shuffle=shuffle,num_workers=0,collate_fn=ImageBatch.collate)
    def train_dataloader(self): return self.loader(self.train,True)
    def stats_dataloader(self): return self.loader(self.stats)
    def val_dataloader(self): return self.stats_dataloader()

class ProtocolEfficientAd(EfficientAd):
    def on_train_start(self):
        if self.trainer.datamodule.train_batch_size!=1: raise ValueError('EfficientAD batch must be one')
        teacher=ROOT/'pre_trained/efficientad_pretrained_weights'/f'pretrained_teacher_{self.model_size.value}.pth'
        if not teacher.is_file() or not self.imagenet_dir.is_dir():
            raise FileNotFoundError('Missing local teacher/Imagenette; automatic downloads disabled by user')
        self.prepare_pretrained_model()
        self.prepare_imagenette_data((256,256))
        if not self.model.is_set(self.model.mean_std):
            self.model.mean_std.update(self.teacher_channel_mean_std(self.trainer.datamodule.stats_dataloader()))
    # Official on_validation_start reads val_dataloader, explicitly stats_normal above.

def create(c):
    return ProtocolEfficientAd(model_size=EfficientAdModelSize.S,lr=c['training']['learning_rate'],
        imagenet_dir=str(ROOT/c['training']['imagenette']),pre_processor=False,post_processor=False,evaluator=False,visualizer=False)

def restore(c,path):
    model=create(c)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.eval()
    if not model.model.is_set(model.model.mean_std) or not model.model.is_set(model.model.quantiles):
        raise ValueError('checkpoint missing normal statistics')
    if not torch.isfinite(model.model.mean_std['std']).all() or not (model.model.mean_std['std']>0).all(): raise ValueError('invalid normal standard deviation')
    return model

class Core(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model.model
    def forward(self,image): return self.model(image).anomaly_map
