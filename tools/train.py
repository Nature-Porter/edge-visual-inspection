import argparse
import importlib.metadata
import os
from common import ROOT, config, save, sha256, verified_records

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/metal_nut.yaml');p.add_argument('--smoke',action='store_true');p.add_argument('--resume');a=p.parse_args();c=config(a.config)
    os.chdir(ROOT)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    for split in ['train_normal','stats_normal','validation']: verified_records(split)
    teacher=ROOT/'pre_trained/efficientad_pretrained_weights/pretrained_teacher_small.pth'
    if not teacher.exists(): teacher=ROOT/'pre_trained/efficientad_pretrained_weights/pretrained_teacher_s.pth'
    if not teacher.exists() or not (ROOT/c['training']['imagenette']).is_dir(): raise FileNotFoundError('official teacher/Imagenette missing; place user-downloaded files as described in README.md')
    import torch
    from lightning.pytorch import Trainer, seed_everything
    from model import ExplicitData,create,restore
    if not torch.cuda.is_available(): raise RuntimeError('real GPU training blocked; no implicit CPU substitution')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seed_everything(c['project']['seed'],workers=True)
    out=ROOT/'artifacts/metal_nut'/('smoke' if a.smoke else 'training');out.mkdir(parents=True,exist_ok=True)
    if (out/'model.ckpt').exists() and not a.resume: raise FileExistsError('checkpoint exists; use --resume or separate output')
    model=create(c);data=ExplicitData();steps=10 if a.smoke else c['training']['max_steps']
    trainer=Trainer(accelerator='gpu',devices=1,max_steps=steps,max_epochs=-1,num_sanity_val_steps=0,
        default_root_dir=str(out),enable_checkpointing=False,logger=False,enable_progress_bar=False,deterministic=True)
    trainer.fit(model,datamodule=data,ckpt_path=a.resume)
    model.cuda().eval()
    model.model.quantiles.update(model.map_norm_quantiles(data.stats_dataloader()))
    trainer.save_checkpoint(out/'model.ckpt')
    loaded=restore(c,out/'model.ckpt').cuda();sample=next(iter(data.stats_dataloader())).image.cuda()
    with torch.no_grad():
        x=model.model(sample).anomaly_map;y=loaded.model(sample).anomaly_map;z=loaded.model(sample).anomaly_map
    torch.testing.assert_close(x,y,atol=c['alignment']['atol'],rtol=c['alignment']['rtol']);torch.testing.assert_close(y,z,atol=c['alignment']['atol'],rtol=c['alignment']['rtol'])
    save(out/'training.json',{'status':'smoke' if a.smoke else 'trained','steps':trainer.global_step,'seed':42,
        'checkpoint_sha256':sha256(out/'model.ckpt'),'teacher_sha256':sha256(teacher),'imagenette_archive_sha256':sha256(ROOT/'data/raw/imagenette2.tgz'),
        'train_sha256':sha256(ROOT/'data/train_normal.json'),'stats_sha256':sha256(ROOT/'data/stats_normal.json'),
        'tools':{k:importlib.metadata.version(k) for k in ['anomalib','torch','torchvision']},
        'repeat_max_abs':float((y-z).abs().max()),'restore_max_abs':float((x-y).abs().max()),
        'statistics':{k:{n:v.detach().cpu().tolist() for n,v in getattr(model.model,k).items()} for k in ['mean_std','quantiles']}})
if __name__=='__main__':main()
