import argparse
import hashlib
from pathlib import Path
import urllib.request

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def download(url, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != expected:
            raise ValueError('Existing file checksum differs: ' + str(target))
        return
    temp = target.with_suffix(target.suffix + '.download')
    urllib.request.urlretrieve(url, temp)
    if sha(temp) != expected:
        raise ValueError('Download checksum differs: ' + str(temp))
    temp.replace(target)

def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', choices=['resnet34', 'vgg16', 'transunet', 'sam'], required=True)
    args = p.parse_args()
    home = root / 'benchmark'
    specs = {'resnet34': ('https://download.pytorch.org/models/resnet34-b627a593.pth', home / 'torch_cache/hub/checkpoints/resnet34-b627a593.pth', 'b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f'), 'vgg16': ('https://download.pytorch.org/models/vgg16-397923af.pth', home / 'pretrained/vgg16-397923af.pth', '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'), 'transunet': ('https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz', home / 'pretrained/R50+ViT-B_16.npz', 'ff009bf39bb4f9198b834cfe46aba2bfdaf730e933ab3e3c4b1edf4226eaafbe'), 'sam': ('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', home / 'extended_baselines/samed/weights/sam_vit_b_01ec64.pth', 'ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912')}
    for name in args.models:
        download(*specs[name])
        print(name, 'verified')
        if name == 'vgg16':
            import torch
            full = torch.load(specs[name][1], map_location='cpu', weights_only=True)
            features = {k: v for k, v in full.items() if k.startswith('features.')}
            if len(features) != 26:
                raise ValueError('Expected 26 VGG16 feature tensors')
            target = home / 'extended_baselines/panet/vgg16_features.pt'
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(features, target)
if __name__ == '__main__':
    main()
