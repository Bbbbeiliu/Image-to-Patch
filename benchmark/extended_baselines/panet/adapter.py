from pathlib import Path
import hashlib, importlib.util, json, subprocess, sys, os
import torch
from torch import nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parent
HOME = ROOT.parents[1]
VENDOR = ROOT / 'vendor'
WEIGHTS = ROOT / 'vgg16_features.pt'
CURRENT = None

def official_class():
    name = '_clinical_official_panet'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, VENDOR / 'models/__init__.py', submodule_search_locations=[str(VENDOR / 'models')])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(name + '.fewshot').FewShotSeg

def fixed_support():
    lock = json.loads(Path(os.environ.get('SCAR_DATASET_LOCK', str(HOME / 'dataset_lock.json'))).read_text(encoding='utf-8'))
    records = sorted([r for r in lock['records'] if r['split'] == 'train'], key=lambda r: r['sample_id'])
    generator = torch.Generator().manual_seed(42)
    return records[int(torch.randperm(len(records), generator=generator)[0])]

def model_spec(name='panet', **kwargs):
    files = [Path(__file__), VENDOR / 'models/fewshot.py', VENDOR / 'models/vgg.py', WEIGHTS]
    return dict(name='PANet (official VGG16), clinical 1-way 1-shot adaptation', repository='https://github.com/kaixin96/PANet', revision=subprocess.check_output(['git', '-C', str(VENDOR), 'rev-parse', 'HEAD'], text=True).strip(), source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, initialization='ImageNet VGG16 features', input_normalization='ImageNet mean/std inside adapter', training_objective='binary cross-entropy (equivalent to 2-class CE) + prototype alignment weight 1', deterministic_adaptation='Binary prototype-alignment CE computed as BCE of foreground-minus-background logits; avoids nondeterministic CUDA nll_loss2d', episodes='1 way, 1 support, 1 query per episode; batch 4 episodes; support group excludes every query group in batch', validation_test_support=fixed_support()['sample_id'], support_selection='seed42 train-only sample before evaluation', support_augmentation='deterministic letterbox; query uses existing training augmentation', limitation='Within-domain single-class adaptation; not novel-class few-shot generalization; one fixed evaluation support')

class PANetAdapter(nn.Module):

    def __init__(self, initialize=False):
        super().__init__()

        class BinaryDeterministicPANet(official_class()):

            def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
                labels = pred.argmax(dim=1, keepdim=True)
                if not (labels == 1).any():
                    return qry_fts.sum() * 0
                masks = torch.stack([labels == 0, labels == 1], dim=1).float()
                prototypes = (qry_fts.unsqueeze(1) * masks).sum((0, 3, 4)) / (masks.sum((0, 3, 4)) + 1e-05)
                loss = 0
                for shot in range(len(fore_mask[0])):
                    features = supp_fts[0, [shot]]
                    scores = torch.stack([self.calDist(features, p) for p in [prototypes[[0]], prototypes[[1]]]], dim=1)
                    scores = F.interpolate(scores, size=fore_mask.shape[-2:], mode='bilinear')
                    valid = (fore_mask[0, shot] == 1) | (back_mask[0, shot] == 1)
                    difference = (scores[:, 1] - scores[:, 0])[0]
                    loss = loss + F.binary_cross_entropy_with_logits(difference[valid], fore_mask[0, shot][valid]) / len(fore_mask[0])
                return loss
        self.net = BinaryDeterministicPANet(pretrained_path=str(WEIGHTS) if initialize else None, cfg={'align': True})
        import benchmark as b
        datasets = b.datasets(b.DEFAULT_DATA)
        self.support_dataset = datasets['train']
        self.support_dataset.transform = datasets['val'].transform
        lock = json.loads(Path(os.environ.get('SCAR_DATASET_LOCK', str(HOME / 'dataset_lock.json'))).read_text(encoding='utf-8'))
        mapping = {r['sample_id']: r['group'] for r in lock['records'] if r['split'] == 'train'}
        self.groups = [mapping[Path(n).stem] for n in self.support_dataset.image_names]
        sid = fixed_support()['sample_id']
        self.fixed_index = [Path(n).stem for n in self.support_dataset.image_names].index(sid)
        self.alignment_loss = 0

    def forward(self, x):
        query = x[:, :3]
        indices = x[:, 3, 0, 0].long().tolist() if self.training and x.shape[1] == 4 else None
        if self.training and indices is None:
            raise ValueError('PANet training requires query indices for support/query group isolation')
        excluded = {self.groups[i] for i in indices} if indices is not None else set()
        candidates = [i for i, g in enumerate(self.groups) if g not in excluded]
        supports = []
        masks = []
        for _ in range(len(x)):
            idx = candidates[int(torch.randint(len(candidates), (1,)).item())] if self.training else self.fixed_index
            image, mask, _ = self.support_dataset[idx]
            supports.append(image)
            masks.append(mask[0])
        support = torch.stack(supports).to(x.device)
        fg = torch.stack(masks).to(x.device)
        mean = query.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = query.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        logits, align = self.net([[(support - mean) / std]], [[fg]], [[1 - fg]], [(query - mean) / std])
        self.alignment_loss = align
        return logits[:, 1:2] - logits[:, :1]

class IndexedDataset:

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __getitem__(self, index):
        x, y, z = self.base[index]
        return (torch.cat([x, torch.full_like(x[:1], float(index))]), y, z)

def install(b):
    original_datasets = b.datasets
    original_loss = b.loss_fn

    def datasets(data):
        result = original_datasets(data)
        result['train'] = IndexedDataset(result['train'])
        return result
    b.datasets = datasets
    b._panet_original_datasets = original_datasets

    def loss(logits, targets):
        if CURRENT is not None:
            return F.binary_cross_entropy_with_logits(logits, targets) + CURRENT.alignment_loss
        return original_loss(logits, targets)
    b.loss_fn = loss
    return original_datasets

def build_model(name='panet', encoder_weights=None, **kwargs):
    global CURRENT
    import benchmark as b
    active = b.datasets
    original = getattr(b, '_panet_original_datasets', b.datasets)
    b.datasets = original
    try:
        CURRENT = PANetAdapter(initialize=True)
    finally:
        b.datasets = active
    return CURRENT
