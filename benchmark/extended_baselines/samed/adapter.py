from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import torch
from torch import nn
PACKAGE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = PACKAGE_ROOT / 'vendor'
RUNTIME_DEPS = PACKAGE_ROOT / 'runtime_deps'
DEFAULT_SAM_CHECKPOINT = PACKAGE_ROOT / 'weights' / 'sam_vit_b_01ec64.pth'
REFERENCE_SAMED_CHECKPOINT = PACKAGE_ROOT / 'weights' / 'epoch_159.pth'
SAMED_REPOSITORY = 'https://github.com/hitachinsk/SAMed.git'
SAM_REPOSITORY = 'https://github.com/facebookresearch/segment-anything'
SAM_CHECKPOINT_URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
SAMED_CHECKPOINT_URL = 'https://drive.google.com/file/d/1P0Bm-05l-rfeghbrT1B62v5eN-3A-uOr/view?usp=share_link'
SAMED_COMMIT = 'e351840481f51d0c7dee50007f29d02197b63d99'
SAM_CHECKPOINT_SHA256 = 'ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912'
SAMED_CHECKPOINT_SHA256 = '9c207f07fd08c133cb9e654b421a399f799f7e0222d62c10e431ab9839f541b6'
IMAGE_SIZE = 256
LORA_RANK = 4

def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _git_revision() -> str | None:
    try:
        result = subprocess.run(['git', '-C', str(VENDOR_ROOT), 'rev-parse', '--verify', 'HEAD'], capture_output=True, text=True, check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None

def _normalise_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError('model name must be a string')
    if name.strip().lower().replace('-', '_') not in {'samed', 'samed_binary'}:
        raise ValueError(f'Unknown SAMed adapter name: {name!r}')
    return 'samed'

def _validate_options(name: str, encoder_weights: str | None, pretrained: bool | str | os.PathLike[str] | None, *, sam_checkpoint: str | os.PathLike[str] | None, image_size: int, rank: int, lora_checkpoint: str | os.PathLike[str] | None, transunet_root: str | os.PathLike[str] | None, transunet_pretrained: str | os.PathLike[str] | None) -> tuple[Path | None, Path | None]:
    _normalise_name(name)
    if encoder_weights is not None:
        raise ValueError('SAMed accepts a SAM checkpoint path via pretrained/sam_checkpoint, not encoder_weights')
    if transunet_root is not None or transunet_pretrained is not None:
        raise ValueError('TransUNet arguments are not valid for the SAMed adapter')
    if lora_checkpoint is not None:
        raise ValueError('The Synapse-trained SAMed LoRA checkpoint is intentionally not loaded for this binary adapter; use the official rank-4 zero-init LoRA and train on the benchmark task.')
    if image_size != IMAGE_SIZE:
        raise ValueError(f'This adapter is fixed to the benchmark input size {IMAGE_SIZE}')
    if rank != LORA_RANK:
        raise ValueError(f'The upstream SAMed configuration uses LoRA rank {LORA_RANK}')
    if isinstance(pretrained, bool):
        default_checkpoint = DEFAULT_SAM_CHECKPOINT if pretrained else None
    elif pretrained is None:
        default_checkpoint = DEFAULT_SAM_CHECKPOINT
    elif isinstance(pretrained, (str, os.PathLike)):
        if str(pretrained).strip().lower() in {'imagenet', 'none', 'false', 'true'}:
            raise ValueError('pretrained must be bool or a local SAM ViT-B checkpoint path')
        default_checkpoint = Path(pretrained).expanduser()
    else:
        raise TypeError('pretrained must be bool, a local checkpoint path, or None')
    selected_checkpoint = Path(sam_checkpoint).expanduser() if sam_checkpoint is not None else default_checkpoint
    if selected_checkpoint is not None:
        selected_checkpoint = selected_checkpoint.resolve()
        if not selected_checkpoint.is_file():
            raise FileNotFoundError(f'SAM ViT-B checkpoint not found: {selected_checkpoint}')
    reference_checkpoint = REFERENCE_SAMED_CHECKPOINT if REFERENCE_SAMED_CHECKPOINT.is_file() else None
    return (selected_checkpoint, reference_checkpoint)

def _load_official_components():
    if not (VENDOR_ROOT / 'sam_lora_image_encoder.py').is_file():
        raise FileNotFoundError(f'Vendored SAMed source is incomplete: {VENDOR_ROOT}')
    for directory in (VENDOR_ROOT, RUNTIME_DEPS):
        if directory.is_dir() and str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    existing = sys.modules.get('segment_anything')
    existing_file = getattr(existing, '__file__', None) if existing is not None else None
    if existing_file and (not Path(existing_file).resolve().is_relative_to(VENDOR_ROOT.resolve())):
        raise ImportError("A different 'segment_anything' package is already loaded. Run SAMed in a clean process or import this adapter before another Segment Anything implementation.")
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from sam_lora_image_encoder import LoRA_Sam
        from segment_anything import sam_model_registry
    finally:
        sys.dont_write_bytecode = previous
    return (LoRA_Sam, sam_model_registry)

class SAMedBinary(nn.Module):

    def __init__(self, lora_sam: nn.Module, image_size: int=IMAGE_SIZE) -> None:
        super().__init__()
        self.lora_sam = lora_sam
        self.image_size = image_size

    def _two_class_logits(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f'Expected (B, 3, {self.image_size}, {self.image_size}) RGB input')
        if tuple(images.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(f'Expected {self.image_size}x{self.image_size} letterboxed inputs')
        if not torch.is_floating_point(images):
            raise TypeError('SAMed expects float32 RGB values scaled to [0, 1] by the shared dataset')
        outputs = self.lora_sam(images, multimask_output=True, image_size=self.image_size)
        mask_logits = outputs['masks']
        if mask_logits.ndim != 4 or mask_logits.shape[1] != 2:
            raise RuntimeError(f'Expected the two binary mask-token logits, got {tuple(mask_logits.shape)}')
        return mask_logits

    def forward_two_class_logits(self, images: torch.Tensor) -> torch.Tensor:
        return self._two_class_logits(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        class_logits = self._two_class_logits(images)
        return class_logits[:, 1:2] - class_logits[:, 0:1]

def build_model(name: str='samed', encoder_weights: str | None=None, pretrained: bool | str | os.PathLike[str] | None=None, *, sam_checkpoint: str | os.PathLike[str] | None=None, lora_checkpoint: str | os.PathLike[str] | None=None, image_size: int=IMAGE_SIZE, rank: int=LORA_RANK, transunet_root: str | os.PathLike[str] | None=None, transunet_pretrained: str | os.PathLike[str] | None=None, **kwargs: Any) -> SAMedBinary:
    if kwargs:
        unknown = ', '.join(sorted(kwargs))
        raise TypeError(f'Unexpected SAMed build option(s): {unknown}')
    selected_checkpoint, _ = _validate_options(name, encoder_weights, pretrained, sam_checkpoint=sam_checkpoint, image_size=image_size, rank=rank, lora_checkpoint=lora_checkpoint, transunet_root=transunet_root, transunet_pretrained=transunet_pretrained)
    LoRA_Sam, sam_model_registry = _load_official_components()
    sam, _embedding_size = sam_model_registry['vit_b'](image_size=IMAGE_SIZE, num_classes=1, checkpoint=str(selected_checkpoint) if selected_checkpoint is not None else None, pixel_mean=[0.0, 0.0, 0.0], pixel_std=[1.0, 1.0, 1.0])
    return SAMedBinary(LoRA_Sam(sam, r=rank), image_size=IMAGE_SIZE)

def model_spec(name: str='samed', encoder_weights: str | None=None, pretrained: bool | str | os.PathLike[str] | None=None, *, sam_checkpoint: str | os.PathLike[str] | None=None, lora_checkpoint: str | os.PathLike[str] | None=None, image_size: int=IMAGE_SIZE, rank: int=LORA_RANK, transunet_root: str | os.PathLike[str] | None=None, transunet_pretrained: str | os.PathLike[str] | None=None, **kwargs: Any) -> dict[str, Any]:
    if kwargs:
        unknown = ', '.join(sorted(kwargs))
        raise TypeError(f'Unexpected SAMed spec option(s): {unknown}')
    selected_checkpoint, reference_checkpoint = _validate_options(name, encoder_weights, pretrained, sam_checkpoint=sam_checkpoint, image_size=image_size, rank=rank, lora_checkpoint=lora_checkpoint, transunet_root=transunet_root, transunet_pretrained=transunet_pretrained)
    sam_path = selected_checkpoint
    reference_path = reference_checkpoint
    if sam_path is None:
        sam_weight_record = {'path': None, 'sha256': None, 'loaded': False}
    else:
        sam_weight_record = {'path': str(sam_path), 'sha256': _sha256(sam_path), 'loaded': True, 'source_url': SAM_CHECKPOINT_URL if sam_path == DEFAULT_SAM_CHECKPOINT.resolve() else None}
    if reference_path is None:
        reference_record = {'path': str(REFERENCE_SAMED_CHECKPOINT), 'sha256': None, 'loaded': False}
    else:
        reference_record = {'path': str(reference_path), 'sha256': _sha256(reference_path), 'loaded': False, 'source_url': SAMED_CHECKPOINT_URL, 'reason_not_loaded': 'This checkpoint has an eight-organ Synapse head, so it is not used to initialize the clinical binary decoder.'}
    manifest_path = PACKAGE_ROOT / 'source_hashes.json'
    vendored_source = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
    return {'name': _normalise_name(name), 'architecture': 'Official hitachinsk/SAMed SAM ViT-B image encoder with rank-4 q/v LoRA and a two-token binary mask head', 'implementation': 'SAMedBinary wrapping vendor/sam_lora_image_encoder.py::LoRA_Sam and vendor/segment_anything', 'external_repository': SAMED_REPOSITORY, 'external_commit': _git_revision() or SAMED_COMMIT, 'source_hash_manifest': str(manifest_path), 'vendor_source_hashes': vendored_source.get('files', {}), 'sam_checkpoint_repository': SAM_REPOSITORY, 'configuration': {'image_size': IMAGE_SIZE, 'vit': 'vit_b', 'rank': LORA_RANK, 'num_classes_argument': 1, 'mask_tokens': ['background', 'foreground'], 'prompts': {'points': None, 'boxes': None, 'masks': None}, 'pixel_mean': [0.0, 0.0, 0.0], 'pixel_std': [1.0, 1.0, 1.0]}, 'input_shape': ['B', 3, IMAGE_SIZE, IMAGE_SIZE], 'input_protocol': 'RGB float32; shared loader scales raw uint8 pixels by /255 and applies aspect-preserving centered zero letterbox to 256x256', 'output_shape': ['B', 1, IMAGE_SIZE, IMAGE_SIZE], 'output': 'foreground logit minus background logit (equivalent binary two-class log odds)', 'loss_protocol': {'benchmark': 'shared BCEWithLogits + sigmoid soft Dice, Dice coefficient 1.0', 'official_samed': 'multiclass CrossEntropy + softmax Dice, combined with dice_param=0.8 on low-resolution labels', 'deviation': "The clinical binary run uses the benchmark's shared binary objective; it differs from official SAMed's eight-organ Synapse training objective."}, 'weights': {'sam_vit_b': sam_weight_record, 'official_samed_synapse_checkpoint_reference_only': reference_record, 'lora_initialization': 'official upstream rank-4 LoRA_Sam.reset_parameters: Kaiming-uniform A and zero B; no Synapse LoRA weights loaded', 'binary_decoder_initialization': 'SAM base checkpoint loads shared decoder transformer/upscaling weights; two mask tokens, per-token hypernetworks and IoU head are initialized for the binary task by official SAMed builder'}, 'trainable_parameters': 'upstream wrapper freezes the original image encoder; rank-4 image q/v LoRA, prompt encoder and mask decoder remain trainable', 'pretrained_default': str(DEFAULT_SAM_CHECKPOINT) if sam_path == DEFAULT_SAM_CHECKPOINT.resolve() else None, 'git_revision': SAMED_COMMIT}
build = build_model
spec = model_spec
