from __future__ import annotations
from contextlib import contextmanager
import copy
import hashlib
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any
from unittest import mock
try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError('The segmentation benchmark requires PyTorch. Install its pinned dependencies from requirements.txt.') from exc
_ROOT = Path(__file__).resolve().parent.parent
_VANILLA_UNET_SOURCE = _ROOT / 'benchmark/vanilla_unet.py'
_RETAINED_SOURCE = _ROOT / 'segmentation/models/resnet34_unet_attention.py'
_TRANSUNET_REPOSITORY = 'https://github.com/Beckschen/TransUNet.git'
_SMP_MODELS = {'unet_resnet34', 'unetpp', 'deeplabv3plus'}
_MODEL_NAMES = {'unet', 'unet_resnet34', 'unetpp', 'deeplabv3plus', 'transunet', 'retained'}

@contextmanager
def _without_bytecode_writes():
    old_value = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = old_value

def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _git_info(path: Path | None) -> dict[str, str | None]:
    if path is None:
        return {'git_revision': None, 'git_remote': None}
    try:
        rev = subprocess.run(['git', '-C', str(path), 'rev-parse', '--verify', 'HEAD'], capture_output=True, text=True, check=False, timeout=5)
        remote = subprocess.run(['git', '-C', str(path), 'remote', 'get-url', 'origin'], capture_output=True, text=True, check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return {'git_revision': None, 'git_remote': None}
    return {'git_revision': rev.stdout.strip() if rev.returncode == 0 else None, 'git_remote': remote.stdout.strip() if remote.returncode == 0 else None}

def _source_record(path: Path, repo_root: Path | None=None) -> dict[str, Any]:
    return {'path': str(path.resolve()), 'sha256': _sha256(path), **_git_info(repo_root or path.parent)}

def _normalise_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError('model name must be a string')
    normalised = name.strip().lower()
    if normalised not in _MODEL_NAMES:
        allowed = ', '.join(sorted(_MODEL_NAMES))
        raise ValueError(f'Unknown model {name!r}; choose one of: {allowed}')
    return normalised

def _optional_pretrained_weights(encoder_weights: str | None, pretrained: bool | str | None) -> str | None:
    if pretrained is None:
        if encoder_weights is not None and (not isinstance(encoder_weights, str)):
            raise TypeError('encoder_weights must be a string or None')
        return encoder_weights
    if encoder_weights is not None:
        raise ValueError('Pass encoder_weights or pretrained, not both')
    if isinstance(pretrained, bool):
        return 'imagenet' if pretrained else None
    if isinstance(pretrained, str):
        return pretrained
    raise TypeError('pretrained must be a bool, weight-name string, or None')

def _load_unique_python_file(source: Path, module_prefix: str) -> ModuleType:
    if not source.is_file():
        raise FileNotFoundError(f'Required local model source is missing: {source}')
    source_hash = _sha256(source)
    module_name = f'_{module_prefix}_{source_hash[:16]}'
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load Python model source: {source}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with _without_bytecode_writes():
            spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module

def _load_vanilla_unet() -> nn.Module:
    module = _load_unique_python_file(_VANILLA_UNET_SOURCE, 'benchmark_vanilla_unet')
    model_type = getattr(module, 'UNet', None)
    if model_type is None:
        raise ImportError(f'Local U-Net source does not define UNet: {_VANILLA_UNET_SOURCE}')
    return model_type(in_channels=3, out_channels=1)

def _require_smp():
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError('SMP baseline models require segmentation-models-pytorch==0.5.0. Install requirements.txt in the active environment.') from exc
    return smp

def _load_retained_without_pretrained_weights() -> nn.Module:
    smp = _require_smp()
    module = _load_unique_python_file(_RETAINED_SOURCE, 'benchmark_retained_unet')
    unet_model_module = importlib.import_module('segmentation_models_pytorch.decoders.unet.model')
    original_get_encoder = smp.encoders.get_encoder

    def no_pretrained_encoder(*args, **kwargs):
        kwargs['weights'] = None
        return original_get_encoder(*args, **kwargs)
    with mock.patch.object(smp.encoders, 'get_encoder', side_effect=no_pretrained_encoder), mock.patch.object(unet_model_module, 'get_encoder', side_effect=no_pretrained_encoder), _without_bytecode_writes():
        model_type = getattr(module, 'UNetWithAttentionResNet34', None)
        if model_type is None:
            raise ImportError(f'Retained source does not define UNetWithAttentionResNet34: {_RETAINED_SOURCE}')
        return model_type(in_channels=3, out_channels=1)

def _official_transunet_root(root: str | os.PathLike[str] | None) -> Path:
    if root is None:
        raise ValueError('TransUNet requires an explicit checkout path via transunet_root; the official Beckschen/TransUNet source is not vendored or downloaded automatically.')
    root_path = Path(root).expanduser().resolve()
    modeling_file = root_path / 'networks' / 'vit_seg_modeling.py'
    config_file = root_path / 'networks' / 'vit_seg_configs.py'
    if not modeling_file.is_file() or not config_file.is_file():
        raise FileNotFoundError(f'Expected networks/vit_seg_modeling.py and networks/vit_seg_configs.py under {root_path}')
    info = _git_info(root_path)
    remote = (info['git_remote'] or '').lower().rstrip('/')
    if remote.endswith('.git'):
        remote = remote[:-4]
    official_remotes = {'https://github.com/beckschen/transunet', 'http://github.com/beckschen/transunet', 'git@github.com:beckschen/transunet', 'ssh://git@github.com/beckschen/transunet'}
    if not info['git_revision'] or remote not in official_remotes:
        raise ValueError(f"transunet_root must be a git checkout whose origin is the official {_TRANSUNET_REPOSITORY} repository; found origin={info['git_remote']!r}")
    return root_path

def _load_transunet(root: str | os.PathLike[str], pretrained_npz: str | os.PathLike[str] | None) -> nn.Module:
    root_path = _official_transunet_root(root)
    model_file = (root_path / 'networks' / 'vit_seg_modeling.py').resolve()
    package_path = (root_path / 'networks').resolve()
    cached_package = sys.modules.get('networks')
    if cached_package is not None:
        cached_paths = [Path(p).resolve() for p in getattr(cached_package, '__path__', ())]
        if package_path not in cached_paths:
            raise ImportError("A different package named 'networks' is already imported; start a fresh process to load this explicit TransUNet checkout safely.")
    cached_modeling = sys.modules.get('networks.vit_seg_modeling')
    if cached_modeling is not None and Path(cached_modeling.__file__).resolve() != model_file:
        raise ImportError('A different networks.vit_seg_modeling module is already imported; start a fresh process to load this TransUNet checkout safely.')
    root_string = str(root_path)
    added_path = root_string not in sys.path
    if added_path:
        sys.path.insert(0, root_string)
    try:
        with _without_bytecode_writes():
            try:
                api = importlib.import_module('networks.vit_seg_modeling')
            except ModuleNotFoundError as exc:
                if exc.name in {'ml_collections', 'scipy'}:
                    package = 'ml-collections' if exc.name == 'ml_collections' else 'scipy'
                    raise ImportError(f'TransUNet requires {package}; install requirements.txt in the active environment.') from exc
                raise
            if Path(api.__file__).resolve() != model_file:
                raise ImportError(f'Loaded TransUNet module from unexpected path: {api.__file__}')
            config_template = api.CONFIGS.get('R50-ViT-B_16')
            if config_template is None:
                raise ImportError("Official TransUNet checkout lacks CONFIGS['R50-ViT-B_16']")
            config = copy.deepcopy(config_template)
            config.n_classes = 1
            config.n_skip = 3
            config.patches.grid = (16, 16)
            model = api.VisionTransformer(config, img_size=256, num_classes=1)
            if pretrained_npz is not None:
                npz_path = Path(pretrained_npz).expanduser().resolve()
                if not npz_path.is_file():
                    raise FileNotFoundError(f'TransUNet pretrained .npz file not found: {npz_path}')
                import numpy as np
                with np.load(npz_path) as weights:

                    class PortableArchive:

                        def __getitem__(self, key):
                            return weights[key.replace('\\', '/')]
                    model.load_from(PortableArchive())
            return model
    finally:
        if added_path:
            try:
                sys.path.remove(root_string)
            except ValueError:
                pass

def build_model(name: str, encoder_weights: str | None=None, transunet_root: str | os.PathLike[str] | None=None, transunet_pretrained: str | os.PathLike[str] | None=None, *, pretrained: bool | str | None=None) -> nn.Module:
    model_name = _normalise_name(name)
    if model_name == 'unet':
        if encoder_weights is not None or pretrained is not None:
            raise ValueError('The local vanilla U-Net has no encoder_weights/pretrained option')
        if transunet_root is not None or transunet_pretrained is not None:
            raise ValueError("TransUNet arguments are valid only with name='transunet'")
        model = _load_vanilla_unet()
    elif model_name in _SMP_MODELS:
        if transunet_root is not None or transunet_pretrained is not None:
            raise ValueError("TransUNet arguments are valid only with name='transunet'")
        resolved_weights = _optional_pretrained_weights(encoder_weights, pretrained)
        smp = _require_smp()
        model_type = {'unet_resnet34': smp.Unet, 'unetpp': smp.UnetPlusPlus, 'deeplabv3plus': smp.DeepLabV3Plus}[model_name]
        model = model_type(encoder_name='resnet34', encoder_weights=resolved_weights, in_channels=3, classes=1, activation=None)
    elif model_name == 'transunet':
        if encoder_weights is not None or pretrained is not None:
            raise ValueError('TransUNet does not accept encoder_weights/pretrained; use transunet_pretrained to load an explicit local .npz file.')
        model = _load_transunet(transunet_root, transunet_pretrained)
    else:
        if encoder_weights is not None or pretrained is not None:
            raise ValueError('The retained model is evaluation-only and rejects encoder_weights/pretrained')
        if transunet_root is not None or transunet_pretrained is not None:
            raise ValueError("TransUNet arguments are valid only with name='transunet'")
        model = _load_retained_without_pretrained_weights()
    return model

def model_spec(name: str, encoder_weights: str | None=None, transunet_root: str | os.PathLike[str] | None=None, transunet_pretrained: str | os.PathLike[str] | None=None, *, pretrained: bool | str | None=None) -> dict[str, Any]:
    model_name = _normalise_name(name)
    result: dict[str, Any] = {'name': model_name, 'input_shape': ['B', 3, 256, 256], 'output_shape': ['B', 1, 256, 256], 'output': 'logits', 'pretrained_default': None, 'git_revision': None}
    if model_name == 'unet':
        if encoder_weights is not None or pretrained is not None:
            raise ValueError('The local vanilla U-Net has no encoder_weights/pretrained option')
        if transunet_root is not None or transunet_pretrained is not None:
            raise ValueError("TransUNet arguments are valid only with name='transunet'")
        source = _source_record(_VANILLA_UNET_SOURCE, _VANILLA_UNET_SOURCE.parents[2])
        result.update(architecture='local vanilla U-Net (64/128/256/512/1024 channels)', implementation='benchmark/vanilla_unet.py::UNet', paper_baseline_identity='vanilla U-Net implementation from this workspace', source_sha256=source['sha256'], git_revision=source['git_revision'], git_remote=source['git_remote'], source_files={'unet.py': source})
    elif model_name in _SMP_MODELS:
        if transunet_root is not None or transunet_pretrained is not None:
            raise ValueError("TransUNet arguments are valid only with name='transunet'")
        resolved_weights = _optional_pretrained_weights(encoder_weights, pretrained)
        source_files: dict[str, Any] = {}
        try:
            smp = _require_smp()
            smp_version = getattr(smp, '__version__', None)
            implementation_type = {'unet_resnet34': smp.Unet, 'unetpp': smp.UnetPlusPlus, 'deeplabv3plus': smp.DeepLabV3Plus}[model_name]
            implementation_path = inspect.getsourcefile(implementation_type)
            if implementation_path:
                source = Path(implementation_path).resolve()
                source_files[source.name] = _source_record(source)
        except ImportError:
            smp_version = None
        result.update(architecture={'unet_resnet34': 'segmentation_models_pytorch.Unet with ResNet-34 encoder', 'unetpp': 'segmentation_models_pytorch.UnetPlusPlus with ResNet-34 encoder', 'deeplabv3plus': 'segmentation_models_pytorch.DeepLabV3Plus with ResNet-34 encoder'}[model_name], implementation='segmentation-models-pytorch', smp_version=smp_version, encoder='resnet34', encoder_weights=resolved_weights, paper_baseline_identity='SMP ResNet-34 variant; not the local vanilla U-Net' if model_name == 'unet_resnet34' else 'SMP ResNet-34 segmentation architecture', source_files=source_files)
    elif model_name == 'transunet':
        if encoder_weights is not None or pretrained is not None:
            raise ValueError('TransUNet does not accept encoder_weights/pretrained; use transunet_pretrained')
        root_path = Path(transunet_root).expanduser().resolve() if transunet_root is not None else None
        if root_path is not None:
            official_root = _official_transunet_root(root_path)
            records = {rel: _source_record(official_root / rel, official_root) for rel in ('networks/vit_seg_modeling.py', 'networks/vit_seg_configs.py', 'networks/vit_seg_modeling_resnet_skip.py')}
            provenance = _git_info(official_root)
        else:
            records = {}
            provenance = {'git_revision': None, 'git_remote': _TRANSUNET_REPOSITORY}
        result.update(architecture='Beckschen/TransUNet VisionTransformer R50-ViT-B_16', implementation='official external networks.vit_seg_modeling.VisionTransformer', config={'name': 'R50-ViT-B_16', 'n_classes': 1, 'n_skip': 3, 'patches.grid': [16, 16], 'img_size': 256}, external_repository=_TRANSUNET_REPOSITORY, external_root=str(root_path) if root_path is not None else None, pretrained_npz=str(Path(transunet_pretrained).expanduser().resolve()) if transunet_pretrained is not None else None, windows_npz_key_adapter='normalize backslashes to slash during archive lookup; vendor code unchanged', source_files=records, **provenance)
    else:
        if encoder_weights is not None or pretrained is not None:
            raise ValueError('The retained model is evaluation-only and rejects encoder_weights/pretrained')
        if transunet_root is not None or transunet_pretrained is not None:
            raise ValueError("TransUNet arguments are valid only with name='transunet'")
        source = _source_record(_RETAINED_SOURCE, _RETAINED_SOURCE.parents[2])
        result.update(architecture='retained UNetWithAttentionResNet34 checkpoint architecture', implementation='segmentation/models/resnet34_unet_attention.py', encoder='resnet34', encoder_weights='None forced during construction; load retained checkpoint strictly afterward', evaluation_only=True, source_sha256=source['sha256'], git_revision=source['git_revision'], git_remote=source['git_remote'], source_files={'resnet34_unet_attention.py': source})
    return result
