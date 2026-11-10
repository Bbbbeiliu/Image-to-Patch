from __future__ import annotations
import hashlib
import inspect
from pathlib import Path
import re
from typing import Any, Callable
_EXTENDED_FACTORIES: dict[str, tuple[Callable[..., Any], Callable[..., dict[str, Any]]]] = {}

def _normalise_registry_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError('model name must be a string')
    normalised = name.strip().lower()
    if not re.fullmatch('[a-z][a-z0-9_]*', normalised):
        raise ValueError(f'Invalid extended model name: {name!r}')
    return normalised

def register_model(name: str, builder: Callable[..., Any], specifier: Callable[..., dict[str, Any]]) -> None:
    key = _normalise_registry_name(name)
    if key in _EXTENDED_FACTORIES:
        raise ValueError(f'An extended model is already registered as {key!r}')
    if not callable(builder) or not callable(specifier):
        raise TypeError('builder and specifier must be callable')
    _EXTENDED_FACTORIES[key] = (builder, specifier)

def supports(name: str) -> bool:
    return _normalise_registry_name(name) in _EXTENDED_FACTORIES

def model_names() -> tuple[str, ...]:
    return tuple(sorted(_EXTENDED_FACTORIES))

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _normalise_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError('model name must be a string')
    normalised = name.strip().lower()
    if normalised != 'manet':
        raise ValueError(f"Unknown extended model {name!r}; choose 'manet'")
    return normalised

def _validate_extra_kwargs(transunet_root, transunet_pretrained, pretrained) -> None:
    if transunet_root is not None or transunet_pretrained is not None:
        raise ValueError("TransUNet arguments are valid only with name='transunet'")
    if pretrained is not None:
        raise ValueError('Use encoder_weights for the MAnet ImageNet encoder')

def _build_manet(encoder_weights: str | None=None, transunet_root=None, transunet_pretrained=None, *, pretrained=None):
    _validate_extra_kwargs(transunet_root, transunet_pretrained, pretrained)
    if encoder_weights not in (None, 'imagenet'):
        raise ValueError("MAnet supports encoder_weights=None or 'imagenet'")
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError('MAnet requires segmentation-models-pytorch==0.5.0; install requirements.txt in the benchmark environment.') from exc
    return smp.MAnet(encoder_name='resnet34', encoder_weights=encoder_weights, in_channels=3, classes=1, activation=None)

def _manet_spec(encoder_weights: str | None=None, transunet_root=None, transunet_pretrained=None, *, pretrained=None) -> dict[str, Any]:
    _validate_extra_kwargs(transunet_root, transunet_pretrained, pretrained)
    if encoder_weights not in (None, 'imagenet'):
        raise ValueError("MAnet supports encoder_weights=None or 'imagenet'")
    import segmentation_models_pytorch as smp
    implementation = Path(inspect.getsourcefile(smp.MAnet) or '').resolve()
    if not implementation.is_file():
        raise FileNotFoundError('Could not locate the installed SMP MAnet implementation')
    factory_file = Path(__file__).resolve()
    return {'name': 'manet', 'input_shape': ['B', 3, 256, 256], 'output_shape': ['B', 1, 256, 256], 'output': 'logits', 'architecture': 'segmentation_models_pytorch.MAnet with ResNet-34 encoder', 'implementation': 'segmentation-models-pytorch', 'smp_version': getattr(smp, '__version__', None), 'encoder': 'resnet34', 'encoder_weights': encoder_weights, 'pretrained_default': None, 'paper_baseline_identity': 'SMP MAnet attention decoder with ResNet-34 encoder', 'source_files': {implementation.name: {'path': str(implementation), 'sha256': _sha256(implementation)}}, 'extension_factory': str(factory_file), 'extension_factory_sha256': _sha256(factory_file)}

def build_model(name: str, **kwargs):
    key = _normalise_registry_name(name)
    try:
        builder, _ = _EXTENDED_FACTORIES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown extended model {name!r}; registered: {', '.join(model_names())}") from exc
    return builder(**kwargs)

def model_spec(name: str, **kwargs) -> dict[str, Any]:
    key = _normalise_registry_name(name)
    try:
        _, specifier = _EXTENDED_FACTORIES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown extended model {name!r}; registered: {', '.join(model_names())}") from exc
    return specifier(**kwargs)
register_model('manet', _build_manet, _manet_spec)
