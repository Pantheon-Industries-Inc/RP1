import json
import urllib.request
from pathlib import Path
import torch

from loguru import logger as logging
from tqdm import tqdm

from stable_worldmodel.utils import HF_BASE_URL
from stable_worldmodel.data import get_cache_dir, ensure_dir_exists


def save_pretrained(
    model: torch.nn.Module,
    run_name: str,
    config: dict | None = None,
    config_key: str | None = None,
    filename: str = 'weights.pt',
    cache_dir: str = None,
):
    from omegaconf import OmegaConf

    ckpt_dir = get_cache_dir(cache_dir, sub_folder='checkpoints') / run_name
    ensure_dir_exists(ckpt_dir)

    checkpoint_path = ckpt_dir / filename
    torch.save(model.state_dict(), checkpoint_path)

    if config is None:
        logging.warning('No config! Loading will have to be done manually.')
        return

    if config_key is not None and config_key in config:
        config = config[config_key]

    config_path = ckpt_dir / 'config.json'

    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logging.info(f'📦📦📦 Model saved to {checkpoint_path} 📦📦📦')

    return


def load_pretrained(name: str, cache_dir: str = None, extra_args=None):
    """Load a model from a local checkpoint or a HuggingFace repository.

    Supported formats for `name`:

    1. **`.pt` file** — path to a specific checkpoint file.
       A `config.json` must live in the same directory.

        ```python
        model = load_pretrained('my_run/weights_epoch_10.pt')
        ```

    2. **Folder** — path to a directory containing exactly one `.pt` file
       and a `config.json`.

        ```python
        model = load_pretrained('my_run/')
        ```

    3. **HuggingFace repo** (`<user>/<repo>`) — loaded from the local cache
       if already present, otherwise fetched from HF.

        ```python
        model = load_pretrained('nice-user/my-worldmodel')
        ```

    All local paths are resolved relative to `<cache_dir>/checkpoints/`.
    """
    from hydra.utils import instantiate

    cache_dir = get_cache_dir(cache_dir, sub_folder='checkpoints')
    ensure_dir_exists(cache_dir)
    checkpoint_path, config = _resolve(name, cache_dir)
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # assume keys with the dotted notation
    if extra_args is not None:
        for key, value in extra_args.items():
            parts = key.split('.')
            d = config
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value

    model = instantiate(config)
    state_dict = _vit_layout_compat(state_dict, model)
    model.load_state_dict(state_dict)
    return model


# HF renamed the ViT internals in transformers 5.x:
#   encoder.encoder.layer.N.attention.attention.{query,key,value}
#     -> encoder.layers.N.attention.{q,k,v}_proj
#   encoder.encoder.layer.N.attention.output.dense
#     -> encoder.layers.N.attention.o_proj
#   encoder.encoder.layer.N.{intermediate,output}.dense
#     -> encoder.layers.N.mlp.{fc1,fc2}
# A checkpoint saved under one major version therefore cannot be loaded into a
# model built under the other -- every encoder key is missing/unexpected. This
# bit the TwoRoom archive: the LeWM/PLDM checkpoints were written under 4.57.6
# and refused to instantiate under 5.12.1. Remap on the fly so a stored
# checkpoint is portable across the boundary in either direction. DINOv2-backed
# models are unaffected (that layout did not change) and fall through untouched.
_VIT_OLD_TO_NEW = [
    (r'^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.query\.',
     r'encoder.layers.\1.attention.q_proj.'),
    (r'^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.key\.',
     r'encoder.layers.\1.attention.k_proj.'),
    (r'^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.value\.',
     r'encoder.layers.\1.attention.v_proj.'),
    (r'^encoder\.encoder\.layer\.(\d+)\.attention\.output\.dense\.',
     r'encoder.layers.\1.attention.o_proj.'),
    (r'^encoder\.encoder\.layer\.(\d+)\.intermediate\.dense\.',
     r'encoder.layers.\1.mlp.fc1.'),
    (r'^encoder\.encoder\.layer\.(\d+)\.output\.dense\.',
     r'encoder.layers.\1.mlp.fc2.'),
    (r'^encoder\.encoder\.layer\.(\d+)\.layernorm_(before|after)\.',
     r'encoder.layers.\1.layernorm_\2.'),
]


def _vit_layout_compat(state_dict: dict, model) -> dict:
    """Remap ViT encoder keys when the checkpoint and model disagree on layout."""
    import re

    want = set(model.state_dict().keys())
    have = set(state_dict.keys())
    if not (have - want):  # already compatible
        return state_dict

    def remap(sd, rules):
        out = {}
        for k, v in sd.items():
            for pat, rep in rules:
                k2, n = re.subn(pat, rep, k)
                if n:
                    k = k2
                    break
            out[k] = v
        return out

    cand = remap(state_dict, _VIT_OLD_TO_NEW)
    if not (set(cand.keys()) - want):
        logging.info('[vit-compat] remapped ViT keys old->new (transformers '
                     f'{__import__("transformers").__version__})')
        return cand
    # new->old: invert by swapping the two sides of each rule
    back = [
        (r'^encoder\.layers\.(\d+)\.attention\.q_proj\.',
         r'encoder.encoder.layer.\1.attention.attention.query.'),
        (r'^encoder\.layers\.(\d+)\.attention\.k_proj\.',
         r'encoder.encoder.layer.\1.attention.attention.key.'),
        (r'^encoder\.layers\.(\d+)\.attention\.v_proj\.',
         r'encoder.encoder.layer.\1.attention.attention.value.'),
        (r'^encoder\.layers\.(\d+)\.attention\.o_proj\.',
         r'encoder.encoder.layer.\1.attention.output.dense.'),
        (r'^encoder\.layers\.(\d+)\.mlp\.fc1\.',
         r'encoder.encoder.layer.\1.intermediate.dense.'),
        (r'^encoder\.layers\.(\d+)\.mlp\.fc2\.',
         r'encoder.encoder.layer.\1.output.dense.'),
        (r'^encoder\.layers\.(\d+)\.layernorm_(before|after)\.',
         r'encoder.encoder.layer.\1.layernorm_\2.'),
    ]
    cand = remap(state_dict, back)
    if not (set(cand.keys()) - want):
        logging.info('[vit-compat] remapped ViT keys new->old '
                     f'(transformers {__import__("transformers").__version__})')
        return cand
    return state_dict  # let load_state_dict raise with its own diagnostic


def _resolve(name: str, cache_dir: Path) -> tuple[Path, dict]:
    """Return ``(checkpoint_path, config_dict)`` for *name*.

    Resolution order:
      1. ``<cache_dir>/<name>``  as a ``.pt`` file
      2. ``<cache_dir>/<name>``  as a folder
      3. HuggingFace repo (cached locally under ``<cache_dir>/<user>/<repo>/``)
    """
    local = cache_dir / name

    # format 1: explicit .pt file
    if local.suffix == '.pt':
        if not local.exists():
            raise FileNotFoundError(f'Checkpoint not found: {local}')
        return local, _load_config(local.parent)

    # format 2: folder containing a .pt and config.json
    # (skip if it has no .pt — likely a sibling output dir, e.g. eval videos —
    # and fall through to HF resolution when name looks like a repo id)
    if local.is_dir() and list(local.glob('*.pt')):
        return _resolve_folder(local)

    # format 3: HuggingFace repo (<user>/<repo>)
    if '/' in name:
        return _resolve_hf(name, cache_dir)

    raise ValueError(
        f"Cannot resolve '{name}': not a .pt file, a folder, or a HF repo id."
    )


def _resolve_folder(folder: Path) -> tuple[Path, dict]:
    """Load from a folder containing one ``.pt`` file and a ``config.json``."""
    pt_files = list(folder.glob('*.pt'))
    if not pt_files:
        raise FileNotFoundError(f'No .pt file found in {folder}')
    if len(pt_files) > 1:
        raise ValueError(
            f'Ambiguous checkpoint: multiple .pt files in {folder}. '
            'Specify the file directly.'
        )
    logging.info(f'Loading checkpoint from folder {folder}...')
    return pt_files[0], _load_config(folder)


def _resolve_hf(repo_id: str, cache_dir: Path) -> tuple[Path, dict]:
    """Resolve a HuggingFace repo id, using a local cache when available.

    Local layout: ``<cache_dir>/models--<user>--<repo>/``
    """
    local_dir = cache_dir / f'models--{repo_id.replace("/", "--")}'

    if local_dir.is_dir():
        logging.info(f'Loading {repo_id} from local cache...')
        return _resolve_folder(local_dir)

    logging.info(f'Downloading {repo_id} from HuggingFace...')
    local_dir.mkdir(parents=True, exist_ok=True)
    for filename in ('config.json', 'weights.pt'):
        url = f'{HF_BASE_URL}/{repo_id}/resolve/main/{filename}'
        dest = local_dir / filename
        logging.info(f'Fetching {url}')
        _download(url, dest)

    return _resolve_folder(local_dir)


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a tqdm progress bar."""
    response = urllib.request.urlopen(url)
    total = int(response.headers.get('Content-Length', 0)) or None
    with (
        open(dest, 'wb') as f,
        tqdm(total=total, unit='B', unit_scale=True, desc=dest.name) as bar,
    ):
        while chunk := response.read(8192):
            f.write(chunk)
            bar.update(len(chunk))


def _load_config(folder: Path) -> dict:
    config_path = folder / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'config.json not found in {folder}')
    with open(config_path) as f:
        return json.load(f)


__all__ = ['load_pretrained', 'save_pretrained']
