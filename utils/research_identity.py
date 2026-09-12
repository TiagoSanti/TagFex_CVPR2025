"""Scientific configuration identity, independent of deployment paths and labels."""
from copy import deepcopy
from utils.argument import get_args
from utils.provenance import canonical_hash, normalize_json

_OPERATIONAL = {
    'command', 'exp_configs', 'exp_name', 'exp_id', 'host_name', 'device',
    'dist_backend', 'dataset_root', 'log_dir', 'ckpt_dir', 'train_beton_path',
    'val_beton_path', 'disable_log_file', 'disable_save_ckpt', 'terminal_only',
    'output_file_prefix', 'save_ckpt_tasks', 'force_no_debug',
}


def scientific_configuration(config):
    value = vars(get_args(['train']))
    value.update(deepcopy(config))
    if value.get('force_no_debug'):
        value['debug'] = False
    value['dataset_name'] = value['dataset_name'].lower()
    value['scenario'] = value['scenario'].lower()
    backbone = value.get('backbone_configs', {})
    if backbone.get('name') == 'resnet18':
        protocol = value['scenario'].split()[-1]
        automatic = protocol == 'joint' or len(set(protocol.split('-'))) == 1
        params = dict(backbone.get('params') or {})
        params.update(dataset_name=value['dataset_name'], small_base=params.get('small_base', automatic))
        backbone['params'] = params
        new = value.get('network_configs', {}).get('new_backbone_configs')
        if new:
            new_params = dict(new.get('params') or {})
            new_params.update(dataset_name=value['dataset_name'], small_base=new_params.get('small_base', params['small_base']))
            new['params'] = new_params
    return normalize_json({k: v for k, v in value.items() if k not in _OPERATIONAL})


def scientific_hash(config):
    return canonical_hash(scientific_configuration(config))


def guard_architecture_groups(groups):
    """Reject known architecture conflicts; legacy missing manifests stay explicit unknowns.

    Each input is (dataset/method group, gistlog path). This does not establish
    equivalence for legacy inputs; report provenance records their missing proof.
    """
    import json
    from pathlib import Path
    known = {}
    for group, log in groups:
        manifests = list((Path(log).parent / 'provenance').glob('run-*.json'))
        if len(manifests) != 1:
            continue
        try:
            manifest = json.loads(manifests[0].read_text())
            recorded = manifest['configuration']['effective']
            if manifest['configuration'].get('effective_hash') != canonical_hash(recorded):
                continue
            normalized = scientific_configuration(recorded)
            if 'backbone_configs' not in normalized:
                continue
            signature = canonical_hash({key: normalized.get(key) for key in ('backbone_configs', 'network_configs')})
        except (OSError, ValueError, KeyError, TypeError):
            continue
        previous = known.setdefault(group, signature)
        if previous != signature:
            raise ValueError(f'Conflicting architectures for {group}; select a homogeneous research edition explicitly.')
