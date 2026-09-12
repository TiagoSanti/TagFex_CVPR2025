import copy
import json
from pathlib import Path
import pytest
from utils.research_identity import scientific_hash, guard_architecture_groups
from utils.provenance import canonical_hash
from utils.experiment_paths import experiment_log_dir, completed_run


def config(root):
    return dict(log_dir=str(root/'logs'), dataset_name='cub200', scenario='cil 100-20',
                class_order=list(range(200)),seed=1993, backbone_configs={'name':'resnet18'}, init_epochs=200)


def manifest(directory, cfg, status='completed'):
    p=directory/'provenance';p.mkdir(parents=True,exist_ok=True)
    (p/'run-test.json').write_text(json.dumps(dict(kind='experiment_run',status=status,
        configuration=dict(effective=cfg,effective_hash=canonical_hash(cfg)))))


def test_effective_default_and_explicit_stem(tmp_path):
    a=config(tmp_path);b=copy.deepcopy(a)
    b['backbone_configs']['params']={'dataset_name':'cub200','small_base':False}
    assert scientific_hash(a)==scientific_hash(b)
    b['backbone_configs']['params']['small_base']=True
    assert scientific_hash(a)!=scientific_hash(b)
    assert experiment_log_dir(a)==experiment_log_dir(b)
    assert a['backbone_configs']=={'name':'resnet18'}


@pytest.mark.parametrize('change',[{'study_max_epochs':1},{'seed':1994},{'backbone_configs':{'name':'resnet18','params':{'small_base':True}}}])
def test_same_name_and_full_gist_cannot_prove_equivalence(tmp_path,change):
    desired=config(tmp_path);recorded={**desired,**change};directory=experiment_log_dir(desired)
    directory.mkdir(parents=True);(directory/'exp_gistlog.log').write_text('avg_nme1: 70\n'*6)
    assert completed_run(desired) is None
    manifest(directory,recorded)
    assert completed_run(desired) is None
    manifest(directory,desired,'running')
    assert completed_run(desired) is None
    manifest(directory,desired)
    assert completed_run(desired)==directory


def test_expanded_backbone_override_is_part_of_identity(tmp_path):
    a=config(tmp_path);a['network_configs']={'new_backbone_configs':{'name':'resnet18'}}
    b=copy.deepcopy(a);b['network_configs']['new_backbone_configs']['params']={'small_base':True}
    assert scientific_hash(a)!=scientific_hash(b)


def test_reports_reject_known_architecture_mixture(tmp_path):
    a=config(tmp_path);b=copy.deepcopy(a);b['backbone_configs']['params']={'small_base':True}
    manifest(tmp_path/'a',a);manifest(tmp_path/'b',b)
    with pytest.raises(ValueError,match='Conflicting architectures'):
        guard_architecture_groups([(('cub','baseline'),tmp_path/'a/exp_gistlog.log'),(('cub','baseline'),tmp_path/'b/exp_gistlog.log')])
