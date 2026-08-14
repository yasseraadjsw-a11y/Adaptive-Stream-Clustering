from __future__ import annotations
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SEEDS=[7,13,19,23,31,37,41,43,47,53]
expected={
 'covertype':{'n_clusters':7,'window_size':1000,'projection_dim':32,'microcluster_radius':1.5},
 'electricity':{'n_clusters':2,'window_size':1000,'projection_dim':32,'microcluster_radius':1.0},
 'tweeteval':{'n_clusters':3,'window_size':2000,'projection_dim':256,'microcluster_radius':0.85},
 'synthetic_gmm':{'n_clusters':4,'window_size':1000,'projection_dim':32,'microcluster_radius':1.5},
}

def main():
    errors=[]; checks=[]
    for ds,e in expected.items():
        p=ROOT/'configs/datasets'/f'{ds}.json'
        if not p.exists(): errors.append(f'missing config {p.relative_to(ROOT)}'); continue
        c=json.loads(p.read_text())
        for k,v in e.items():
            if c.get(k)!=v: errors.append(f'{ds}: {k}={c.get(k)!r}, expected {v!r}')
        for k,v in [('initial_rank',8),('min_rank',4),('max_rank',32),('decay',.95),('threshold_smoothing',.9),('leverage_regularization',.001),('retained_energy',.95),('projection_sparsity',3),('max_microclusters',200)]:
            if c.get(k)!=v: errors.append(f'{ds}: {k}={c.get(k)!r}, expected {v!r}')
        if c.get('seeds')!=SEEDS: errors.append(f'{ds}: seed list mismatch')
        b=c.get('baseline_settings',{})
        rad=e['microcluster_radius']
        if b.get('clustream',{}).get('radius')!=rad: errors.append(f'{ds}: CluStream radius mismatch')
        den=b.get('denstream',{})
        for k,v in [('epsilon',rad),('beta',.2),('mu',6.0),('lambda',.01),('max_microclusters',200)]:
            if den.get(k)!=v: errors.append(f'{ds}: DenStream {k} mismatch')
        sk=b.get('streamkmpp',{})
        if sk.get('coreset_size')!=500 or sk.get('buffer_size')!=1000: errors.append(f'{ds}: StreamKM++ sizes mismatch')
    ctrl=json.loads((ROOT/'configs/controlled_representation_drift.json').read_text())
    if ctrl['evaluation']['seeds']!=SEEDS: errors.append('controlled seed list mismatch')
    pm=ctrl['proposed_model']
    for k,v in [('projection_dim',32),('window_size',1000),('initial_rank',8),('min_rank',4),('max_rank',32),('basis_update_interval',100),('rank_tolerance',.04),('stable_intervals_before_shrink',4),('uniform_floor',.05),('leverage_sampling_rate',.65),('min_sampling_probability',.15),('max_sampling_probability',1.0),('retained_energy',.95),('decay',.95)]:
        if pm.get(k)!=v: errors.append(f'controlled proposed {k} mismatch')
    tw=ctrl['recent_methods']['twstream']; fra=ctrl['recent_methods']['fra_art']
    if tw.get('k')!=8 or tw.get('lambda')!=.0028 or abs(tw.get('radius_value',0)-10.411281070423867)>1e-12: errors.append('TWStream controlled settings mismatch')
    if fra.get('fractional_order')!=.5 or fra.get('vigilance')!=.8 or fra.get('choice')!=.001 or fra.get('learning_rate')!=1.0: errors.append('FRA-ART settings mismatch')
    if (ROOT/'configs/paper_protocol').exists() or (ROOT/'configs/paper_literal').exists(): errors.append('duplicate protocol config directory exists')
    forbidden=['basis_decay','ASC_PROTOCOL_PROFILE','paper_literal','reported_reproduction']
    scan=list((ROOT/'src').rglob('*.py'))+list((ROOT/'experiments').rglob('*.py'))+list((ROOT/'configs').rglob('*'))
    for p in scan:
        if not p.is_file(): continue
        try:t=p.read_text(encoding='utf-8')
        except: continue
        for term in forbidden:
            if term in t: errors.append(f'forbidden duplicate-protocol term {term!r} in {p.relative_to(ROOT)}')
    # Code-level semantic constants not represented in flat JSONs.
    from sys import path as syspath
    syspath.insert(0,str(ROOT/'src'))
    from asc_stream.paper_protocol import asc_config_from_paper
    cfg=asc_config_from_paper(ROOT,'electricity',8,7)
    for k,v in [('basis_update_interval',100),('rank_tolerance',.04),('stable_intervals_before_shrink',4),('uniform_floor',.05),('leverage_sampling_rate',.65),('min_sampling_probability',.15),('max_sampling_probability',1.0)]:
        if getattr(cfg,k)!=v: errors.append(f'ASCConfig {k} mismatch')
    if errors:
        print(json.dumps({'pass':False,'errors':errors},indent=2)); raise SystemExit(1)
    checks=['single configs/datasets source','primary baseline parameters match study protocol','controlled comparator settings match protocol','no dual basis-decay/profile path','fixed seeds and adaptive-rank constants verified']
    print(json.dumps({'pass':True,'checks':checks},indent=2))
if __name__=='__main__': main()
