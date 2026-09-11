"""Reproducible presets and whole experiment entry, always under q2."""
from pathlib import Path
from dataclasses import asdict,replace
import argparse,json
from q2_config import Config,HERE,load_config
from q2_archive import save_json
from q2_backtest import execute,STRATEGIES
from q2_export import export
from q2_stress import run_stress

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['configure','run','export','stress'])
    p.add_argument('--config',default=str(HERE/'config_formal_v2.json'))
    p.add_argument('--output',default=str(HERE/'result/formal_v2'))
    p.add_argument('--end',default='2025-12-31')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.mode=='configure':
        cfg=replace(Config(),reference_backend='highspy',accelerated_paths=True,parallel_workers=6,
                    bounded_seed=True,seed_time_limit_seconds=5.,seed_mip_gap=1e-4,
                    optimizer_version='coordinate-best-improvement-bounded-seed-v2.0')
        save_json(HERE/'config_formal_v2.json',asdict(cfg))
        return
    cfg=load_config(args.config)
    root=Path(args.output)
    if args.mode=='run':
        execute(cfg,root,args.end,args.resume)
    elif args.mode=='export':
        for strategy in STRATEGIES:
            export(root,root/'exports'/strategy,cfg,strategy)
        selected=root.parent/'selected_days'
        target=export(root,selected,cfg)
        if target.name=='result2.xlsx':
            target.replace(root.parent/'result2.xlsx')
    elif args.mode=='stress':
        run_stress(root,root.parent/(root.name+'_stress'),cfg)

if __name__=='__main__':
    main()
