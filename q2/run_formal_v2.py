"""Restart with a January-profiled uniform auxiliary time budget."""
from dataclasses import replace,asdict
import sys
from q2_config import Config,HERE
from q2_archive import save_json
from q2_backtest import execute
from common.logging import Tee

def main():
    cfg=replace(Config(),reference_backend='highspy',accelerated_paths=True,parallel_workers=6,
                 bounded_seed=True,seed_time_limit_seconds=5.,seed_mip_gap=1e-4,
                 optimizer_version='coordinate-best-improvement-bounded-seed-v2.0')
    save_json(HERE/'config_formal_v2.json',asdict(cfg))
    root=HERE/'result/formal_v2'
    root.mkdir(parents=True,exist_ok=True)
    with (root/'console.log').open('a',encoding='utf-8') as log:
        stdout,stderr=sys.stdout,sys.stderr
        sys.stdout,sys.stderr=Tee(stdout,log),Tee(stderr,log)
        try:
            execute(cfg,root,resume=(root/'checkpoint.pkl').exists())
        finally:
            sys.stdout,sys.stderr=stdout,stderr

if __name__=='__main__':
    main()
