"""统计风险主线入口；历史 A/B/C 入口保存在 legacy_run.py。"""

import argparse
import json
import os
from pathlib import Path
from microgrid.config import STATISTICAL_RESULT, load_config
from microgrid.models import MODELS, DEFAULT_MODELS


def expand_models(names, output, default=None):
    from microgrid.storage import read_json
    result=[]
    for name in names or default or []:
        if name=='selected':
            result.append(read_json(output/'selection_frozen.json')['selected'])
        elif name=='all':
            result.extend(DEFAULT_MODELS)
        else:
            result.append(name)
    return list(dict.fromkeys(result))


def main():
    parser=argparse.ArgumentParser(description='问题二：统计模型、风险预算、3%费用约束与独立压力验证')
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','calibrate','evaluate','stress','report','plot','export','run-all'):
        p=commands.add_parser(name)
        p.add_argument('--output',type=Path,default=STATISTICAL_RESULT,help='实验根目录，支持断点续跑')
        if name in ('prepare','calibrate','run-all'):
            p.add_argument('--config',type=Path)
            p.add_argument('--attachments',type=Path)
        if name in ('calibrate','evaluate','stress','run-all'):
            p.add_argument('--workers',type=int,choices=range(1,min(8,os.cpu_count() or 1)+1),
                           default=min(6,max(1,(os.cpu_count() or 1)//2)),
                           help='不同模型独立并行，最多8个进程；每个HiGHS实例只用1个线程')
            p.add_argument('--models',nargs='+',choices=['all','selected',*MODELS])
        if name=='evaluate':
            p.add_argument('--max-days',type=int,help='仅用于冒烟检查；部分结果不能导出提交表')
        if name=='stress':
            p.add_argument('--anchors',nargs='+',help='压力测试起始日期，默认四季代表日')
        if name=='report':
            p.add_argument('--destination',type=Path)
            p.add_argument('--no-plots',action='store_true')
        if name=='plot':
            p.add_argument('--models',nargs='+',choices=['all','selected',*MODELS])
            p.add_argument('--destination',type=Path)
        if name=='export':
            p.add_argument('--model',choices=['selected',*MODELS],default='selected')
            p.add_argument('--destination',type=Path)
    args=parser.parse_args()
    from microgrid.statistical_experiment import prepare,calibrate,evaluate,load_run
    output=args.output.resolve()
    if args.command in ('prepare','calibrate','run-all'):
        existing=(output/'protocol_frozen.json').exists()
        cfg=load_config(args.config) if args.config or not existing else load_run(output)[0]
        attachments=args.attachments or (load_run(output)[1]['attachments'] if existing else None)
        prepare(output,cfg,attachments)
    if args.command=='prepare':
        result=dict(output=str(output),prepared=True)
    elif args.command=='calibrate':
        names=expand_models(args.models,output,DEFAULT_MODELS)
        result=calibrate(output,names,args.workers)
    elif args.command=='evaluate':
        if args.max_days is not None and args.max_days < 1:
            parser.error('--max-days must be positive')
        names=expand_models(args.models,output,['selected','M2'])
        result=evaluate(output,names,args.workers,args.max_days)
    elif args.command=='stress':
        from microgrid.stress_testing import run_stress,DEFAULT_ANCHORS
        names=expand_models(args.models,output,['selected','M2'])
        result=run_stress(output,names,args.anchors or DEFAULT_ANCHORS,args.workers)
    elif args.command=='report':
        from microgrid.risk_reporting import report
        result=report(output,args.destination,not args.no_plots)
    elif args.command=='plot':
        from microgrid.statistical_plotting import plot_results
        result=plot_results(output,expand_models(args.models,output,['selected','M2']),args.destination)
    elif args.command=='export':
        from microgrid.statistical_submission import export
        model=expand_models([args.model],output)[0]
        result=export(output,model,args.destination or output/'submission'/model)
    else:
        from microgrid.risk_reporting import report
        from microgrid.stress_testing import run_stress
        names=expand_models(args.models,output,DEFAULT_MODELS)
        choice=calibrate(output,names,args.workers)
        if choice['selected'] is None:
            result=report(output)
        else:
            evaluate(output,names,args.workers)
            run_stress(output,list(dict.fromkeys([choice['selected'],'M2'])),workers=args.workers)
            result=report(output)
    if args.command in ('calibrate','evaluate','stress'):
        result={k:v for k,v in result.items() if k not in ('evaluations','models','comparison')}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
