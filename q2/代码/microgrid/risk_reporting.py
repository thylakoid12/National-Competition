"""统一风险报告与论文用表，数值来自同一个有版本记录的实验。"""

import csv
from pathlib import Path
from .statistical_experiment import load_run
from .storage import read_json, write_json


def _csv(path, rows):
    if not rows:
        return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _plot_frontier(selection, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    rows=selection['evaluations']; baseline=rows['M2']['cash_cost']
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for column,(key,label) in enumerate(zip(('emergency_cvar','development_stress_cvar'),
                                ('实测紧急费用 CVaR / 元','开发压力紧急费用 CVaR / 元'))):
        for level in (0,1):
            ax=axes[level,column]
            groups={}
            for name,row in rows.items():
                x=100*(row['cash_cost']/baseline-1)
                if level==1 and x>100*selection['cost_premium_limit']+1e-8:
                    continue
                groups.setdefault((round(x,8),round(row[key],6)),[]).append(name)
            for (x,y),names in groups.items():
                chosen=selection['selected'] in names
                color='#16755a' if chosen else '#46628a'
                ax.scatter(x,y,s=70 if chosen else 35,color=color)
                offset={'M0':(5,6),'M1':(4,6),'M2':(-10,-14),'M3':(5,-14)}.get(names[0],(-70,6) if len(names)>1 else (5,6))
                if level==0 and x<4:
                    continue
                ax.annotate(' / '.join(names),(x,y),xytext=offset,textcoords='offset points',fontsize=8)
            ax.axvline(100*selection['cost_premium_limit'],color='#bc6b25',ls='--',lw=1,
                       label=f"费用增幅上限 {selection['cost_premium_limit']:.0%}")
            ax.set(xlabel='相对 M2 的总费用增幅 / %',ylabel=label,
                   title='全部候选' if level==0 else '费用合格候选放大')
            ax.margins(x=.15,y=.2)
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('一月验证：费用与风险折中（选择规则在后续评价前冻结）')
    fig.savefig(output/'费用风险前沿.png',dpi=170)
    plt.close(fig)


def report(out, output=None, plots=True):
    out=Path(out); output=Path(output or out/'report'); output.mkdir(parents=True,exist_ok=True)
    cfg,protocol=load_run(out,verify=False)
    selection=read_json(out/'selection_frozen.json') if (out/'selection_frozen.json').exists() else (
        read_json(out/'validation_comparison.json') if (out/'validation_comparison.json').exists() else None)
    lines=['**问题二：统计预测与风险约束决策结果**','',
        f'来源：`{out.resolve()}`。配置与原始附件、计算代码哈希见 protocol_frozen.json。',
        f'费用增幅容忍上限：{cfg.cost_premium:.0%}，相对于相同预测与控制器下的 M2；CVaR 水平 β={cfg.risk_beta:g}。',
        '日初计划固定，日内只用已揭示观测执行。完整模型是候选，不预设复杂模型必胜。','']
    if selection:
        lines += [f"一月冻结选型：**{selection['selected'] or '没有合格候选'}**。依据 1 月 15—31 日总费用、实测紧急费用 CVaR 和开发压力费用 CVaR；留出压力与二月后结果未用于本次选型。",'']
        lines += ['论文主消融为 **M0—M3 四个模型**。M3-R、M3-S、M3-RS 是额外的风险保护扩展实验，单独列示，不增加主模型编号。本次保存的选型记录确实评估了四个主模型及三个扩展候选；这里不改写该历史记录。','']
    aggregate={}
    for phase,label in (('validation','一月验证'),('evaluation','二月至十二月连续评价')):
        summaries={p.parent.name:read_json(p) for p in sorted((out/phase).glob('*/summary.json'))}
        if not summaries:
            continue
        aggregate[phase]=summaries
        base=summaries.get('M2')
        lines += [f'**{label}：主模型 M0—M3**','',
            '| 模型 | 天数 | 实际费用 / 元 | 相对 M2 | 紧急电量 / kWh | 紧急费用 CVaR / 元 | 开发压力 CVaR / 元 | 风险限额不满足天数 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        flat=[]
        ordered=[n for n in ('M0','M1','M2','M3','M3-R','M3-S','M3-RS') if n in summaries]
        ordered += [n for n in summaries if n not in ordered]
        extended=False
        for name in ordered:
            row=summaries[name]
            if name not in ('M0','M1','M2','M3') and not extended:
                extended=True
                lines += ['',f'**{label}：风险保护扩展实验**','',
                    '| 变体 | 天数 | 实际费用 / 元 | 相对 M2 | 紧急电量 / kWh | 紧急费用 CVaR / 元 | 开发压力 CVaR / 元 | 风险限额不满足天数 |',
                    '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
            premium=row['cash_cost']/base['cash_cost']-1 if base and base['days']==row['days'] else None
            premium_text=f'{premium:.2%}' if premium is not None else '—'
            lines.append(f"| {name} | {row['days']} | {row['cash_cost']:.2f} | {premium_text} | {row['H']:.2f} | {row['emergency_cvar']:.2f} | {row['development_stress_cvar']:.2f} | {row['risk_budget_unmet_days']} |")
            scalar={k:v for k,v in row.items() if isinstance(v,(str,int,float,bool)) or v is None}
            scalar['cost_premium_vs_M2']=premium; flat.append(scalar)
            daily=[]
            for day in sorted((out/phase/name).glob('2025-??-??.json')):
                record=read_json(day)
                r={k:v for k,v in record.items() if isinstance(v,(str,int,float,bool)) or v is None}
                r.update({'predicted_'+k:v for k,v in record.get('forecast_risk',{}).items() if isinstance(v,(int,float))})
                r['optimizer_evaluations']=record['optimizer']['evaluations']
                daily.append(r)
            _csv(output/f'{phase}_{name}_daily.csv',daily)
        _csv(output/f'{phase}_comparison.csv',flat)
        lines.append('')
        if phase=='evaluation' and selection and selection['selected'] in summaries and base:
            selected=summaries[selection['selected']]
            if selected['days']==base['days']:
                premium=selected['cash_cost']/base['cash_cost']-1
                passed=premium<=cfg.cost_premium+1e-9
                lines += [f"冻结模型后续实测费用增幅为 {premium:.2%}，{'满足' if passed else '未满足'} {cfg.cost_premium:.0%} 上限。这是事后核验，一月费用筛选不保证未来也满足。",'']
                lines += ['| 对照指标 | M2 | 冻结模型 | 相对变化 |',
                          '| --- | ---: | ---: | ---: |']
                for key,name in (('cash_cost','总现金费用 / 元'),('H','紧急购电量 / kWh'),
                    ('emergency_cvar','紧急费用 CVaR / 元'),('development_stress_cvar','开发压力 CVaR / 元'),
                    ('peak_emergency_kw','最大紧急购电功率 / kW'),('pv_utilization','光伏消纳率')):
                    delta=selected[key]/base[key]-1 if base[key] else None
                    text=f'{delta:+.2%}' if delta is not None else '—'
                    left=f'{base[key]:.4%}' if key=='pv_utilization' else f'{base[key]:.4f}'
                    right=f'{selected[key]:.4%}' if key=='pv_utilization' else f'{selected[key]:.4f}'
                    lines.append(f'| {name} | {left} | {right} | {text} |')
                lines += ['',f"冻结模型的真实紧急费用超过事前 VaR 的比例为 {selected['var_exceedance_rate']:.2%}，名义参考为 {1-cfg.risk_beta:.0%}；应作为风险校准诊断报告，不能把 CVaR 当作逐日费用上界。",
                    f"其紧急费用 CVaR 的移动日块重采样区间为 {selected['emergency_cvar_bootstrap_interval'][0]:.2f}—{selected['emergency_cvar_bootstrap_interval'][1]:.2f} 元；该区间反映有限样本不确定性。",'']
        if not all(r['complete'] for r in summaries.values()):
            lines += ['本表包含部分日期，尚不能作为完整评价期结果。','']
    if (out/'stress_summary.json').exists():
        stress=read_json(out/'stress_summary.json'); records=[]
        lines += ['**留出压力测试**','',
            '| 模型 | 案例数 | 最坏案例紧急电量 / kWh | 最坏案例额外现金费用 / 元 | 最大物理残差 |',
            '| --- | ---: | ---: | ---: | ---: |']
        for name,group in stress['models'].items():
            r=group['results']
            lines.append(f"| {name} | {len(r)} | {max(x['emergency_kwh'] for x in r):.2f} | {max(x['additional_cash_cost'] for x in r):.2f} | {max(x['max_physical_residual'] for x in r):.3g} |")
            for case in r:
                flat={k:v for k,v in case.items() if isinstance(v,(str,int,float,bool))}
                flat['shock']=case['shock']['name']; records.append(flat)
        _csv(output/'heldout_stress_cases.csv',records)
        lines += ['', '压力档位是人为指定的检验条件，无发生概率假设；多日案例含连续冲击及两天恢复期，各模型从相同初始库存出发，随后衔接自己的库存。第一天受压与配对正常运行使用完全相同的已锁定计划。','']
        if selection and selection['selected'] in stress['models'] and 'M2' in stress['models']:
            chosen=stress['models'][selection['selected']]['results']
            base_cases=stress['models']['M2']['results']
            for key,label in (('emergency_kwh','紧急电量'),('additional_cash_cost','额外现金费用')):
                ratio=sum(r[key] for r in chosen)/sum(r[key] for r in base_cases)-1
                lines.append(f"相同留出案例的{label}合计，{selection['selected']} 相对 M2 为 {ratio:+.2%}。")
            lines += ['这些合计仅描述固定案例集合；案例长度不同且无发生概率，不能当成年期望风险。','']
    lines += ['**解释与边界**','',
        '- M0/M1/M2/M3 分别为点预测、等权场景、条件加权场景、TV 概率鲁棒；R 为紧急费用 CVaR 限额，S 为设计压力保护。',
        '- 风险限额基于点预测净负荷购电费用的固定比例，避免随着优化计划费用变大而自动放宽。',
        '- 事前风险采用共同条件历史场景评估；M0 的历史风险审计只在其计划完成后进行，不参与其搜索。',
        '- 实际现金费用不扣终端库存价值；净评分 Q 与真实账单分开。日末库存及跨日连续性保存在逐日文件。',
        '- 一月验证仅 17 天，残差窗口最多 60 天，尾部样本较少。移动日块重采样区间用于估计敏感性，不构成未来安全保证。',
        '- 目前允许无限紧急购电补齐缺口。压力指标描述费用和应急依赖，不代表停电或上级电网限供时的供电可靠性。',
        '- 外层是预算约束下的多起点局部搜索，不认证全局最优。模型属事后重新设计，不能声称完全未见全年数据的预注册实验。','']
    (output/'模型与风险结果.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(output/'report_data.json',dict(selection=selection,results=aggregate,protocol=protocol))
    if plots and selection:
        _plot_frontier(selection,output)
    return dict(report=str(output/'模型与风险结果.md'),selected=selection['selected'] if selection else None,
                phases=list(aggregate),output=str(output))
