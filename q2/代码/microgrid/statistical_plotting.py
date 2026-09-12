"""直接从已保存结果生成科研图与十分钟分析明细，不调用优化器。"""

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from .statistical_experiment import load_run
from .storage import read_json, read_arrays, write_json, file_hash
from .reporting import daily_results
from .plotting import DEFAULT_DATES, combined_figure, make_figures

COLORS={'M2':'#567696','M3':'#16755a'}


def plot_results(source, models, output=None):
    source=Path(source)
    output=Path(output or source/'figures')
    output.mkdir(parents=True,exist_ok=True)
    cfg,protocol=load_run(source)
    selected=read_json(source/'selection_frozen.json')['selected']
    bank=read_arrays(source/'evaluation_bank.npz')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})
    files=[]
    model_rows={}
    source_hashes={}

    def save(fig, stem):
        stem=output/stem
        stem.parent.mkdir(parents=True,exist_ok=True)
        for ext in ('png','svg'):
            path=stem.with_suffix('.'+ext)
            fig.savefig(path,dpi=300,bbox_inches='tight')
            files.append(str(path.relative_to(output)))
        plt.close(fig)

    for model in dict.fromkeys(models):
        _,directory,rows=daily_results(source,model)
        if not read_json(directory/'complete.json')['complete']:
            raise ValueError(f'{model} 未完成，不能生成完整评价图')
        model_rows[model]=rows
        source_hashes[model]={}
        daily={k:[] for k in ('load','pv','G','H')}
        blocks=[]
        for row in rows:
            date=row['date']; path=directory/(date+'.npz')
            assert file_hash(path)==row['array_sha256'],path
            source_hashes[model][date]=row['array_sha256']
            day=read_arrays(path)
            index=list(bank['dates']).index(date)
            for key in daily:
                daily[key].append(float(day[key].sum()))
            def stamp(t):
                return f'{t//6:02d}:{10*(t%6):02d}'
            block=pd.DataFrame({'model':model,'date':date,'slot':np.arange(1,145),
                'time_start':[stamp(t) for t in range(144)],
                'time_end':[stamp(t) for t in range(1,145)],
                'price_yuan_per_kwh':bank['price'],
                **{key+'_kwh':day[key] for key in ('load','pv','G','H','charge','discharge','U','W')},
                'E_start_kwh':day['E'][:-1],'E_end_kwh':day['E'][1:],
                'reference_end_kwh':day['reference'][1:],
                'forecast_load_kwh':bank['f0'][index,:,0],
                'forecast_pv_kwh':bank['f0'][index,:,1],
                'plan_cost_yuan':bank['price']*day['G'],
                'emergency_cost_yuan':cfg.emergency_multiplier*bank['price']*day['H']})
            blocks.append(block)
            if date in DEFAULT_DATES:
                save(combined_figure(day,f'{model}：{date} 购电与供需'),f'{model}/四个指定日/{date}_购电')
                figs=make_figures(day,date,cfg)
                save(figs.pop('storage'),f'{model}/储能图/{date}')
                for fig in figs.values():
                    plt.close(fig)
        frame=pd.concat(blocks,ignore_index=True)
        assert len(frame)==334*144
        assert abs(frame['H_kwh'].sum()-sum(r['H'] for r in rows))<1e-6
        assert abs(frame['plan_cost_yuan'].sum()+frame['emergency_cost_yuan'].sum()-sum(r['cash_cost'] for r in rows))<1e-6
        csv=output/'绘图数据'/f'{model}_十分钟明细.csv'
        csv.parent.mkdir(exist_ok=True)
        # CSV 是计算结果的文本导出，保留数值原始精度；不依赖任何 Excel 模板。
        frame.to_csv(csv,index=False,encoding='utf-8-sig')
        files.append(str(csv.relative_to(output)))
        daily={k:np.asarray(v) for k,v in daily.items()}
        dates=[datetime.fromisoformat(r['date']) for r in rows]
        save(combined_figure(daily,f'{model}：全年购电与供需总览（2—12 月，334 天）',dates),f'{model}/全年总览')
        risk=pd.DataFrame({'date':[r['date'] for r in rows],
            'actual_emergency_cost':[r['emergency_cost'] for r in rows],
            'predicted_emergency_var':[r['forecast_risk']['emergency_var'] for r in rows],
            'predicted_emergency_cvar':[r['forecast_risk']['emergency_cvar'] for r in rows]})
        risk.to_csv(output/'绘图数据'/f'{model}_事前风险与真实费用.csv',index=False,encoding='utf-8-sig')
        fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
        ax.plot(dates,risk['predicted_emergency_cvar'],color='#bd8542',lw=1,label='事前场景 CVaR 90%')
        ax.plot(dates,risk['predicted_emergency_var'],color='#557ea2',lw=1,label='事前场景 VaR 90%')
        ax.plot(dates,risk['actual_emergency_cost'],color='#9c3434',lw=.9,label='当天真实紧急费用')
        ax.set(title=f'{model}：事前风险与真实紧急费用',ylabel='元',xlabel='2025 年日期',ylim=(0,None),xlim=(dates[0],dates[-1]))
        ax.xaxis.set_major_locator(mdates.MonthLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%m月'))
        ax.grid(alpha=.2);ax.legend(ncol=3,frameon=False,loc='upper center')
        save(fig,f'{model}/事前风险与真实费用')
    if 'M2' in model_rows and selected in model_rows and selected!='M2':
        fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
        for model in ('M2',selected):
            rows=model_rows[model]
            dates=[datetime.fromisoformat(r['date']) for r in rows]
            for ax,key,unit in zip(axes,('cash_cost','H'),('累计费用 / 万元','累计紧急电量 / 万 kWh')):
                ax.plot(dates,np.cumsum([r[key] for r in rows])/10000,color=COLORS.get(model),label=model)
                ax.set(ylabel=unit,xlabel='2025 年日期',xlim=(dates[0],dates[-1]))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%m月'))
                ax.grid(alpha=.2);ax.legend(frameon=False)
        fig.suptitle('冻结方案与统计基准：真实连续评价')
        save(fig,'模型对照/累计费用与紧急购电')
        rows=[]
        for model in ('M2',selected):
            s=read_json(source/'evaluation'/model/'summary.json')
            rows.append(dict(model=model,**{k:s[k] for k in ('cash_cost','H','emergency_cvar',
                'development_stress_cvar','peak_emergency_kw','pv_utilization')}))
        frame=pd.DataFrame(rows).set_index('model')
        labels=['总现金费用','紧急购电量','紧急费用 CVaR','开发压力 CVaR','最大紧急功率','光伏消纳率']
        pct=100*(frame.loc[selected]/frame.loc['M2']-1)
        fig,ax=plt.subplots(figsize=(10,4.3),layout='constrained')
        colors=['#16755a' if (v<=0 if k!='pv_utilization' else v>=0) else '#bd8542' for k,v in pct.items()]
        bars=ax.barh(labels,pct.values,color=colors,height=.6)
        for b,v in zip(bars,pct):
            ax.annotate(f'{v:+.2f}%',(v,b.get_y()+b.get_height()/2),xytext=(5 if v>=0 else -5,0),
                        textcoords='offset points',ha='left' if v>=0 else 'right',va='center')
        ax.axvline(0,color='#555',lw=.8);ax.invert_yaxis();ax.margins(x=.25)
        ax.set(title=f'{selected} 相对 M2 的变化：同时呈现改善与退化',xlabel='相对变化 / %')
        ax.grid(axis='x',alpha=.2)
        save(fig,'模型对照/费用风险指标变化')
        frame.to_csv(output/'绘图数据'/'M2与冻结模型指标.csv',encoding='utf-8-sig')
    if (source/'stress_summary.json').exists():
        stress=read_json(source/'stress_summary.json')['models']
        names=[m for m in ('M2',selected) if m in stress]
        shocks=protocol['heldout_shocks']
        titles=['负载 +20%','光伏 −50%','晚峰 +30%','联合 15%/40%','联合冲击 3 天','低库存冲击 3 天']
        anchors=sorted({r['anchor'] for m in names for r in stress[m]['results']})
        fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
        for ax,shock,title in zip(axes.flat,shocks,titles):
            for i,model in enumerate(names):
                lookup={r['anchor']:r for r in stress[model]['results'] if r['shock']['name']==shock['name']}
                ax.bar(np.arange(len(anchors))+(i-(len(names)-1)/2)*.36,
                    [lookup[d]['emergency_kwh']/1000 for d in anchors],width=.34,
                    label=model,color=COLORS.get(model))
            ax.set(xticks=np.arange(len(anchors)),xticklabels=[d[5:] for d in anchors],
                   title=title,ylabel='案例紧急电量 / 千 kWh',xlabel='冲击起始日')
            ax.grid(axis='y',alpha=.2)
        axes[0,0].legend(frameon=False)
        fig.suptitle('留出压力测试：各面板纵轴独立；三日冲击案例含两天恢复期')
        save(fig,'压力测试/六类冲击对照')
    summaries={p.parent.name:read_json(p) for p in sorted((source/'evaluation').glob('*/summary.json'))}
    for candidates,figure_title,filename in (
            (('M0','M1','M2','M3'),'主模型 M0—M3：完整连续评价；绿色为一月冻结模型','四个主模型消融对照'),
            (('M2','M3','M3-R','M3-S','M3-RS'),'风险保护扩展：与基准 M2、主模型 M3 比较；不增加主模型编号','风险保护扩展对照')):
        if not all(m in summaries and summaries[m].get('complete') for m in candidates):
            continue
        order=list(candidates)
        fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
        for ax,key,title,scale in zip(axes,('cash_cost','emergency_cvar','development_stress_cvar'),
                ('总费用 / 万元','紧急费用 CVaR / 元','开发压力 CVaR / 万元'),(10000,1,10000)):
            ax.barh(order,[summaries[m][key]/scale for m in order],
                color=['#16755a' if m==selected else '#567696' for m in order])
            ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set(xlabel=title)
            if key=='cash_cost':
                ax.axvline(summaries['M2'][key]*(1+cfg.cost_premium)/scale,color='#bc6b25',ls='--',
                    label='后续核验：M2 费用 × 1.03')
                ax.legend(frameon=False,fontsize=8)
        fig.suptitle(figure_title)
        save(fig,'模型对照/'+filename)
    write_json(output/'绘图来源与核验.json',dict(selected=selected,models=list(model_rows),
        days_per_model=334,slots_per_model=334*144,optimization_rerun=False,
        protocol_sha256=file_hash(source/'protocol_frozen.json'),
        source_array_hashes=source_hashes,files=files))
    (output/'绘图说明.md').write_text(
        '# 结果图与绘图数据\n\n'
        '图件从逐日保存结果读取，不重新优化。每张图提供 300 dpi PNG 和可编辑 SVG。\n\n'
        '- 各模型目录：全年总览、四个指定日购电图、独立储能图、事前风险图。\n'
        '- 模型对照：累计费用与紧急购电、同时呈现改善与退化的风险指标变化。\n'
        '- 压力测试：四季起点、六类冲击，面板纵轴独立；仅比较相同案例。\n'
        '- 绘图数据：十分钟 CSV，每个模型 48096 行，可直接用 Excel、Python 或 Origin 打开。\n\n'
        '十分钟 CSV 字段：load/pv 为真实负载/光伏，G 为日前计划，H 为紧急补购，'
        'charge/discharge 为电池交流侧充放电，U 为未用计划额度，W 为弃光，'
        'E_start/E_end 为时段前后库存，reference_end 为时段末参考库存。'
        '后缀 kwh 均为电量；若画功率曲线，流量字段乘以 6 得到 kW，库存不能乘以 6。'
        'forecast_load/pv 是当日日初因果统计预测，price 为元/kWh，cost 为元。'
        '24:00 表示当日日末，下一天 00:00 库存与之衔接。\n\n'
        '购电组合图的红色紧急购电使用右轴，左右轴刻度不同。累计图只含 2—12 月。'
        '压力档位没有发生概率，不可把案例合计作为年期望风险；当前允许无限紧急补购。'
        '预测 CVaR 不是当天费用上界，其覆盖性需用跨日验证衡量。\n',encoding='utf-8')
    return dict(output=str(output),models=list(model_rows),files=len(files),slots_per_model=334*144)
