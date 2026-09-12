"""Data-backed Chinese discussion blocks for the existing paper-results HTML."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

DATES = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def num(value, digits=2):
    return f'{value:,.{digits}f}'


def change(after, before):
    return 100 * (after / before - 1)


def signed(value):
    return f'{value:+.2f}%'.replace('-', '−')


def table(headers, rows, caption=None):
    head = ''.join(f'<th scope="col">{html.escape(str(x))}</th>' for x in headers)
    body = ''.join('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in row)+'</tr>' for row in rows)
    title = f'<caption>{html.escape(caption)}</caption>' if caption else ''
    return f'<div class="table-scroll"><table>{title}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def paragraph(text, kind=''):
    return f'<p{f" class={kind}" if kind else ""}>{text}</p>'


def draft(text):
    return '<div class="paper-draft"><h4>可用于论文的表述</h4><blockquote>'+text+'</blockquote></div>'


def make_analyses(source):
    source = Path(source)
    summaries = {m:read(source/'evaluation'/m/'summary.json') for m in ['M0','M1','M2','M3']}
    supplement = read(source/'report'/'论文补充分析.json')
    assert supplement['post_hoc_analysis'] and not supplement['used_for_selection']
    assert all(s['days']==334 and s['complete'] for s in summaries.values())
    b, s = summaries['M2'], summaries['M3']
    result = {}

    mechanisms = {'M0':'点预测','M1':'等权误差场景','M2':'条件加权场景','M3':'条件场景 + TV 鲁棒'}
    main_rows = [[m, mechanisms[m], num(v['cash_cost']/1e4), num(v['H']), num(v['emergency_cvar']),
                  signed(change(v['cash_cost'],b['cash_cost']))] for m,v in summaries.items()]
    incremental = []
    for before,after,mechanism in [('M0','M1','纳入预测误差'),('M1','M2','按条件赋权'),('M2','M3','加入 TV 鲁棒')]:
        x,y=summaries[before],summaries[after]
        incremental.append([before+' → '+after,mechanism,*[signed(change(y[k],x[k])) for k in ['cash_cost','H','emergency_cvar']]])
    v2=read(source/'validation'/'M2'/'summary.json')
    v3=read(source/'validation'/'M3'/'summary.json')
    a = '<h3>消融实验结果分析</h3>'
    a += paragraph('<b>结论：</b>M2 的总费用最低；M3 在小幅增加费用的同时，进一步降低紧急电量与尾部费用。M0—M3 的意义是逐级检验机制，而不是让四个模型分别争取所有指标第一。')
    a += table(['模型','机制','总费用 / 万元','紧急电量 / kWh','日紧急费用 CVaR₀.₉ / 元','费用相对 M2'],main_rows,'2—12 月完整连续评价；CVaR 为真实日紧急费用的经验统计量')
    a += table(['逐级对照','新增机制','总费用变化','紧急电量变化','紧急费用 CVaR 变化'],incremental)
    a += paragraph('<b>M0 → M1：</b>点预测只提供一条供需路径。加入整日联合误差场景后，模型在制定计划时会考虑可能发生的偏差。本次费用下降 7.80%、紧急电量下降 75.38%，说明单一路径决策遗漏的不确定性会带来较大的应急支出。')
    a += paragraph('<b>M1 → M2：</b>在相同误差场景基础上，条件权重区分历史日对当前日的参考价值。费用、紧急电量与 CVaR 均下降，为条件赋权提供增量证据。各模型共用预测器，因此不能把这一步收益说成预测精度提高。')
    a += paragraph('<b>M2 → M3：</b>TV 鲁棒目标允许概率向不利场景移动，促使购电决策对概率估计偏差更谨慎。本次以 0.64% 的费用增量，换取 16.99% 的紧急电量降幅和 20.42% 的尾部费用降幅。这是费用与风险之间的取舍；TV 只调整已有场景的概率，不会生成历史范围之外的新极端曲线。')
    a += '<h4>为什么采用 M3</h4>'
    a += paragraph(f'模型先在 1 月 15—31 日验证：M3 相对 M2 费用增加 {change(v3["cash_cost"],v2["cash_cost"]):.2f}%，紧急费用 CVaR 下降 {-change(v3["emergency_cvar"],v2["emergency_cvar"]):.2f}%，开发压力 CVaR 下降 {-change(v3["development_stress_cvar"],v2["development_stress_cvar"]):.2f}%。在费用增幅不超过 3% 的合格候选中，按两项相对风险比值的较大者选型，随后冻结 M3。2—12 月的 0.64% 是后续核验结果，不是看完后续数据再选型。')
    a += table(['费用分解','M3 相对 M2 / 元'],[
        ['计划购电费增加', '+'+num(s['plan_cost']-b['plan_cost'])],
        ['紧急购电费减少', '−'+num(b['emergency_cost']-s['emergency_cost'])],
        ['总现金费用净增加', '+'+num(s['cash_cost']-b['cash_cost'])]])
    a += paragraph('更多计划支出部分替代了昂贵的紧急支出，净增加约 8.95 万元。这支持更谨慎采购的总体解释，但不意味着 M3 每天、每个时段都多购电。')
    ext=[]
    for m in ['M3-R','M3-S','M3-RS']:
        v=read(source/'validation'/m/'summary.json')
        ext.append([m,num(v['cash_cost']/1e4),signed(change(v['cash_cost'],v2['cash_cost'])),'超过 3%，未入选'])
    a += '<details><summary>风险保护扩展：为什么没有选择更强约束</summary>'
    a += table(['候选','1月验证费用 / 万元','相对 M2','结论'],ext)
    a += paragraph('这些结果说明当前 R/S 保护阈值带来的费用代价较高，不能据此推断所有风险约束都不经济。最终 M3 未启用 CVaR 或压力硬约束。主消融是递进对照，不是组件交互效应的完全分解；外层局部搜索也不认证全局最优。')+'</details>'
    prediction_audit=source/'audit'/'表6与本次预测误差核对.json'
    if prediction_audit.exists():
        prediction=read(prediction_audit)
        names={
            ('负载','previous_day'):'前一日',('负载','previous_week'):'上周同日',
            ('负载','weekday_2'):'最近两个同星期日均值',('负载','weekday_3'):'最近三个同星期日均值',
            ('负载','mean_7'):'最近七日均值',('光伏','mean_1'):'最近一日均值',
            ('光伏','mean_3'):'最近三日均值',('光伏','mean_7'):'最近七日均值',
            ('光伏','mean_14'):'最近十四日均值',('光伏','linear_7'):'最近七日线性外推',
            ('光伏','linear_14'):'最近十四日线性外推'}
        a+='<details id="表6预测误差"><summary>表6是否更新：基础规则误差与修正后误差</summary>'
        a+=paragraph('<b>原表6的11个数值可以保留。</b>本次保存的基础候选规则评分，换算成 kW 并保留三位小数后，与原表全部一致。表6的评价期为 1 月 2—31 日，共 30 天；它比较未加偏差修正的基础统计规则。')
        a+=table(['对象','基础候选规则','原表 MAE / kW','本次 MAE / kW'],[
            [r['target'],names[(r['target'],r['method'])],num(r['old_mae_kw'],3),num(r['new_mae_kw'],3)]
            for r in prediction['old_table6_comparison']])
        a+=paragraph('基础规则选择仍为：负载采用最近三个同星期日均值，光伏采用最近七日均值。本次改进在基础预测之后加入近期有界偏差修正与历史夜间光伏掩码；表6并没有统计这一层的效果。建议将表题明确为“一月基础统计候选规则的预测误差（未加偏差修正）”，另列修正前后对照。')
        correction=[]
        for period in prediction['deployed_forecast_comparison']:
            for target_idx,target in enumerate(['负载','光伏']):
                correction.append([period['period'],target,num(period['base_mae_kw'][target_idx],3),
                                   num(period['corrected_mae_kw'][target_idx],3),signed(period['change_pct'][target_idx])])
        a+=table(['评价期','对象','实际发布基础预测 MAE / kW','修正后 MAE / kW','变化'],correction,
                 '同一天、同一基础预测修正前后配对比较；修正后列与本次保存的预测数组逐日一致')
        a+=paragraph('一月的基础方法按之前已结算的累计误差逐日选择，尚未固定为月底得出的最佳单一规则。因此，一月实际发布基础预测的负载 MAE 为 345.028 kW，不能与“整月都使用最近三个同星期日均值”的 288.054 kW 混为同一口径，也不能将两者差异归因于偏差修正。二月起基础方法冻结，修正前后两列共用相同基础规则。')
        a+=paragraph('2—12 月修正后负载 MAE 下降 9.57%、光伏 MAE 下降约 1.00%；但一月滚动发布的负载误差增加约 0.86%。修正有适应收益，也有阶段差异，不能写成在所有时期均提高准确率。M0—M3 共用修正后的预测，因此模型之间的购电风险差异仍应归因于场景及决策机制，而不能归因于预测器不同。')
        a+='</details>'
    a += draft('在统一预测器、储能执行规则与求解预算下，误差场景和条件权重依次改善了经济性与应急表现。进一步引入 TV 概率鲁棒后，M3 在后续 334 天总费用增加 0.64%，紧急购电量和日紧急费用 CVaR 分别下降 16.99% 和 20.42%，表明该机制在本次数据和计算预算内形成了较合适的费用与风险折中。')
    result['图1']=a

    days={date:{m:read(source/'evaluation'/m/(date+'.json')) for m in ['M2','M3']} for date in DATES}
    arrays={}
    for date in DATES:
        with np.load(source/'evaluation'/'M3'/(date+'.npz')) as saved:
            arrays[date]={k:saved[k] for k in saved.files}
    a='<h3>购电计划分析</h3>'
    a+=paragraph('先比较绿色日前计划与灰色真实净负荷，再看红色紧急购电。计划高于即时净负荷时，差额可能用于充电或成为未使用购电；计划低于净负荷时，可以由储能放电补足。两条曲线不重合，不能直接认定计划错误。')
    a+=table(['日期','M3 日前购电 / kWh','M2 紧急电量 / kWh','M3 紧急电量 / kWh','M3 总费用 / 元'],[
        [d[5:],num(x['M3']['G']),num(x['M2']['H']),num(x['M3']['H']),num(x['M3']['cash_cost'])] for d,x in days.items()])
    june=arrays['2025-06-21']
    a+=paragraph(f'<b>6 月 21 日的中午：</b>12:00—12:10 的实际负载为 {num(june["load"][72])} kWh，光伏为 {num(june["pv"][72])} kWh，日前购电为 {num(june["G"][72])} kWh。该时段光伏已经超过负载，且起始库存达到 {num(june["E"][72])} kWh 上界，因此图中出现中午少购电、库存维持上界的组合。这里是在解释已发生的运行路径，并非模型在日初知道了实际光伏。')
    a+=paragraph('3 月、9 月和 12 月部分时段出现较高的计划购电，同时图3库存快速上升，与计划补能和储能跨时段调节的运行关系一致。购电的尖峰和不规则变化也受十分钟自由度及有限预算局部搜索影响，不能仅凭曲线形状宣称已经获得唯一最优的峰谷套利策略。')
    a+=paragraph('<b>这组指定日没有显示 M3 全面优于 M2。</b>3 月 20 日、9 月 23 日，M3 紧急电量略高；6 月 21 日、12 月 21 日两者均为零。这组图用于解释决策与执行，整体风险改善应由完整 334 天对照支持。','note')
    a+=draft('四个指定日的购电曲线体现了日前采购、即时供需与储能调节的耦合关系。计划购电不需要逐时等于预测或实际净负荷，盈余可用于储能补能，缺口则由放电或紧急购电补齐。指定日存在模型间互有优劣的情况，因此本文结合连续评价而非单日轨迹判断总体性能。')
    result['图2']=a

    a='<h3>储能响应分析</h3>'
    a+=paragraph('绿色和蓝色表示实际库存，橙色虚线表示 M3 控制器采用的参考库存。参考线决定缺电时保留多少电量；它不是要求电池必须追踪的目标曲线，也不是新的物理下界。真正的库存上下界为 1200 与 10800 kWh。')
    a+=table(['日期','M3 日初 / kWh','M3 日末 / kWh','M3 最低 / kWh','M3 最高 / kWh','α'],[
        [d[5:],num(x['M3']['E0']),num(x['M3']['Eend']),num(x['M3']['min_energy']),
         num(arrays[d]['E'].max()),num(x['M3']['alpha'])] for d,x in days.items()])
    a+=paragraph('四天均达到库存上界，说明电池的容量限制确实参与了运行；3 月和 9 月的晚间库存接近下界，剩余可放电空间较小，仍可能需要紧急补购。要评价容量是否足够，应联合查看缺口发生时刻、参考线、充放电功率和已锁定购电量，不能仅看当天是否曾经充满。')
    a+=paragraph('6 月 21 日 α=0，参考库存降到物理下界，但实际库存仍显著高于下界。原因是 α 只是控制器允许放电的保留阈值，实际放电还取决于是否存在供需缺口和充放电功率限制。较大的 α 也不能单独等同于更低风险，必须与同一天的购电计划一起评价。')
    a+=paragraph('四个日期的日初库存并不相同，M2 与 M3 的日初库存也可能不同，因为模型在共同评价初始状态之后，连续沿用自己的真实日末库存。这里反映连续运行结果，不是每天重新分配一个更有利的初始状态。')
    a+=draft('储能轨迹表明，日前购电与反馈放电共同决定库存的日内变化。参考库存通过限制可释放电量参与风险调节，实际库存则受到供需缺口、功率上限和历史状态的共同影响。模型间指定日初始库存差异来自连续运行的状态积累，比较结果据此解释为完整策略的表现。')
    result['图3']=a

    a='<h3>风险评估与校准分析</h3>'
    a+=paragraph('图中的 VaR/CVaR 是每天决策时估计的风险；消融图中的 CVaR 则是评价结束后，从 334 天真实日紧急费用计算的经验统计量。两者的样本与用途不同，不应混为同一条风险曲线。CVaR₀.₉ 可理解为最差 10% 概率质量对应的平均费用，离散样本需要处理分位点处的部分权重。')
    a+=table(['指标','M2','M3','解释'],[
        ['经验日紧急费用 CVaR / 元',num(b['emergency_cvar']),num(s['emergency_cvar']),'M3 下降 20.42%'],
        ['真实费用超过事前 VaR','40 / 334（11.98%）','42 / 334（12.57%）','名义参考 10%，仍需校准'],
        ['发生紧急购电的天数',b['emergency_days'],s['emergency_days'],'应急天数未减少'],
        ['最大日紧急费用 / 元',num(b['worst_daily_emergency_cost']),num(s['worst_daily_emergency_cost']),'单日最大值仅小幅下降'],
        ['最大紧急功率 / kW',num(b['peak_emergency_kw']),num(s['peak_emergency_kw']),'M3 上升 12.41%'],
        ['光伏消纳率',f'{100*b["pv_utilization"]:.4f}%',f'{100*s["pv_utilization"]:.4f}%','下降约 0.36 个百分点']])
    a+=paragraph('M3 的改善主要体现在紧急购电量及高损失日的平均费用，而没有减少发生应急的天数，也没有压低瞬时应急峰值。图中 2 月部分事前风险估计较高、后续仍有实际费用尖峰，提示应进一步检查残差窗口、误差分布随时间变化及有限尾部样本造成的校准偏差。仅凭这张图不能断言存在某一种天气原因。')
    monthly=supplement['monthly_M2_M3']
    nbetter=sum(x['M3']['H']<x['M2']['H'] for x in monthly)
    a+=paragraph(f'分月补充分析中，M3 在 11 个月里的 {nbetter} 个月紧急电量较低，2 月和 5 月有所增加。这说明总体改善存在时间差异，不能解释为每天更新后能力单调提升。')
    a+='<details><summary>查看配对重采样与分月结果</summary>'
    a+=table(['连续日块长度','M3−M2 的 CVaR 差：95% 百分位区间 / 元'],[
        [str(x['block_days'])+' 天',f'[{num(x["cvar_difference_yuan_percentile_95"][0])}, {num(x["cvar_difference_yuan_percentile_95"][1])}]']
        for x in supplement['paired_block_bootstrap']])
    a+=paragraph('各块长均重复 3000 次，两模型使用相同的日期块索引。所检查区间均在零以下，为本样本中改善方向提供补充支持。这是事后描述性分析，没有重新模拟拼接后的电池状态，不能消除季节非平稳、长期状态依赖及只有一年资料的局限，也不是未来安全保证。')
    a+=table(['月份','M2 紧急电量 / kWh','M3 紧急电量 / kWh','费用增幅'],[
        [x['month'],num(x['M2']['H']),num(x['M3']['H']),signed(x['cost_premium_pct'])] for x in monthly])
    a+='</details>'
    a+=draft('M3 的经验尾部费用下降，但事前风险估计仍存在校准偏差，且最大紧急功率有所增加。因此，当前结果支持尾部费用与累计应急依赖的改善，尚不足以支持所有风险维度同步改善。后续应重点检验事前风险覆盖表现，并研究具有明确容量限制的应急保护机制。')
    result['图4']=a

    stress=read(source/'stress_summary.json')['models']
    lookup={m:{(x['shock']['name'],x['anchor']):x for x in stress[m]['results']} for m in ['M2','M3']}
    assert lookup['M2'].keys()==lookup['M3'].keys()
    defs=[
        ('test_load_20','负载 +20%','1 天'),
        ('test_pv_50','光伏 −50%','1 天'),
        ('test_peak_30','17—21 时负载 +30%','1 天'),
        ('test_joint_15_40','负载 +15%、光伏 −40%','1 天'),
        ('test_joint_3days','联合 +15%/−40%','3 天 + 2 天恢复'),
        ('test_low_initial','低库存，负载 +10%、光伏 −30%','3 天 + 2 天恢复'),
    ]
    # Use the protocol order to avoid relying on display labels as case identifiers.
    protocol=read(source/'protocol_frozen.json')
    assert len(protocol['heldout_shocks'])==len(defs)
    defs=[(shock['name'],label,duration) for shock,(_,label,duration) in zip(protocol['heldout_shocks'],defs)]
    groups=[]
    allcases=[]
    for key,label,duration in defs:
        pair=[(lookup['M2'][(key,d)],lookup['M3'][(key,d)]) for d in DATES]
        h2=sum(x['emergency_kwh'] for x,y in pair);h3=sum(y['emergency_kwh'] for x,y in pair)
        c2=sum(x['additional_cash_cost'] for x,y in pair);c3=sum(y['additional_cash_cost'] for x,y in pair)
        improved=sum(y['emergency_kwh'] < x['emergency_kwh']-1e-8 for x,y in pair)
        groups.append([label,duration,num(h2/1e4),num(h3/1e4),signed(change(h3,h2)),
                       f'{improved}/4',signed(change(c3,c2))])
        for d,(x,y) in zip(DATES,pair):
            allcases.append([label,d[5:],num(x['emergency_kwh']),num(y['emergency_kwh']),
                             signed(change(y['emergency_kwh'],x['emergency_kwh'])),
                             num(x['additional_cash_cost']),num(y['additional_cash_cost'])])
    total=lambda m,k:sum(x[k] for x in stress[m]['results'])
    hdelta=change(total('M3','emergency_kwh'),total('M2','emergency_kwh'))
    cdelta=change(total('M3','additional_cash_cost'),total('M2','additional_cash_cost'))
    a='<h3>压力测试结果分析</h3>'
    a+=paragraph(f'<b>总体结果：</b>四个起始日、六类冲击组成 24 个配对案例，M2 与 M3 合计运行 48 个案例。M3 在 20/24 个案例中紧急电量较低，在 23/24 个案例中额外现金费用较低。固定案例集合的紧急电量合计减少 {-hdelta:.2f}%，额外现金费用合计减少 {-cdelta:.2f}%。这体现多数受测条件下的改善，同时保留了退化情况。')
    a+=table(['冲击条件','观察长度','M2 紧急电量 / 万 kWh','M3 紧急电量 / 万 kWh','电量变化','电量改善案例','额外费用变化'],groups,
             '每行汇总同类冲击的四个季节起点；费用变化以各模型相对其配对正常运行的额外费用为基础')
    a+=paragraph('<b>如何读六个面板：</b>单次负载上升、光伏下降和晚峰冲击检验已锁定计划对当天意外缺口的承受能力；连续三日冲击进一步检验库存积累和后续统计更新的影响；低库存案例检验初始缓冲不足时的脆弱性。各面板纵轴独立，三日案例还包含两天恢复期，不能按不同面板的柱高直接判断哪一种冲击“更危险”。')
    a+=paragraph('两类连续冲击的四个季节案例均减少了紧急电量，说明本次测试中，M3 在多日状态衔接下保留了改善方向。但这些合计包含冲击期与恢复期，不能仅凭总电量减少就认定恢复速度更快；恢复速度需要另行定义恢复时间或库存恢复指标。')
    a+='<h4>退化案例也需要解释</h4>'
    worst_h=max(lookup['M2'],key=lambda k:lookup['M3'][k]['emergency_kwh']-lookup['M2'][k]['emergency_kwh'])
    x,y=lookup['M2'][worst_h],lookup['M3'][worst_h]
    a+=paragraph(f'紧急电量增加最多的是 3 月 20 日负载 +20%：M2 为 {num(x["emergency_kwh"])} kWh，M3 为 {num(y["emergency_kwh"])} kWh，增加 {num(y["emergency_kwh"]-x["emergency_kwh"])} kWh（{change(y["emergency_kwh"],x["emergency_kwh"]):.2f}%）。这表明对历史场景概率保持谨慎，并不能保证每一种新增冲击都占优。')
    worse_cost=[k for k in lookup['M2'] if lookup['M3'][k]['additional_cash_cost']>lookup['M2'][k]['additional_cash_cost']+1e-8]
    assert len(worse_cost)==1
    x,y=lookup['M2'][worse_cost[0]],lookup['M3'][worse_cost[0]]
    a+=paragraph(f'额外费用唯一增加的案例是 12 月 21 日晚峰负载 +30%：M2 为 {num(x["additional_cash_cost"])} 元，M3 为 {num(y["additional_cash_cost"])} 元，增加 {change(y["additional_cash_cost"],x["additional_cash_cost"]):.2f}%。额外费用是各模型受压费用减去各自配对正常运行费用，不能与压力运行的总账单混用。')
    a+='<details><summary>查看全部 24 个配对压力案例</summary>'
    a+=table(['冲击','日期','M2 紧急电量 / kWh','M3 紧急电量 / kWh','电量变化','M2 额外费 / 元','M3 额外费 / 元'],allcases)
    a+='</details>'
    a+=paragraph('<b>抗压结论的范围：</b>首日受压和正常运行执行同一条冲击前锁定计划，之后只能根据已发生的历史重新决策。压力曲线是对真实数据施加的人工变换，不是实测极端事件样本；案例没有发生概率，因此合计不是年期望损失。当前紧急购电没有容量上限，模型衡量的是费用和应急依赖方面的抗压表现，不能据此声称停电或电网限供时仍能保证供电。','note')
    a+=draft(f'在四季起点的六类留出冲击下，M3 在多数配对案例中减少了紧急购电需求。固定案例集合的紧急电量和额外现金费用合计分别降低 {-hdelta:.2f}% 和 {-cdelta:.2f}%，两类连续冲击的四个季节案例均呈现电量改善。然而，单日负载及晚峰冲击存在退化，说明当前鲁棒机制具有场景依赖性。本文据此将其抗压效果限定为受测条件下的费用与应急依赖改善，不将其扩大为所有极端运行条件下的可靠供电保证。')
    result['图5']=a
    return result
