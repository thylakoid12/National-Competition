"""Q2 semantic adapter, using copied q1 table and Excel export conventions."""
from pathlib import Path
from copy import copy
from datetime import date as Date,datetime
import argparse,json,re
import numpy as np
import pandas as pd
from openpyxl import load_workbook,Workbook
from openpyxl.styles import Alignment,Font
from q2_config import ATTACHMENTS,interval_labels,state_labels,load_config
from q2_data import load_data
from q2_controller import validate
from q2_archive import save_json
from common.tables import build_tables
from common.result_io import write_result

SELECTED=('2025-03-20','2025-06-21','2025-09-23','2025-12-21')

def emergency_intervals(H,price,tolerance=0.):
    labels=state_labels()
    rows=[]
    start=None
    for t in range(145):
        active=t<144 and H[t]>tolerance
        if active and start is None:
            start=t
        if not active and start is not None:
            rows.append((labels[start]+'-'+labels[t],float(H[start:t].sum()),float(5*price[start:t]@H[start:t])))
            start=None
    return rows

def read_records(root,strategy,price,cfg):
    rows=[]
    for path in sorted((Path(root)/'runs'/strategy).glob('*.npz')):
        with np.load(path) as f:
            out={k:f[k].copy() for k in f.files}
        meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        out.update(meta['settlement'])
        validate(out,price,cfg)
        with np.load(Path(root)/'locked_plans'/strategy/path.name) as f:
            np.testing.assert_array_equal(out['G'],f['G'])
        if rows:
            assert abs(rows[-1][1]['E'][-1]-out['E'][0])<cfg.physical_tolerance
        rows.append((path.stem,out))
    return rows

def schedule(out,price):
    return pd.DataFrame(dict(time=interval_labels(),grid_purchase=out['G'],price=price,
               battery_charge=out['charge'],battery_discharge=out['discharge'],
               energy_start=out['E'][:-1],energy_end=out['E'][1:]))

def plot_day(date,o,price,destination,capacity=12000.):
    from common.plotting import plt,set_style
    set_style()
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    x=np.arange(144)/6
    edges=np.arange(145)/6
    axes[0].stairs(o['load'],edges,label='Load')
    axes[0].stairs(o['pv'],edges,label='PV')
    axes[0].stairs(o['G'],edges,label='Locked plan')
    axes[0].set_ylabel('kWh / slot')
    axes[1].plot(np.arange(145)/6,o['E'],label='Actual battery energy')
    axes[1].axhline(1200,color='grey',linestyle=':')
    axes[1].axhline(10800,color='grey',linestyle=':')
    axes[1].set_ylabel('Internal kWh')
    axes[1].secondary_yaxis('right',functions=(lambda e:100*e/capacity,lambda s:s*capacity/100)).set_ylabel('SOC (%)')
    axes[2].bar(x,o['H'],width=1/6,align='edge',label='Emergency energy')
    axes[2].set_ylabel('kWh / slot')
    axes[2].set_xlabel('Hour (interval start)')
    for ax in axes:
        ax.legend(loc='upper left')
        ax.grid(alpha=.2)
    axes[0].set_title(date+' actual operation')
    axes[2].set_xticks(np.arange(0,25,2))
    fig.tight_layout()
    fig.savefig(destination,dpi=180)
    plt.close(fig)

def export(root,output,cfg,strategy=None):
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    strategy=strategy or cfg.formal_strategy
    data=load_data()
    rows=read_records(root,strategy,data.price,cfg)
    if not rows:
        raise ValueError('No formal records to export')
    complete=[d for d,_ in rows]==list(data.dates[31:])
    detail=[]
    for date,o in rows:
        frame=pd.DataFrame({k:o[k] for k in ['G','H','charge','discharge','U','W','load','pv']})
        frame.insert(0,'date',date)
        frame.insert(1,'interval_start',data.interval_start)
        frame.insert(2,'interval_end',data.interval_end)
        frame['price_CNY_per_kWh']=data.price
        frame['energy_start_kWh']=o['E'][:-1]
        frame['energy_end_kWh']=o['E'][1:]
        frame['plan_fee_CNY']=data.price*o['G']
        frame['emergency_fee_CNY']=5*data.price*o['H']
        frame['cash_fee_CNY']=frame['plan_fee_CNY']+frame['emergency_fee_CNY']
        detail.append(frame)
    pd.concat(detail,ignore_index=True).to_csv(output/'slot_details_kWh.csv',index=False,encoding='utf-8-sig')
    book=load_workbook(ATTACHMENTS/'附件5/result2.xlsx')
    expected={'计划购电量','充放电量','紧急购电量'}
    if set(book.sheetnames)!=expected:
        raise ValueError('Unexpected official result2 sheet semantics')
    purchase=book['计划购电量']
    source_headers=[purchase.cell(1,c).value for c in range(2,146)]
    if purchase.max_column!=147 or purchase.cell(1,146).value!='全天购电量' or purchase.cell(1,147).value!='全天购电费':
        raise ValueError('Unexpected result2 purchase headers')
    labels=interval_labels()
    from common.data_io import parse_time_minutes
    parsed=[]
    for label in source_headers:
        left,right=str(label).replace('—','-').replace('–','-').split('-')
        a,b=parse_time_minutes(left),parse_time_minutes(right)
        if a==0 and b>1440:
            a=1440
        if b<a:
            b+=1440
        if b-a!=10:
            raise ValueError('Template interval is not 10 minutes: '+str(label))
        parsed.append(a)
    if parsed not in [list(range(10,1450,10)),list(range(0,1440,10))]:
        raise ValueError('Unrecognized template interval sequence')
    if cfg.template_policy!='correct_copy_headers':
        raise ValueError('Header correction requires an explicit configured policy; standard records remain available')
    # Authorised by the user: correct labels in this OUTPUT COPY only.
    for c,label in enumerate(labels,2):
        purchase.cell(1,c,label)
    date_rows={pd.Timestamp(purchase.cell(r,1).value).date().isoformat():r for r in range(2,336)}
    if set(date_rows)!=set(data.dates[31:]):
        raise ValueError('Template dates do not match formal period')
    battery,emergency=book['充放电量'],book['紧急购电量']
    if [battery.cell(1,c).value for c in range(1,7)]!=['日期','时间段','充电量','放电量','时刻','储电量']:
        raise ValueError('Unexpected battery headers')
    if [emergency.cell(1,c).value for c in range(1,4)]!=['日期','购电时间段','购电量']:
        raise ValueError('Unexpected emergency headers')
    for sheet in [battery,emergency]:
        for merged in list(sheet.merged_cells.ranges):
            sheet.unmerge_cells(str(merged))
        sheet.delete_rows(2,sheet.max_row)
    total_H=total_emergency_cost=0.
    for date,out in rows:
        r=date_rows[date]
        for c,label in enumerate(labels,2):
            t=labels.index(label)
            purchase.cell(r,c,float(out['G'][t]))
        purchase.cell(r,146,float(out['G'].sum()))
        purchase.cell(r,147,float(out['cash_cost']))
        _,table1,table2,blocks,states=build_tables(schedule(out,data.price))
        table1[0]=['时间段','计划购电量 (kWh)','时间段','计划购电量 (kWh)','时间段','计划购电量 (kWh)']
        table2[0]=['时间段','充电量 (kWh)','放电量 (kWh)','时间段','充电量 (kWh)','放电量 (kWh)']
        table1[-1][0]='全天计划购电量 (kWh)'
        table1[-1][3]='全天现金购电费 (元)'
        table2[-1][0]='0:00 储电量 (kWh)'
        table2[-1][3]='24:00 储电量 (kWh)'
        table1[-1][-1]=float(out['cash_cost'])
        stamp=datetime.combine(Date.fromisoformat(date),datetime.min.time())
        for j,(label,(charge,discharge)) in enumerate(blocks.items()):
            battery.append([stamp,label,charge,discharge,'0:00' if j==0 else ('24:00' if j==1 else None),
                            states[j] if j<2 else None])
        intervals=emergency_intervals(out['H'],data.price)
        for label,energy,fee in intervals:
            emergency.append([stamp,label,energy])
            total_H+=energy
            total_emergency_cost+=fee
        if date in SELECTED:
            daybook=Workbook()
            daybook.remove(daybook.active)
            for name,table in [('表1',table1),('表2',table2),('表3',[['紧急购电时间段','电量（kWh）','费用（元）'],*intervals])]:
                sheet=daybook.create_sheet(name)
                for row in table:
                    sheet.append(row)
                sheet.freeze_panes='B2'
                sheet.sheet_view.showGridLines=False
                for cells in sheet.columns:
                    sheet.column_dimensions[cells[0].column_letter].width=24
                for row in sheet:
                    for cell in row:
                        cell.font=Font(name='Microsoft YaHei',bold=cell.row==1)
                        cell.alignment=Alignment(vertical='center',wrap_text=True)
                        if isinstance(cell.value,(float,int)):
                            cell.number_format='0.0000'
                sheet.row_dimensions[1].height=36
            daypath=output/(date+'_tables.xlsx')
            daybook.save(daypath)
            checked=load_workbook(daypath,data_only=True)
            for name,expected_table in [('表1',table1),('表2',table2)]:
                for ri,record in enumerate(expected_table,1):
                    for ci,value in enumerate(record,1):
                        actual=checked[name].cell(ri,ci).value
                        if isinstance(value,(float,int)):
                            assert abs(actual-value)<1e-6
                        else:
                            assert actual==value
            assert abs(sum(r[1] for r in checked['表3'].iter_rows(min_row=2,values_only=True))-out['H'].sum())<1e-6
            plot_day(date,out,data.price,output/(date+'_actual.png'))
    for sheet in book:
        sheet.freeze_panes='B2'
        for cells in sheet.columns:
            sheet.column_dimensions[cells[0].column_letter].width=20
        for cell in sheet[1]:
            cell.font=Font(name='Microsoft YaHei',bold=True)
            cell.alignment=Alignment(wrap_text=True)
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value,datetime):
                    cell.number_format='yyyy-mm-dd'
                elif isinstance(cell.value,(float,int)):
                    cell.number_format='0.0000'
    target=output/('result2.xlsx' if complete else 'result2_DRAFT_partial.xlsx')
    book.save(target)
    # Re-read exact values, typed dates, interval semantics and independent cash totals.
    reread=load_workbook(target,data_only=True)
    sh=reread['计划购电量']
    for date,out in rows:
        r=date_rows[date]
        values=np.array([sh.cell(r,c).value for c in range(2,146)])
        np.testing.assert_allclose(values,out['G'],rtol=0,atol=1e-8)
        assert abs(sh.cell(r,146).value-out['G'].sum())<1e-6
        assert abs(sh.cell(r,147).value-out['cash_cost'])<1e-6
    assert [sh.cell(1,c).value for c in range(2,146)]==labels
    read_H=sum(row[2] for row in reread['紧急购电量'].iter_rows(min_row=2,values_only=True))
    expected_H=sum(o['H'].sum() for _,o in rows)
    assert abs(read_H-expected_H)<1e-6
    assert abs(total_emergency_cost-sum(o['emergency_cost'] for _,o in rows))<1e-5
    batt=list(reread['充放电量'].iter_rows(min_row=2,values_only=True))
    for i,(_,out) in enumerate(rows):
        for j in range(6):
            record=batt[6*i+j]
            assert abs(record[2]-out['charge'][24*j:24*(j+1)].sum())<1e-6
            assert abs(record[3]-out['discharge'][24*j:24*(j+1)].sum())<1e-6
        assert abs(batt[6*i][5]-out['E'][0])<1e-6
        assert abs(batt[6*i+1][5]-out['E'][-1])<1e-6
    save_json(output/'export_validation.json',dict(complete=complete,formal_days=len(rows),strategy=strategy,
              selection_basis=cfg.strategy_basis,units='kWh and CNY',reread_passed=True,
              daily_purchase_fee_semantics='cash: sum(price*G)+sum(5*price*H), user confirmed',
              source_headers=source_headers,corrected_headers=labels,correction_authorized=True,
              emergency_kwh=read_H,emergency_fee=total_emergency_cost,
              selected_dates_available=[date for date,_ in rows if date in SELECTED]))
    print(target,flush=True)
    return target

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--strategy')
    a=p.parse_args()
    export(a.run,a.output,load_config(a.config),a.strategy)

if __name__=='__main__':
    main()
