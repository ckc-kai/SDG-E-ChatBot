from __future__ import annotations
import argparse, csv, hashlib, json, math, re, zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

GUIDELINES={
2023:("v3.1","https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?fileid=53475&shareable=true"),
2024:("v3.2","https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?fileid=56226&shareable=true"),
2025:("v4.01","https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?fileid=58132&shareable=true"),
}
NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'; REL='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
ACTUAL_HEADERS=['series_id','row_aligned_v4_metric_number','crosswalk_status','legacy_metric_number','metric_type','metric_name','wind_warning_status_raw','wind_warning_status','legacy_program_name','hftd_tier_raw','hftd_tier','line_type','inspection_type','inspection_method','unit_raw','unit_canonical','actual_value_raw','actual_value_canonical','unit_conversion','comments','blank_meaning','utility_id','reporting_year','reporting_quarter','schema_version','source_revision','source_report_quarter','source_file','source_sheet','source_row','source_value_cell','guideline_url']
PROJ_HEADERS=['series_id','row_aligned_v4_metric_number','crosswalk_status','legacy_metric_number','metric_type','metric_name','wind_warning_status_raw','wind_warning_status','legacy_program_name','hftd_tier_raw','hftd_tier','line_type','inspection_type','inspection_method','unit_raw','unit_canonical','projected_value_raw','projected_value_canonical','unit_conversion','comments','blank_meaning','utility_id','projection_target_year','projection_as_of_year','projection_as_of_quarter','schema_version','source_revision','source_file','source_sheet','source_row','source_value_cell','guideline_url']

def clean(v):
    if v is None:return None
    if isinstance(v,str):
        s=v.replace('\xa0',' ').strip(); return s or None
    return v

def num(v):
    v=clean(v)
    if v is None:return None
    if isinstance(v,bool):return int(v)
    if isinstance(v,(int,float)):return v
    s=v.strip().replace(',','').replace('$','')
    x=float(s); return int(x) if x.is_integer() else x

def norm(field,v):
    v=clean(v)
    if not isinstance(v,str):return v
    s=' '.join(v.split())
    if s.casefold() in {'n/a','na','not applicable'}:return None
    if field=='wind' and s=='None':s='Neither'
    if field=='hftd':s=s.replace('Non- HFTD','Non-HFTD')
    if field=='method' and s.casefold()=='other':s='Other'
    return s

def unit(v): return '$1,000' if clean(v)==1000 else clean(v)
def col_letter(i):
    n=i+1;s=''
    while n:n,r=divmod(n-1,26);s=chr(65+r)+s
    return s

def col_idx(s):
    n=0
    for c in s:n=n*26+ord(c)-64
    return n-1

def read_sheet(path,sheet_name='Table 2'):
    with zipfile.ZipFile(path) as z:
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            root=ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in root.findall(f'{{{NS}}}si'):
                shared.append(''.join((t.text or '') for t in si.iter(f'{{{NS}}}t')))
        wb=ET.fromstring(z.read('xl/workbook.xml')); rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        rmap={r.attrib['Id']:r.attrib['Target'] for r in rels}; target=None
        for sh in wb.find(f'{{{NS}}}sheets'):
            if sh.attrib['name']==sheet_name:target=rmap[sh.attrib[f'{{{REL}}}id']];break
        if target is None:raise KeyError(sheet_name)
        sp=target.lstrip('/') if target.startswith('/') else 'xl/'+target.replace('../','')
        root=ET.fromstring(z.read(sp)); dim=root.find(f'{{{NS}}}dimension'); ref=dim.attrib.get('ref','A1')
        m=re.match(r'([A-Z]+)(\d+)',ref.split(':')[-1]); cols=col_idx(m.group(1))+1; rows=int(m.group(2)); out=[[None]*cols for _ in range(rows)]
        sd=root.find(f'{{{NS}}}sheetData')
        for row in sd:
            ri=int(row.attrib['r'])-1
            for c in row:
                m=re.match(r'([A-Z]+)(\d+)',c.attrib['r']); ci=col_idx(m.group(1)); t=c.attrib.get('t')
                if t=='inlineStr':v=''.join((x.text or '') for x in c.iter(f'{{{NS}}}t'))
                else:
                    vn=c.find(f'{{{NS}}}v')
                    if vn is None:v=None
                    elif t=='s':v=shared[int(vn.text)]
                    elif t=='b':v=vn.text=='1'
                    elif t in {'str','e'}:v=vn.text
                    else:
                        try:x=float(vn.text);v=int(x) if x.is_integer() else x
                        except:v=vn.text
                out[ri][ci]=v
        return out

def parse_name(p):
    m=re.search(r'SDGE_(\d{4})_Q([1-4])',p.name,re.I)
    if not m:return None
    rm=re.search(r'(?:_R|_Rev)(\d+)',p.name,re.I)
    return {'path':p,'name':p.name,'year':int(m.group(1)),'quarter':int(m.group(2)),'revision':int(rm.group(1)) if rm else 0}

def select_sources(indir):
    cs=[x for p in indir.glob('*.xlsx') if (x:=parse_name(p))]
    out={}
    for y in (2023,2024,2025):
        for q in ((4,) if y==2023 else (1,2,3,4)):
            ms=[x for x in cs if x['year']==y and x['quarter']==q]
            if not ms:raise FileNotFoundError(f'{y} Q{q}')
            out[(y,q)]=max(ms,key=lambda x:x['revision'])
    return out

def parse_legacy(vals,vcol):
    out=[];mt=None;mn=None
    for zr,row in enumerate(vals[9:],start=9):
        if clean(row[2]):mt=clean(row[2])
        if clean(row[3]):mn=clean(row[3])
        if clean(row[4]) is None:continue
        out.append({'idx':len(out),'row':zr+1,'metric_type':mt,'legacy_num':mn,'metric_name':clean(row[4]),'wind_raw':clean(row[5]),'program':clean(row[6]),'hftd_raw':clean(row[7]),'line_raw':clean(row[8]),'itype_raw':clean(row[9]),'method_raw':clean(row[10]),'value':row[vcol],'unit':row[-3],'comments':clean(row[-2]),'blank':clean(row[-1])})
    return out

def parse_v4(vals):
    exp=['METRIC NUMBER','METRIC TYPE','METRIC NAME','WIND WARNING STATUS','HFTD TIER','LINE TYPE','INSPECTION TYPE','INSPECTION METHOD','UNIT(S)','COMMENTS','BLANK MEANING','UTILITY ID','REPORTING YEAR','REPORTING QUARTER','ACTUAL VALUE']
    if [clean(x) for x in vals[0][:15]]!=exp:raise ValueError('2025 schema mismatch')
    out=[]
    for zr,row in enumerate(vals[1:],start=1):
        if clean(row[2]) is None:continue
        out.append({'idx':len(out),'row':zr+1,'v4num':int(row[0]),'metric_type':clean(row[1]),'metric_name':clean(row[2]),'wind_raw':clean(row[3]),'hftd_raw':clean(row[4]),'line_raw':clean(row[5]),'itype_raw':clean(row[6]),'method_raw':clean(row[7]),'unit':row[8],'comments':clean(row[9]),'blank':clean(row[10]),'utility':clean(row[11]),'year':int(row[12]),'quarter':int(row[13]),'value':row[14]})
    return out

def sem(r):return (norm('type',r['metric_type']),norm('name',r['metric_name']),norm('wind',r['wind_raw']),norm('hftd',r['hftd_raw']),norm('line',r['line_raw']),norm('itype',r['itype_raw']),norm('method',r['method_raw']))
def sid(s):return 'T2-'+hashlib.sha1(json.dumps(s,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()[:14]
def cv(v,ru,cu):
    x=num(v)
    if x is None:return None,None
    if ru=='hours' and cu=='days':return x/24,'divide_by_24_hours_to_days'
    if ru=='$1,000' and cu=='Dollars':return x*1000,'multiply_by_1000_thousands_to_dollars'
    return x,None

def crosswalk(lg,v4):
    out=[]
    for a,b in zip(lg,v4):
        sa,sb=sem(a),sem(b);ua,ub=unit(a['unit']),unit(b['unit'])
        if sa==sb and ua==ub:st,note='equivalent',None
        elif sa==sb and (ua,ub)==('hours','days'):st,note='equivalent_after_unit_conversion','Divide legacy hours by 24.'
        elif sa==sb and (ua,ub)==('$1,000','Dollars'):st,note='equivalent_after_unit_conversion','Multiply legacy values by 1,000.'
        elif [(x,y) for x,y in zip(sa,sb) if x!=y]==[('All (regardless of RFW/HWW status)','Neither')]:st,note='semantic_change_wind_status','Keep legacy All and v4 Neither as separate series.'
        else:raise AssertionError((a['row'],sa,sb,ua,ub))
        out.append({'legacy_num':a['legacy_num'],'v4num':b['v4num'],'status':st,'note':note,'legacy_unit':ua,'v4_unit':ub,'legacy_sem':sa,'v4_sem':sb})
    counts=Counter(x['status'] for x in out)
    assert counts==Counter({'equivalent':788,'equivalent_after_unit_conversion':13,'semantic_change_wind_status':3}),counts
    return out

def legacy_row(r,cw,y,q,src,report_q,vcol):
    ru,cu=unit(r['unit']),cw['v4_unit'];raw=num(r['value']);canon,conv=cv(r['value'],ru,cu);s=sem(r);ver,url=GUIDELINES[y]
    return [sid(s),cw['v4num'],cw['status'],r['legacy_num'],norm('type',r['metric_type']),norm('name',r['metric_name']),r['wind_raw'],norm('wind',r['wind_raw']),r['program'],r['hftd_raw'],norm('hftd',r['hftd_raw']),norm('line',r['line_raw']),norm('itype',r['itype_raw']),norm('method',r['method_raw']),ru,cu,raw,canon,conv,r['comments'],r['blank'],'SDG&E',y,q,ver,src['revision'],report_q,src['name'],'Table 2',r['row'],f'{col_letter(vcol)}{r["row"]}',url]

def v4_row(r,src):
    s=sem(r);ru=unit(r['unit']);raw=num(r['value']);ver,url=GUIDELINES[2025]
    return [sid(s),r['v4num'],'v4_native',None,norm('type',r['metric_type']),norm('name',r['metric_name']),r['wind_raw'],norm('wind',r['wind_raw']),None,r['hftd_raw'],norm('hftd',r['hftd_raw']),norm('line',r['line_raw']),norm('itype',r['itype_raw']),norm('method',r['method_raw']),ru,ru,raw,raw,None,r['comments'],r['blank'],r['utility'],2025,r['quarter'],ver,src['revision'],r['quarter'],src['name'],'Table 2',r['row'],f'O{r["row"]}',url]

def write_csv(path,headers,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:w=csv.writer(f);w.writerow(headers);w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-dir',default='/mnt/data');ap.add_argument('--output-dir',default='/mnt/data/table2_output');args=ap.parse_args()
    indir=Path(args.input_dir);outdir=Path(args.output_dir);outdir.mkdir(exist_ok=True)
    sel=select_sources(indir);loaded={k:read_sheet(v['path']) for k,v in sel.items()}
    l23=parse_legacy(loaded[(2023,4)],23);l24=parse_legacy(loaded[(2024,4)],11);v4=parse_v4(loaded[(2025,4)])
    assert len(l23)==len(l24)==len(v4)==804
    assert [sem(x) for x in l23]==[sem(x) for x in l24]
    for q in (1,2,3,4):
        assert [sem(x) for x in parse_legacy(loaded[(2024,q)],11)]==[sem(x) for x in l24]
        pv=parse_v4(loaded[(2025,q)]);assert [(x['v4num'],sem(x),unit(x['unit'])) for x in pv]==[(x['v4num'],sem(x),unit(x['unit'])) for x in v4]
    cw=crosswalk(l24,v4);actual=[]
    for q,col in zip((1,2,3,4),(23,24,25,26)):
        actual += [legacy_row(r,c,2023,q,sel[(2023,4)],4,col) for r,c in zip(parse_legacy(loaded[(2023,4)],col),cw)]
    for q in (1,2,3,4):actual += [legacy_row(r,c,2024,q,sel[(2024,q)],q,11) for r,c in zip(parse_legacy(loaded[(2024,q)],11),cw)]
    for q in (1,2,3,4):actual += [v4_row(r,sel[(2025,q)]) for r in parse_v4(loaded[(2025,q)])]
    assert len(actual)==9648
    proj=[]
    for sy in (2023,2024):
        vals=loaded[(sy,4)];start=[i for i,x in enumerate(vals[6]) if clean(x)=='Projected'][0];cols=[];i=start
        while i<len(vals[8]) and isinstance(vals[8][i],(int,float)):cols.append((i,int(vals[8][i])));i+=1
        for col,ty in cols:
            for r,c in zip(parse_legacy(vals,col),cw):
                ru,cu=unit(r['unit']),c['v4_unit'];raw=num(r['value']);canon,conv=cv(r['value'],ru,cu);s=sem(r);ver,url=GUIDELINES[sy]
                proj.append([sid(s),c['v4num'],c['status'],r['legacy_num'],norm('type',r['metric_type']),norm('name',r['metric_name']),r['wind_raw'],norm('wind',r['wind_raw']),r['program'],r['hftd_raw'],norm('hftd',r['hftd_raw']),norm('line',r['line_raw']),norm('itype',r['itype_raw']),norm('method',r['method_raw']),ru,cu,raw,canon,conv,r['comments'],r['blank'],'SDG&E',ty,sy,4,ver,sel[(sy,4)]['revision'],sel[(sy,4)]['name'],'Table 2',r['row'],f'{col_letter(col)}{r["row"]}',url])
    ch=['template_row_index','legacy_metric_number','v4_metric_number','crosswalk_status','metric_type','metric_name','legacy_wind_warning_status','v4_wind_warning_status','hftd_tier','line_type','inspection_type','legacy_inspection_method','v4_inspection_method','legacy_unit','v4_unit','crosswalk_note'];cr=[]
    for i,(a,b,c) in enumerate(zip(l24,v4,cw)):cr.append([i,a['legacy_num'],b['v4num'],c['status'],norm('type',a['metric_type']),norm('name',a['metric_name']),norm('wind',a['wind_raw']),norm('wind',b['wind_raw']),norm('hftd',a['hftd_raw']),norm('line',a['line_raw']),norm('itype',a['itype_raw']),norm('method',a['method_raw']),norm('method',b['method_raw']),c['legacy_unit'],c['v4_unit'],c['note']])
    write_csv(outdir/'sdge_table2_2023_2025_unified_actuals.csv',ACTUAL_HEADERS,actual);write_csv(outdir/'sdge_table2_legacy_projections.csv',PROJ_HEADERS,proj);write_csv(outdir/'sdge_table2_metric_crosswalk.csv',ch,cr)
    summary={'actual_rows':len(actual),'projection_rows':len(proj),'crosswalk_counts':dict(Counter(x['status'] for x in cw)),'sources':[sel[k] | {'path':str(sel[k]['path'])} for k in sorted(sel)]}
    (outdir/'validation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
