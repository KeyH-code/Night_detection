"""Agent-led diagnostic review: reproducible human audits and external autosave drafts."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from source_contract import ContractError, NAMES, canonical_hash

DECISIONS = ('correct_detection', 'true_error', 'annotation_suspect', 'uncertain')
LABELS = dict(zip(DECISIONS, ('正确检出/正常对照', '检测错误', '标注/类别疑点', '无法判定')))
CLASS_LABELS = dict(zip(NAMES, ('行人', '骑行者', '汽车', '卡车', 'tram（官方类别）', '三轮车')))


def audit_plan(cases, queue_identity, seed=20261027):
    """Keep the full Agent queue; select 100 human checks without inventing human labels.

    Mandatory 27 on the current queue; 60 decisive judgments stratified by class/decision;
    13 unresolved examples. Random strata use fixed hash ranks, without group substitution.
    This is quality control on the diagnostic queue, not a population accuracy estimate.
    """
    if len({c['case_id'] for c in cases}) != len(cases):
        raise ContractError('Duplicate audit case')
    if any(not c.get('initial_review') for c in cases):
        raise ContractError('Agent review must cover every case before human audit')
    selected, reasons, strata = [], defaultdict(list), []

    def add(c, why):
        cid = c['case_id']
        if cid not in reasons:
            selected.append(cid)
        if why not in reasons[cid]:
            reasons[cid].append(why)

    for c in sorted(cases, key=lambda c: c['case_id']):
        if 'tricycle_all_gt' in c['layers']: add(c, '三轮车全部标注目标')
        if 'tricycle_workpoint_fp' in c['layers']: add(c, '三轮车工作点误报')
        if c['initial_review']['decision'] == 'annotation_suspect': add(c, 'Agent标注疑点')
    mandatory_count = len(selected)

    def rank(c, cell):
        return hashlib.sha256(f'{seed}/{cell}/{c["case_id"]}'.encode('utf-8')).hexdigest()

    pools = {}
    for name in NAMES:
        for decision in ('correct_detection', 'true_error'):
            cell = name+'/'+decision
            pools[cell] = sorted([c for c in cases if c['case_id'] not in reasons and
                                  c['class'] == name and c['initial_review']['decision'] == decision],
                                 key=lambda c: rank(c, cell))
    picks = {cell: min(5, len(pool)) for cell, pool in pools.items()}
    target = min(60, sum(len(p) for p in pools.values()))
    while sum(picks.values()) < target:
        for cell in pools:
            if picks[cell] < len(pools[cell]) and sum(picks.values()) < target:
                picks[cell] += 1
    for cell, pool in pools.items():
        for c in pool[:picks[cell]]: add(c, '明确结论抽查')
        strata.append({'stratum': cell, 'population_after_mandatory': len(pool),
                       'selected': picks[cell], 'case_ids': [c['case_id'] for c in pool[:picks[cell]]]})

    for name, limit in zip(NAMES, (4, 1, 1, 1, 1, 5)):
        cell = name+'/uncertain'
        pool = sorted([c for c in cases if c['case_id'] not in reasons and c['class'] == name and
                       c['initial_review']['decision'] == 'uncertain'], key=lambda c: rank(c, cell))
        for c in pool[:limit]: add(c, '不确定案例抽查')
        strata.append({'stratum': cell, 'population_after_mandatory': len(pool),
                       'selected': min(limit, len(pool)), 'case_ids': [c['case_id'] for c in pool[:limit]]})
    plan = {'mode': 'agent_primary_human_audit', 'seed': seed, 'queue_identity': queue_identity,
            'agent_review_sha256': canonical_hash([{k: c[k] for k in ('case_id', 'initial_review')}
                                                   for c in cases]),
            'case_ids': selected, 'reasons': dict(reasons), 'mandatory_count': mandatory_count,
            'strata': strata,
            'scope': 'Human checks of Agent judgments; unsampled cases remain Agent-only or unresolved.',
            'escalation': 'Any confirmed disagreement pauses the affected class/type conclusion; Agent rechecks '
                          'that pool, then propose 10 additional human cases before drawing conclusions. '
                          'Uncertain does not count as agreement on a definitive judgment.'}
    plan['audit_identity'] = canonical_hash(plan)
    return plan


def audit_summary(cases, plan):
    lookup = {c['case_id']: c for c in cases}
    counts = Counter()
    for cid in plan['case_ids']:
        c = lookup[cid]
        if c.get('status') != 'reviewed':
            counts['pending'] += 1
            continue
        initial, human = c['initial_review']['decision'], c['decision']
        counts['reviewed'] += 1
        if human == 'uncertain':
            counts['human_uncertain'] += 1
            if initial != 'uncertain': counts['definitive_unverified'] += 1
        elif initial == 'uncertain': counts['resolved_uncertain'] += 1
        elif human == initial: counts['agreed'] += 1
        else: counts['disagreed'] += 1
    return {'required': len(plan['case_ids']), **{k: counts[k] for k in
            ('reviewed', 'pending', 'agreed', 'disagreed', 'human_uncertain', 'resolved_uncertain',
             'definitive_unverified')},
            'coverage_complete': counts['pending'] == 0,
            'needs_followup': bool(counts['disagreed'])}


def validate_bundle(bundle, identity, known_ids, allow_pending=False):
    """Validate every entry before mutating a draft or formal case file."""
    for field in ('phase', 'checkpoint_sha256', 'ready_sha256', 'review_identity'):
        expected = 'human_review' if field == 'phase' else identity[field]
        if bundle.get(field) != expected: raise ContractError('Draft identity mismatch: '+field)
    entries = bundle.get('judgments')
    if not isinstance(entries, list): raise ContractError('Draft judgments must be a list')
    seen = set()
    for e in entries:
        cid = e.get('case_id')
        if cid not in known_ids or cid in seen: raise ContractError('Duplicate/unknown draft case')
        seen.add(cid)
        pending_feedback = allow_pending and e.get('decision') is None and bool(e.get('note'))
        if e.get('decision') not in DECISIONS and not pending_feedback:
            raise ContractError('Invalid draft decision')
        if not isinstance(e.get('reviewer'), str) or not e['reviewer'].strip():
            raise ContractError('Draft reviewer is required')
        if not isinstance(e.get('note', ''), str): raise ContractError('Draft note must be text')
        history = e.get('note_history', [])
        if not isinstance(history, list) or any(not isinstance(note, str) for note in history):
            raise ContractError('Draft note history must contain text')
    return entries


def merge_feedback(previous, incoming):
    """Preserve comments independently of decisions, including older conflicting wording.

    Empty/stale cards cannot erase a nonempty comment. Prefix extensions need no duplicate
    history; earlier distinct wording remains minimal review evidence, not UI event logs.
    """
    previous = previous or {}
    allowed = ('case_id', 'decision', 'reviewer', 'note', 'note_history')
    result = {k: v for source in (previous, incoming) for k, v in source.items() if k in allowed}
    if incoming.get('decision') is None and previous.get('decision') in DECISIONS:
        result['decision'] = previous['decision']
    old, new = previous.get('note') or '', incoming.get('note') or ''
    if not new: new = old
    history = list(dict.fromkeys(previous.get('note_history', [])+incoming.get('note_history', [])))
    if old and old != new and not new.startswith(old) and old not in history:
        history.append(old)
    result['note'] = new
    if history: result['note_history'] = history
    return result


class DraftStore:
    """One fixed external file; merge autosaves so a partial page never drops earlier work."""
    def __init__(self, path, identity, cases):
        self.path, self.identity = Path(path), identity
        self.known = {c['case_id'] for c in cases}
        self.answers = {}
        if self.path.exists():
            prior = json.loads(self.path.read_text(encoding='utf-8'))
            self.answers.update({e['case_id']: e for e in validate_bundle(prior, identity, self.known,
                                                                        allow_pending=True)})
        for c in cases:
            if c.get('status') == 'reviewed' or c.get('note'):
                entry = {k: c.get(k, '') for k in ('case_id', 'decision', 'reviewer', 'note')}
                if c.get('note_history'): entry['note_history'] = c['note_history']
                prior = self.answers.get(entry['case_id'])
                if prior and prior.get('decision') in DECISIONS and entry.get('decision') in DECISIONS and prior['decision'] != entry['decision']:
                    raise ContractError('Draft conflicts with imported human decision')
                self.answers[entry['case_id']] = merge_feedback(entry, prior or entry)

    def bundle(self):
        return dict(self.identity, phase='human_review', judgments=list(self.answers.values()))

    def save(self, submitted):
        entries = validate_bundle(submitted, self.identity, self.known, allow_pending=True)
        answers = dict(self.answers)
        for entry in entries:
            answers[entry['case_id']] = merge_feedback(answers.get(entry['case_id']), entry)
        bundle = dict(self.identity, phase='human_review', judgments=list(answers.values()))
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(self.path)
        self.answers = answers
        return len(answers)


PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>S2 多图复验</title>
<style>
body{font:15px sans-serif;margin:12px;color:#222}button,input,select,textarea{font:inherit}button,select{margin:3px;padding:5px}header{position:sticky;top:0;background:white;border-bottom:1px solid #aaa;padding:8px;z-index:2}#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:12px}.card{font-size:14px;border:1px solid #aaa;padding:10px;min-width:0}.card canvas{width:100%;height:180px;object-fit:contain;background:#ddd;cursor:zoom-in}.card p{margin:7px 0}.card textarea{width:98%;height:48px}.muted{font-size:14px;color:#555}.saved{border-color:#287a38}.error{color:#b00020}dialog{max-width:96vw;max-height:95vh}dialog canvas{max-width:100%;max-height:72vh}summary{cursor:pointer}#notice{padding:8px;background:#f4f4f4}h1{font-size:20px;margin:8px 0}
</style>
<h1>Agent 主审 · 人工抽查</h1>
<p>每卡只看黄框，核对 Agent 意见；点图放大。绿框为标注，红框为工作点预测。</p>
<p id="notice">正在读取队列……</p>
<header><label>查看范围 <select id="scope"><option value="audit">本轮抽查</option><option value="all">全部案例（可选）</option><option value="human">已有判断/备注</option></select></label>
<label><input id="pending" type="checkbox">仅未审</label>
<label>审核者 <input id="reviewer" value="用户复验" size="8"></label>
<label>每页 <select id="pageSize"><option>12</option><option>24</option><option>6</option></select></label>
<button id="prev">上一页</button><span id="position"></span><button id="next">下一页</button>
<button id="export">导出判断和全部备注</button><p id="saveStatus" role="status">输入即自动保存；同意、改判或未选结论时的备注全部保留。翻页不会清空。</p></header>
<div id="grid"></div>
<dialog id="detail"><button id="closeDetail">关闭放大</button><label><input type="checkbox" id="full">查看整图</label><label><input type="checkbox" id="overlay" checked>显示参考标注/工作点预测</label><p id="detailText"></p><canvas id="large"></canvas><details><summary>技术证据（可跳过）</summary><pre id="evidence" style="white-space:pre-wrap"></pre></details></dialog>
<script>
const $=id=>document.getElementById(id), labels={correct_detection:'正确检出/正常对照',true_error:'检测错误',annotation_suspect:'标注/类别疑点',uncertain:'无法判定'};
const names=['行人','骑行者','汽车','卡车','tram（官方类别）','三轮车'];
let data,answers={},indices=[],page=0,storage,images=new Map(),chain=Promise.resolve(),problem=false,detailCase,dirty=new Set(),saveTimer;
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function bundle(){return {phase:'human_review',checkpoint_sha256:data.checkpoint_sha256,ready_sha256:data.ready_sha256,review_identity:data.review_identity,judgments:Object.values(answers)}}
function reviewed(id){return Boolean(answers[id]?.decision)}
function progress(){const n=data.audit.case_ids.filter(reviewed).length,notes=Object.values(answers).filter(a=>a.note).length;$('position').textContent=`第 ${page+1}/${Math.max(1,Math.ceil(indices.length/Number($('pageSize').value)))} 页 · 本轮已审 ${n}/${data.audit.case_ids.length} · 有备注 ${notes} 项`}
function mergeFeedback(old={},entry={}){const result={...old,...entry};if(!entry.decision&&old.decision)result.decision=old.decision;const prior=old.note||'',next=entry.note||prior;const history=[...new Set([...(old.note_history||[]),...(entry.note_history||[])])];if(prior&&prior!==next&&!next.startsWith(prior)&&!history.includes(prior))history.push(prior);result.note=next;if(history.length)result.note_history=history;return result}
function backup(){try{localStorage.setItem(storage,JSON.stringify(answers));localStorage.setItem(storage+'_view',JSON.stringify({page,scope:$('scope').value,pending:$('pending').checked,pageSize:$('pageSize').value,reviewer:$('reviewer').value}))}catch(e){$('saveStatus').textContent='浏览器备份失败，正在尝试本机保存；请暂勿关闭页面'}}
function capture(c,card,decision){const old=answers[c.case_id],note=card.querySelector('textarea').value;if(!old&&!note&&!decision)return;const entry=mergeFeedback(old||{},{case_id:c.case_id,decision:decision||old?.decision||null,reviewer:$('reviewer').value.trim()||'用户复验',note});if(JSON.stringify(old)!==JSON.stringify(entry)){answers[c.case_id]=entry;dirty.add(c.case_id);backup()}card.classList.toggle('saved',Boolean(entry.decision));card.querySelector('.result').textContent=entry.decision?'你的结论：'+labels[entry.decision]:'备注已保存，结论尚未选择';progress()}
function collect(){for(const card of document.querySelectorAll('.card')){const c=data.cases.find(c=>c.case_id===card.dataset.id);capture(c,card)}}
function persist(){clearTimeout(saveTimer);backup();if(!dirty.size)return chain;const ids=[...dirty],entries=ids.map(id=>JSON.parse(JSON.stringify(answers[id])));dirty.clear();const value={...bundle(),judgments:entries};chain=chain.catch(()=>{}).then(async()=>{const response=await fetch('/draft',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});if(!response.ok)throw Error(await response.text());await response.json();problem=false;$('saveStatus').textContent='判断与备注已自动保存；未选结论的备注也会保留并导出';$('saveStatus').className='';}).catch(e=>{ids.forEach(id=>dirty.add(id));problem=true;$('saveStatus').textContent='本机保存失败：'+e.message+'。浏览器保留输入，请先导出 JSON；当前页不会自动跳走。';$('saveStatus').className='error';});return chain}
function choose(c,decision,card){capture(c,card,decision);card.querySelector('select').value=decision;persist()}
function task(c){const name=names[data.names.indexOf(c.class)];if(c.object==='gt')return `请看黄框：标注把它记为“${name}”。类别/框是否合理？模型是否正确检出？`;const row=data.rows[c.key],score=row.pred.conf[c.index];return `请看黄框：模型把这里识别为“${name}”（${score.toFixed(4)}，${score>=c.threshold?'达到':'低于'}使用阈值）。它真是这个类别吗？是否框错或重复？`}
async function imageFor(c){if(!images.has(c.key)){images.set(c.key,new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(Error('图片加载失败'));img.src='/image?key='+encodeURIComponent(c.key)}));}return images.get(c.key)}
async function draw(canvas,c,full=false,overlay=false){const img=await imageFor(c),row=data.rows[c.key],box=row[c.object==='gt'?'gt':'pred'].original_boxes[c.index];let x=0,y=0,w=row.width,h=row.height;if(!full){const pad=Math.max(40,box[2]-box[0],box[3]-box[1]);x=Math.max(0,box[0]-pad);y=Math.max(0,box[1]-pad);w=Math.min(row.width-x,box[2]-x+pad);h=Math.min(row.height-y,box[3]-y+pad);}const scale=Math.min(4,(full?1500:800)/w,(full?850:420)/h);canvas.width=Math.max(1,Math.round(w*scale));canvas.height=Math.max(1,Math.round(h*scale));const ctx=canvas.getContext('2d');ctx.drawImage(img,x,y,w,h,0,0,canvas.width,canvas.height);function rect(b,color,text,width){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.strokeRect((b[0]-x)*scale,(b[1]-y)*scale,(b[2]-b[0])*scale,(b[3]-b[1])*scale);ctx.font='14px sans-serif';ctx.fillStyle='#000';const ty=Math.max(17,(b[1]-y)*scale-5),tx=Math.max(0,(b[0]-x)*scale);ctx.fillRect(tx,ty-15,ctx.measureText(text).width+4,18);ctx.fillStyle=color;ctx.fillText(text,tx+2,ty);}
if(overlay){row.gt.original_boxes.forEach((b,i)=>rect(b,'#00cc55',`标注 ${i} ${names[row.gt.cls[i]]}`,2));row.pred.original_boxes.forEach((b,i)=>{if(row.pred.conf[i]>=data.workpoints[data.names[row.pred.cls[i]]].threshold)rect(b,'#ff5555',`预测 ${i} ${names[row.pred.cls[i]]}`,2)})}rect(box,'#ffff00','本项审核对象',4)}
function render(){const size=Number($('pageSize').value);page=Math.min(page,Math.max(0,Math.ceil(indices.length/size)-1));$('grid').replaceChildren();for(const idx of indices.slice(page*size,(page+1)*size)){const c=data.cases[idx],a=answers[c.case_id],initial=c.initial_review;const card=document.createElement('article');card.className='card'+(a?.decision?' saved':'');card.dataset.id=c.case_id;card.innerHTML=`<strong>${esc(task(c))}</strong><p class="muted">${esc(c.case_id)} · ${esc((data.audit.reasons[c.case_id]||['补充查看']).join(' / '))}</p><canvas aria-label="${esc(c.case_id)} 审核图片"></canvas><p class="muted">${esc(data.facts[c.case_id])}</p><p>Agent：<b>${esc(labels[initial.decision])}</b>（${esc({high:'较有把握',medium:'一般把握',low:'把握较低'}[initial.confidence])}）</p><p>${esc(initial.note)}</p><p class="result">${a?.decision?'你的结论：'+esc(labels[a.decision]):a?.note?'备注已保存，结论尚未选择':'尚未复验'}</p><button class="agree">${initial.decision==='uncertain'?'同意保留不确定':'同意 Agent'}</button><button class="uncertain">我看不清</button><select aria-label="修改本项结论"><option value="">不同意／能判断，改为…</option>${Object.entries(labels).map(([v,t])=>`<option value="${v}">${t}</option>`).join('')}</select><textarea aria-label="本项备注" placeholder="任何补充都可写；不选结论也自动保留">${esc(a?.note||'')}</textarea>${a?.note_history?.length?'<details><summary>此前补充（保留原文）</summary><p>'+a.note_history.map(esc).join('</p><p>')+'</p></details>':''}`;$('grid').append(card);const canvas=card.querySelector('canvas');draw(canvas,c,false,true).catch(e=>{card.querySelector('.result').textContent=e.message});canvas.onclick=()=>openDetail(c);card.querySelector('.agree').onclick=()=>choose(c,initial.decision,card);card.querySelector('.uncertain').onclick=()=>choose(c,'uncertain',card);card.querySelector('select').value=a?.decision||'';card.querySelector('select').onchange=e=>{if(e.target.value)choose(c,e.target.value,card)};card.querySelector('textarea').oninput=()=>{capture(c,card);clearTimeout(saveTimer);saveTimer=setTimeout(persist,250)};card.querySelector('textarea').onchange=()=>{capture(c,card);persist()};}progress();$('prev').disabled=page===0;$('next').disabled=(page+1)*size>=indices.length;try{localStorage.setItem(storage+'_view',JSON.stringify({page,scope:$('scope').value,pending:$('pending').checked,pageSize:$('pageSize').value,reviewer:$('reviewer').value}))}catch(e){}}
function select(){const ids=new Set(data.audit.case_ids);indices=data.cases.map((_,i)=>i).filter(i=>($('scope').value==='all'||($('scope').value==='human'?answers[data.cases[i].case_id]:ids.has(data.cases[i].case_id)))&&(!$('pending').checked||!reviewed(data.cases[i].case_id)));if($('scope').value==='audit')indices.sort((a,b)=>data.audit.case_ids.indexOf(data.cases[a].case_id)-data.audit.case_ids.indexOf(data.cases[b].case_id));page=0;render()}
async function move(delta){collect();await persist();if(problem)return;page+=delta;render();window.scrollTo(0,0)}
async function openDetail(c){detailCase=c;$('full').checked=false;$('detailText').textContent=task(c);const trace=data.traces[c.key];$('evidence').textContent=JSON.stringify({case:c,trace:c.object==='gt'?trace?.gt_evidence?.find(e=>e.gt_id===c.index):null},null,2);$('detail').showModal();await draw($('large'),c,false,$('overlay').checked)}
$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);for(const id of ['scope','pending','pageSize'])$(id).onchange=async()=>{collect();await persist();if(!problem)select()};$('closeDetail').onclick=()=>$('detail').close();for(const id of ['full','overlay'])$(id).onchange=()=>draw($('large'),detailCase,$('full').checked,$('overlay').checked);
$('export').onclick=async()=>{collect();await persist();if(!problem){try{const remote=await (await fetch('/draft')).json();for(const entry of remote.judgments)answers[entry.case_id]=mergeFeedback(answers[entry.case_id],entry);backup()}catch(e){$('saveStatus').textContent='本机读取失败，导出浏览器中保留的判断和备注'}}if(!Object.keys(answers).length){$('saveStatus').textContent='没有判断或备注，未导出空文件。';return}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(bundle(),null,2)],{type:'application/json'}));a.download='s2_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
window.addEventListener('pagehide',()=>{if(!data)return;collect();backup();if(dirty.size)navigator.sendBeacon('/draft',new Blob([JSON.stringify({...bundle(),judgments:[...dirty].map(id=>answers[id])})],{type:'application/json'}))});
fetch('/data').then(r=>{if(!r.ok)throw Error('队列加载失败');return r.json()}).then(x=>{data=x;storage='night_s2_'+data.review_identity;let local={};try{local=JSON.parse(localStorage.getItem(storage)||'{}')}catch(e){}const known=new Set(data.cases.map(c=>c.case_id));for(const e of data.draft.judgments)answers[e.case_id]=e;for(const [id,e] of Object.entries(local)){if(!known.has(id))continue;const merged=mergeFeedback(e,answers[id]||e);if(JSON.stringify(merged)!==JSON.stringify(answers[id])){answers[id]=merged;dirty.add(id)}}$('notice').textContent=data.user_assessment?'本轮反馈已收到：可采用 Agent 主审，高置信度意见作为 Agent 证据保留；不确定项可按需人工补充。所有文字独立保存，不要求补齐全量。':`Agent 已审 ${data.cases.length} 项；本轮抽查 ${data.audit.case_ids.length} 项。任何文字均自动保存，不依赖是否选择结论。`;let view;try{view=JSON.parse(localStorage.getItem(storage+'_view'))}catch(e){}if(view){$('scope').value=view.scope;$('pending').checked=view.pending;$('pageSize').value=view.pageSize;if(view.reviewer)$('reviewer').value=view.reviewer}select();if(view){page=view.page;render()}backup();if(dirty.size)persist();}).catch(e=>{$('notice').textContent=e.message+'。请告知 Agent，不要在空页面上审核。';$('notice').className='error'});
</script></html>'''
