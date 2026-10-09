const $ = id => document.getElementById(id);
let market, research, lastRun;
const money = value => new Intl.NumberFormat('en-DE',{style:'currency',currency:'EUR',maximumFractionDigits:0}).format(value);
const params = () => ({power:+$('power').value,duration:+$('duration').value,efficiency:+$('efficiency').value/100,cost:+$('cost').value});

function optimise(prices,p){
  const levels=120, capacity=p.power*p.duration, step=capacity/levels, eta=Math.sqrt(p.efficiency), n=prices.length;
  let value=Array(levels+1).fill(-Infinity); value[0]=0;
  const parents=[];
  for(let t=0;t<n;t++){
    const next=Array(levels+1).fill(-Infinity), parent=Array(levels+1).fill(-1);
    for(let from=0;from<=levels;from++) if(Number.isFinite(value[from])) for(let to=0;to<=levels;to++){
      const delta=(to-from)*step, charge=delta>0?delta/eta:0, discharge=delta<0?-delta*eta:0;
      if(charge>p.power+1e-9||discharge>p.power+1e-9) continue;
      const reward=prices[t]*discharge-prices[t]*charge-p.cost*(charge+discharge);
      if(value[from]+reward>next[to]){next[to]=value[from]+reward;parent[to]=from;}
    }
    value=next; parents.push(parent);
  }
  let state=0; const rows=[];
  for(let t=n-1;t>=0;t--){const from=parents[t][state],delta=(state-from)*step;rows.unshift({charge:delta>0?delta/eta:0,discharge:delta<0?-delta*eta:0,soc:state*step});state=from;}
  return rows;
}
function settle(prices,schedule,p){return schedule.reduce((sum,r,i)=>sum+prices[i]*r.discharge-prices[i]*r.charge-p.cost*(r.charge+r.discharge),0);}
function path(values,x,y,w,h,min,max){return values.map((v,i)=>`${i?'L':'M'} ${x+i*w/(values.length-1)} ${y+h-(v-min)/(max-min||1)*h}`).join(' ');}
function renderChart(actual,forecast,schedule){
  const svg=$('dispatchChart'),x=54,y=25,w=805,h=285,prices=[...actual,...forecast],min=Math.min(...prices),max=Math.max(...prices),maxP=Math.max(...schedule.map(r=>Math.abs(r.discharge-r.charge)),1);
  const grid=[0,.25,.5,.75,1].map(q=>`<line x1="${x}" y1="${y+h*q}" x2="${x+w}" y2="${y+h*q}" stroke="#e3e5df"/><text x="${x-8}" y="${y+h*q+4}" text-anchor="end" font-size="10" fill="#68736c">${Math.round(max-(max-min)*q)}</text>`).join('');
  const bars=schedule.map((r,i)=>{const net=r.discharge-r.charge,bh=Math.abs(net)/maxP*70,bx=x+i*w/24+4,bw=w/24-7,base=y+h;return `<rect x="${bx}" y="${net>=0?base-bh:base}" width="${bw}" height="${bh}" fill="${net>=0?'#1769aa':'#8fb3cf'}" opacity=".82"/>`;}).join('');
  const hours=[0,6,12,18,23].map(i=>`<text x="${x+i*w/23}" y="${y+h+92}" text-anchor="middle" font-size="10" fill="#68736c">${String(i).padStart(2,'0')}:00</text>`).join('');
  svg.innerHTML=`${grid}${bars}<line x1="${x}" y1="${y+h}" x2="${x+w}" y2="${y+h}" stroke="#aeb6bd"/><path d="${path(forecast,x,y,w,h,min,max)}" fill="none" stroke="#9aa5ad" stroke-width="2" stroke-dasharray="5 5"/><path d="${path(actual,x,y,w,h,min,max)}" fill="none" stroke="#182028" stroke-width="3"/>${hours}<text x="12" y="18" font-size="10" fill="#66717c">€/MWh</text><text x="${x}" y="${y+h+65}" font-size="10" fill="#1769aa">Bars: D-1 net dispatch (+ discharge / − charge)</text>`;
}
function run(){
  $('run').disabled=true;$('run').innerHTML='Calculating… <span>·</span>';
  const day=$('day').value,data=market.days[day],p=params(),perfect=optimise(data.realised,p),d1=optimise(data.d1,p),perfectValue=settle(data.realised,perfect,p),d1Value=settle(data.realised,d1,p),capture=perfectValue>0?d1Value/perfectValue:null;
  lastRun={day,data,p,perfect,d1,perfectValue,d1Value,capture};$('empty').hidden=true;$('output').hidden=false;$('resultTitle').textContent=`${p.power} MW / ${p.duration} h battery · ${day}`;
  $('perfectMetric').textContent=money(perfectValue);$('d1Metric').textContent=money(d1Value);$('captureMetric').textContent=capture===null?'n/a':`${(capture*100).toFixed(1)}%`;$('gapMetric').textContent=money(perfectValue-d1Value);
  const spread=Math.max(...data.realised)-Math.min(...data.realised),peak=data.realised.indexOf(Math.max(...data.realised));$('insightText').textContent=`The realised daily spread was €${spread.toFixed(0)}/MWh and the highest price arrived at ${String(peak).padStart(2,'0')}:00. The frozen D-1 schedule ${capture>=.8?'retained most of':'missed a material share of'} the day's available arbitrage value.`;
  renderChart(data.realised,data.d1,d1);updateIntraday();renderInvestment();updateUrl(day,p);$('challenge').href=`https://github.com/Heesunjookr/bess-arbitrage/issues/new?title=${encodeURIComponent('Scenario request: '+day)}&body=${encodeURIComponent(`Please investigate this public lab scenario:\n\n- Delivery day: ${day}\n- Battery: ${p.power} MW / ${p.duration} h\n- Efficiency: ${Math.round(p.efficiency*100)}%\n- Throughput cost: €${p.cost}/MWh\n- D-1 capture: ${(capture*100).toFixed(1)}%`)}`;
  $('run').disabled=false;$('run').innerHTML='Run dispatch <span>→</span>';
}
function switchTab(name){
  document.querySelectorAll('.tab').forEach(button=>{const active=button.dataset.tab===name;button.classList.toggle('active',active);button.setAttribute('aria-selected',active);});
  document.querySelectorAll('.tab-panel').forEach(panel=>panel.hidden=panel.id!==`panel-${name}`);
  if(name==='investment')renderInvestment();
}
function updateIntraday(){
  if(!lastRun)return;
  $('idScenario').textContent=`${lastRun.day} · ${lastRun.p.power} MW / ${lastRun.p.duration} h`;
  $('idBaseline').textContent=`Frozen D-1 schedule: ${money(lastRun.d1Value)} settled value, ${(lastRun.capture*100).toFixed(1)}% of the daily ceiling.`;
}
function irr(cashflows){
  const npv=r=>cashflows.reduce((sum,cash,i)=>sum+cash/((1+r)**i),0);
  let low=-.999,high=5;
  if(npv(low)*npv(high)>0)return null;
  for(let i=0;i<100;i++){const mid=(low+high)/2;if(npv(low)*npv(mid)<=0)high=mid;else low=mid;}
  return (low+high)/2;
}
function renderCashChart(values){
  const svg=$('cashChart'),x=58,y=25,w=805,h=240,max=Math.max(...values,1),bw=w/values.length*.62;
  const bars=values.map((v,i)=>{const bh=Math.max(0,v/max*h),bx=x+(i+.19)*w/values.length;return `<rect x="${bx}" y="${y+h-bh}" width="${bw}" height="${bh}" fill="#1769aa" opacity="${.9-i*.015}"/><text x="${bx+bw/2}" y="${y+h+20}" text-anchor="middle" font-size="10" fill="#66717c">${i+1}</text>`;}).join('');
  svg.innerHTML=`<line x1="${x}" y1="${y+h}" x2="${x+w}" y2="${y+h}" stroke="#9ba39e"/>${bars}<text x="12" y="18" font-size="10" fill="#68736c">EUR/yr</text><text x="${x+w/2}" y="${y+h+48}" text-anchor="middle" font-size="10" fill="#68736c">Project year</text>`;
}
function renderInvestment(){
  if(!research||!lastRun)return;
  const basis=$('revenueBasis').value,base=research.revenue_eur_per_mw_yr[basis],power=lastRun.p.power,duration=lastRun.p.duration,capexRate=+$('capex').value,opex=+$('opex').value*1000*power,degradation=+$('degradation').value/100,stress=+$('stress').value/100,life=+$('life').value;
  const capex=power*duration*1000*capexRate,flows=Array.from({length:life},(_,i)=>base*power*(1-stress)*((1-degradation)**i)-opex),cashflows=[-capex,...flows],projectIrr=irr(cashflows),cumulative=flows.reduce((state,cash,i)=>{if(state.answer!==null)return state;const before=state.total,stateNow=before+cash;return {total:stateNow,answer:stateNow>=capex?i+(capex-before)/cash:null};},{total:0,answer:null}).answer;
  $('caseCapex').textContent=money(capex);$('caseCash').textContent=money(flows[0]);$('casePayback').textContent=cumulative===null?'> project life':`${cumulative.toFixed(1)} yr`;$('caseIrr').textContent=projectIrr===null?'n/a':`${(projectIrr*100).toFixed(1)}%`;
  $('caseBasisNote').textContent=`${research.labels[basis]} · ${money(base)}/MW/yr historical mean`;renderCashChart(flows);
}
function updateUrl(day,p){const q=new URLSearchParams({day,power:p.power,duration:p.duration,efficiency:Math.round(p.efficiency*100),cost:p.cost});history.replaceState(null,'',`${location.pathname}?${q}`);}
function download(){if(!lastRun)return;const r=lastRun,head='hour,realised_price,d1_price,charge_mw,discharge_mw,soc_mwh\n',rows=r.d1.map((s,i)=>[i,r.data.realised[i],r.data.d1[i],s.charge,s.discharge,s.soc].map(v=>typeof v==='number'?v.toFixed(4):v).join(',')).join('\n'),a=document.createElement('a');a.href=URL.createObjectURL(new Blob([head+rows],{type:'text/csv'}));a.download=`bess-dispatch-${r.day}.csv`;a.click();URL.revokeObjectURL(a.href);}
function bind(){
  [['power','powerOut',v=>`${(+v).toFixed(1)} MW`],['duration','durationOut',v=>`${v} h`],['efficiency','efficiencyOut',v=>`${v}%`],['cost','costOut',v=>`€${v}/MWh`]].forEach(([id,out,fmt])=>$(id).addEventListener('input',()=>$(out).textContent=fmt($(id).value)));
  document.querySelectorAll('.preset').forEach(b=>b.onclick=()=>{$('duration').value=b.dataset.duration;$('duration').dispatchEvent(new Event('input'));document.querySelectorAll('.preset').forEach(x=>x.classList.toggle('active',x===b));});
  document.querySelectorAll('.tab').forEach(button=>button.onclick=()=>switchTab(button.dataset.tab));
  [['capex','capexOut',v=>`€${v}/kWh`],['opex','opexOut',v=>`€${v}k/MW/yr`],['degradation','degradationOut',v=>`${(+v).toFixed(1)}%`],['stress','stressOut',v=>`${v}%`],['life','lifeOut',v=>`${v} yr`]].forEach(([id,out,fmt])=>$(id).addEventListener('input',()=>{$(out).textContent=fmt($(id).value);renderInvestment();}));
  $('revenueBasis').onchange=renderInvestment;$('run').onclick=run;$('day').onchange=run;$('download').onclick=download;$('share').onclick=async()=>{await navigator.clipboard.writeText(location.href);$('share').textContent='Link copied';setTimeout(()=>$('share').textContent='Copy scenario link',1600);};
}
async function init(){[market,research]=await Promise.all([fetch('prices.json').then(r=>r.json()),fetch('research.json').then(r=>r.json())]);const days=Object.keys(market.days).reverse();$('day').innerHTML=days.map(d=>`<option>${d}</option>`).join('');bind();const q=new URLSearchParams(location.search);['power','duration','efficiency','cost'].forEach(id=>{if(q.has(id)){$(id).value=q.get(id);$(id).dispatchEvent(new Event('input'));}});if(q.has('day')&&market.days[q.get('day')])$('day').value=q.get('day');run();}
if(typeof module!=='undefined') module.exports={optimise,settle};
if(typeof document!=='undefined') init().catch(()=>{$('resultTitle').textContent='Price data could not be loaded.';$('run').disabled=true;});
