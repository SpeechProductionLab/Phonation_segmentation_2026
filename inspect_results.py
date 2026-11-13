"""Read-only, loopback-only browser viewer for MAE experiment outputs."""
import argparse
import csv
import io
import json
from pathlib import Path
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf
from evaluate import evaluate, match_intervals
from run import read_annotations
from run_mae import mae_summary


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


class Inspection:
    def __init__(self, results, audio_dir, annotations):
        self.audio_dir = audio_dir.resolve()
        self.reference = read_annotations(annotations)
        self.config = json.loads((results / 'config.json').read_text())
        self.threshold = float(self.config.get('iou_threshold', .25))
        self.predictions = {}
        for r in read_csv(results / 'predictions.csv'):
            key = (r['method'], r['filename'])
            self.predictions.setdefault(key, []).append((float(r['start_seconds']), float(r['end_seconds'])))
        self.records = []
        self.pairs = {}
        pooled = {}
        # Metrics enumerate processed recordings, including zero-prediction cases.
        for row in read_csv(results / 'metrics.csv'):
            method, name = row['method'], row['filename']
            if Path(name).name != name or name in ('.', '..'):
                raise ValueError('Results contain an unsafe audio filename')
            if name not in self.reference:
                raise ValueError(f'Missing annotation for {name}')
            key = (method, name)
            if key in self.pairs:
                raise ValueError(f'Duplicate metrics row: {key}')
            ref, pred = self.reference[name], self.predictions.get(key, [])
            minimum = self.config.get('reference_min_duration_seconds', {}).get(method, 0.)
            ref = [(a, b) for a, b in ref if b - a >= minimum - 1e-9]
            if any(not np.isfinite([a, b]).all() or not 0 <= a < b for a, b in pred):
                raise ValueError(f'Invalid prediction for {name}')
            stats = evaluate(ref, pred, self.threshold)
            self.pairs[key] = (ref, pred, stats)
            pooled.setdefault(method, []).append(stats)
            self.records.append(dict(method=method, filename=name, **mae_summary([stats])))
        if not self.records:
            raise ValueError('No selected recording results to inspect')
        if any(key not in self.pairs for key in self.predictions):
            raise ValueError('Predictions contain recordings absent from metrics.csv')
        self.totals = {m: mae_summary(rows) for m, rows in pooled.items()}
        self.cache = None

    def catalog(self):
        return dict(records=self.records, totals=self.totals, threshold=self.threshold,
                    stage=self.config.get('stage', 'unknown'))

    def audio(self, name):
        if name not in {r['filename'] for r in self.records}:
            raise ValueError('Recording is not part of the loaded experiment')
        if self.cache is None or self.cache[0] != name:
            path = (self.audio_dir / name).resolve()
            if path.parent != self.audio_dir:
                raise ValueError('Audio must be inside the selected audio folder')
            x, sr = sf.read(path, dtype='float64', always_2d=True)
            x = x.mean(axis=1)
            if not len(x) or not np.isfinite(x).all() or sr < 8000:
                raise ValueError('Expected finite, nonempty audio sampled at >= 8000 Hz')
            self.cache = (name, x, sr)
        return self.cache[1:]

    def detail(self, method, name):
        ref, pred, stats = self.pairs[(method, name)]
        x, sr = self.audio(name)
        duration = len(x) / sr
        if any(b > duration + 1e-6 for a, b in ref + pred):
            raise ValueError('An interval exceeds the selected audio duration; check that the audio matches')
        matches = match_intervals(ref, pred, self.threshold)
        matched_ref = {i for i, j, v in matches}
        matched_pred = {j for i, j, v in matches}
        rows = []
        for i, j, v in matches:
            start = (pred[j][0] - ref[i][0]) * 1000
            end = (pred[j][1] - ref[i][1]) * 1000
            rows.append(dict(status='matched', reference=i + 1, prediction=j + 1,
                             reference_start=ref[i][0], reference_end=ref[i][1],
                             predicted_start=pred[j][0], predicted_end=pred[j][1], iou=v,
                             signed_start_ms=start, signed_end_ms=end,
                             absolute_start_ms=abs(start), absolute_end_ms=abs(end)))
        for i, (a, b) in enumerate(ref):
            if i not in matched_ref:
                rows.append(dict(status='missed', reference=i + 1, reference_start=a, reference_end=b))
        for j, (a, b) in enumerate(pred):
            if j not in matched_pred:
                rows.append(dict(status='extra', prediction=j + 1, predicted_start=a, predicted_end=b))
        # Bounded overview; exact visible waveform/energy fetched separately when zooming.
        return dict(filename=name, method=method, duration=duration, sample_rate=sr,
                    reference=ref, predicted=pred, attempts=rows,
                    metrics=mae_summary([stats]), **self.plot(x, sr, 0, duration))

    @staticmethod
    def plot(x, sr, start, end):
        duration = len(x) / sr
        if not np.isfinite([start, end]).all() or not 0 <= start < end <= duration + 1e-6:
            raise ValueError('Invalid plot time range')
        first, last = int(start * sr), min(len(x), max(int(end * sr), int(start * sr) + 1))
        width = max(1, int(np.ceil((last - first) / 1600)))
        lows, highs, ts = [], [], []
        for pos in range(first, last, width):
            chunk = x[pos:min(last, pos + width)]
            lows.append(float(chunk.min()))
            highs.append(float(chunk.max()))
            ts.append(pos / sr)
        # Compute at most ~1600 energy frames in the requested range. Original 30ms
        # window and globally aligned 5ms hop; decimation is only for the display.
        hop, frame = max(1, round(.005 * sr)), max(2, round(.030 * sr))
        starts = np.arange((first // hop) * hop, last, hop)
        starts = starts[::max(1, int(np.ceil(len(starts) / 1600)))]
        energy = [float(20 * np.log10(max(np.sqrt(np.mean(np.pad(x[p:min(len(x), p + frame)],
                  (0, max(0, p + frame - len(x)))) ** 2)), 1e-12))) for p in starts]
        return dict(wave_times=ts, wave_min=lows, wave_max=highs,
                    energy_times=(starts / sr).tolist(), energy_db=energy,
                    amplitude_peak=float(max(np.max(np.abs(x)), 1e-12)))


def make_handler(state, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            parsed = urlparse(self.path)
            if not parsed.path.startswith('/' + token + '/'):
                self.send_error(404)
                return
            route = parsed.path[len(token) + 2:]
            query = parse_qs(parsed.query)
            try:
                if route == '':
                    data, kind = HTML.encode(), 'text/html; charset=utf-8'
                elif route == 'catalog':
                    data, kind = json.dumps(state.catalog()).encode(), 'application/json'
                elif route == 'detail':
                    data = json.dumps(state.detail(query['method'][0], query['name'][0])).encode()
                    kind = 'application/json'
                elif route == 'plot':
                    x, sr = state.audio(query['name'][0])
                    data = json.dumps(state.plot(x, sr, float(query['start'][0]), float(query['end'][0]))).encode()
                    kind = 'application/json'
                elif route == 'audio':
                    x, sr = state.audio(query['name'][0])
                    stream = io.BytesIO()
                    sf.write(stream, x, sr, format='WAV', subtype='FLOAT')
                    data, kind = stream.getvalue(), 'audio/wav'
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                self.wfile.write(data)
            except (ValueError, KeyError, OSError) as error:
                data = json.dumps(dict(error=str(error))).encode()
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
    return Handler


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Phonation results inspector</title>
<style>
body{font:15px system-ui;margin:24px;background:#f6f8fb;color:#172333}h1{font-size:24px}button,select,input{font:inherit;padding:7px;margin:3px}button{cursor:pointer}section{background:white;padding:14px;border:1px solid #dce2eb;border-radius:8px;margin:12px 0}.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}canvas{width:100%;height:300px;border:1px solid #ddd;cursor:crosshair}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:left}th{position:sticky;top:0;background:#eef2f8}tr.selected{background:#e3edff}tbody tr{cursor:pointer} .scroll{max-height:260px;overflow:auto}.muted{color:#526176}.error{color:#ad2020;white-space:pre-wrap}.badge{padding:5px;background:#edf2f8;border-radius:4px}#scroll{width:95%}#attempts{min-width:1100px}
</style><h1>Phonation results inspector</h1>
<p class="muted">Read-only · Reference = manual annotation · Predicted = automatic segmentation · All processing stays on this computer.</p>
<div id="error" role="alert" class="error"></div>
<section><div class="row"><label>Method <select id="method"></select></label><label>Find recording <input id="search" placeholder="Filename"></label><label>Sort <select id="sort"><option value="filename">Filename</option><option value="end_mae_ms">Largest end MAE</option><option value="combined_mae_ms">Largest combined MAE</option><option value="extra_attempts">Most extra attempts</option><option value="missed_attempts">Most missed attempts</option></select></label></div><p id="totals"></p><div class="scroll"><table><thead><tr><th>Recording</th><th>Matched</th><th>Missed</th><th>Extra</th><th>Start MAE ms</th><th>End MAE ms</th><th>Combined MAE ms</th></tr></thead><tbody id="records"></tbody></table></div></section>
<section><div class="row"><button id="previous">Previous</button><button id="next">Next</button><strong id="filename">Select a recording</strong></div><p id="metrics"></p><div class="row"><button id="play">Play / pause</button><button id="selected">Play selected interval</button><button id="zoomIn">Zoom +</button><button id="zoomOut">Zoom −</button><button id="fit">Fit recording</button><span id="clock"></span></div><p class="muted">Click waveform to seek. Click an interval lane or attempt row to select it. Green: reference; orange: predicted. Energy is short-time RMS in dBFS (30 ms window, 5 ms hop), not a method-specific decision score.</p><canvas id="plot" aria-label="Waveform, RMS energy, reference and predicted interval lanes"></canvas><input id="scroll" aria-label="Pan time range" type="range" min="0" max="1000" value="0"><p id="selection" class="muted">No interval selected.</p><p id="cursor" class="muted">Hover over the plot for time, amplitude range, and RMS energy.</p></section>
<section><h2>Attempt comparison</h2><p class="muted">Times in seconds. Errors in milliseconds. Signed error = predicted − reference: positive is late, negative is early. Unmatched intervals have no boundary MAE. Click a row to select its predicted interval (reference for missed attempts).</p><div class="scroll"><table id="attempts"><thead><tr id="attemptHead"></tr></thead><tbody id="attemptRows"></tbody></table></div></section>
<script>
const $=id=>document.getElementById(id), canvas=$('plot'), ctx=canvas.getContext('2d');
let catalog, list=[], detail=null, viewStart=0,viewEnd=1,selection=null, audioCtx=null,buffer=null,source=null,playing=false,playTime=0,startedAt=0,stopAt=null,loadVersion=0,plotVersion=0;
const fields=['status','reference','prediction','reference_start','reference_end','predicted_start','predicted_end','iou','signed_start_ms','signed_end_ms','absolute_start_ms','absolute_end_ms'];
for(const field of fields){const th=document.createElement('th');th.textContent=field.replaceAll('_',' ');$('attemptHead').append(th)}
function fmt(v){return typeof v==='number'?(Number.isInteger(v)?String(v):v.toFixed(2)):(v??'—')}
function err(e){$('error').textContent=e.message||String(e)}
async function api(route){const r=await fetch(route);const data=await r.json();if(!r.ok)throw Error(data.error||'Request failed');return data}
function cells(tr,values){for(const v of values){const td=document.createElement('td');td.textContent=fmt(v);tr.append(td)}}
function showMetrics(m){return `Matched ${m.matched_attempts}/${m.reference_attempts} · Missed ${m.missed_attempts} · Extra ${m.extra_attempts} · Start MAE ${fmt(m.start_mae_ms)} ms · End MAE ${fmt(m.end_mae_ms)} ms · Combined MAE ${fmt(m.combined_mae_ms)} ms`}
function filter(){const method=$('method').value, needle=$('search').value.toLowerCase(), key=$('sort').value;list=catalog.records.filter(r=>r.method===method&&r.filename.toLowerCase().includes(needle));list.sort((a,b)=>key==='filename'?a.filename.localeCompare(b.filename):((typeof b[key]==='number'?b[key]:-1)-(typeof a[key]==='number'?a[key]:-1)||a.filename.localeCompare(b.filename)));$('records').replaceChildren();for(const r of list){const tr=document.createElement('tr');if(detail&&r.filename===detail.filename&&r.method===detail.method)tr.className='selected';cells(tr,[r.filename,r.matched_attempts,r.missed_attempts,r.extra_attempts,r.start_mae_ms,r.end_mae_ms,r.combined_mae_ms]);tr.onclick=()=>load(r);$('records').append(tr)}$('totals').textContent=`${catalog.stage} · ${catalog.threshold} IoU matching · ${list.length} recordings shown. Dataset totals: `+showMetrics(catalog.totals[method]);}
function currentTime(){return playing?Math.min(stopAt??detail.duration,playTime+audioCtx.currentTime-startedAt):playTime}
function pause(){if(playing){playTime=currentTime();playing=false;if(source){source.onended=null;source.stop();source=null}}}
async function ensureAudio(){if(buffer)return;const version=loadVersion; audioCtx??=new AudioContext();const r=await fetch('audio?name='+encodeURIComponent(detail.filename));if(!r.ok)throw Error('Could not load audio');const decoded=await audioCtx.decodeAudioData(await r.arrayBuffer());if(version!==loadVersion)throw Error('Recording changed; press play again');buffer=decoded;}
async function play(end=null){await ensureAudio();await audioCtx.resume();pause();if(playTime>=detail.duration)playTime=0;stopAt=end;source=audioCtx.createBufferSource();source.buffer=buffer;source.connect(audioCtx.destination);startedAt=audioCtx.currentTime;playing=true;source.onended=()=>{playTime=stopAt??detail.duration;playing=false;source=null;draw()};source.start(0,playTime,Math.max(.001,(end??detail.duration)-playTime));}
async function load(row){pause();const version=++loadVersion;++plotVersion;buffer=null;detail=null;selection=null;playTime=0;$('error').textContent='';$('filename').textContent='Loading…';try{const d=await api('detail?method='+encodeURIComponent(row.method)+'&name='+encodeURIComponent(row.filename));if(version!==loadVersion)return;detail=d;viewStart=0;viewEnd=d.duration;$('filename').textContent=d.filename+' · '+d.method;$('metrics').textContent=showMetrics(d.metrics);$('selection').textContent='No interval selected.';$('attemptRows').replaceChildren();for(const a of d.attempts){const tr=document.createElement('tr');cells(tr,fields.map(f=>typeof a[f]==='number'&&f.endsWith('_start')||typeof a[f]==='number'&&f.endsWith('_end')?a[f].toFixed(4):f==='iou'&&a[f]!==undefined?a[f].toFixed(3):a[f]));tr.onclick=()=>selectInterval(a.predicted_start!==undefined?[a.predicted_start,a.predicted_end]:[a.reference_start,a.reference_end],a.status);$('attemptRows').append(tr)}filter();draw()}catch(e){err(e)}}
function selectInterval(pair,label){selection=pair;$('selection').textContent=`Selected ${label}: ${pair[0].toFixed(4)}–${pair[1].toFixed(4)} s (${((pair[1]-pair[0])*1000).toFixed(1)} ms)`;pause();playTime=pair[0];draw()}
function xFor(t){return 60+(t-viewStart)/(viewEnd-viewStart)*(canvas.width-80)}
function timeFor(event){const rect=canvas.getBoundingClientRect();return viewStart+((event.clientX-rect.left)*canvas.width/rect.width-60)/(canvas.width-80)*(viewEnd-viewStart)}
function draw(){const width=Math.max(600,Math.round(canvas.getBoundingClientRect().width));if(canvas.width!==width||canvas.height!==300){canvas.width=width;canvas.height=300}ctx.clearRect(0,0,canvas.width,300);if(!detail)return;ctx.font='12px system-ui';ctx.fillStyle='#334';ctx.fillText('Wave',3,25);ctx.fillText('RMS dBFS',3,145);ctx.fillText('Ref',3,246);ctx.fillText('Pred',3,275);ctx.strokeStyle='#dce2eb';for(const y of [70,115,220]){ctx.beginPath();ctx.moveTo(60,y);ctx.lineTo(width-20,y);ctx.stroke()}const peak=detail.amplitude_peak;ctx.strokeStyle='#355c95';for(let i=0;i<detail.wave_times.length;i++){const t=detail.wave_times[i];if(t<viewStart||t>viewEnd)continue;const x=xFor(t);ctx.beginPath();ctx.moveTo(x,70-detail.wave_max[i]/peak*40);ctx.lineTo(x,70-detail.wave_min[i]/peak*40);ctx.stroke()}ctx.fillStyle='#526176';ctx.fillText('±'+peak.toFixed(3),3,55);for(const db of [-90,-60,-30,0]){const y=220-(db+100)/100*80;ctx.fillText(String(db),25,y);ctx.strokeStyle='#edf0f5';ctx.beginPath();ctx.moveTo(60,y);ctx.lineTo(width-20,y);ctx.stroke()}ctx.strokeStyle='#8561b5';ctx.beginPath();for(let i=0;i<detail.energy_times.length;i++){const x=xFor(detail.energy_times[i]),y=220-(Math.max(-100,Math.min(0,detail.energy_db[i]))+100)/100*80;i?ctx.lineTo(x,y):ctx.moveTo(x,y)}ctx.stroke();for(const [pairs,y,color] of [[detail.reference,232,'#19845a'],[detail.predicted,261,'#d77b16']]){ctx.save();ctx.beginPath();ctx.rect(60,y,width-80,24);ctx.clip();pairs.forEach(([a,b],i)=>{ctx.fillStyle=color;ctx.globalAlpha=.65;ctx.fillRect(xFor(a),y,Math.max(1,xFor(b)-xFor(a)),20);ctx.globalAlpha=1;ctx.fillStyle='#102030';ctx.fillText(String(i+1),xFor(a)+2,y+14)});ctx.restore()}if(selection){ctx.strokeStyle='#222';ctx.strokeRect(xFor(selection[0]),20,xFor(selection[1])-xFor(selection[0]),265)}const t=currentTime();ctx.strokeStyle='#dc3345';ctx.beginPath();ctx.moveTo(xFor(t),15);ctx.lineTo(xFor(t),285);ctx.stroke();ctx.fillStyle='#334';ctx.fillText(viewStart.toFixed(3)+' s',60,298);ctx.fillText(viewEnd.toFixed(3)+' s',width-85,298);$('clock').textContent=t.toFixed(3)+' / '+detail.duration.toFixed(3)+' s';$('scroll').value=detail.duration>viewEnd-viewStart?viewStart/(detail.duration-(viewEnd-viewStart))*1000:0;}
async function refreshPlot(){if(!detail)return;const version=++plotVersion,name=detail.filename;try{const data=await api(`plot?name=${encodeURIComponent(name)}&start=${viewStart}&end=${viewEnd}`);if(version!==plotVersion||!detail||detail.filename!==name)return;Object.assign(detail,data);draw()}catch(e){err(e)}}
function zoom(factor){if(!detail)return;const old=viewEnd-viewStart,span=Math.min(detail.duration,Math.max(.05,old*factor)),center=Math.max(viewStart,Math.min(viewEnd,currentTime()));viewStart=Math.max(0,Math.min(detail.duration-span,center-span/2));viewEnd=viewStart+span;refreshPlot()}
canvas.onclick=e=>{if(!detail)return;const y=(e.clientY-canvas.getBoundingClientRect().top)*300/canvas.getBoundingClientRect().height,t=timeFor(e);if(y>=232){const pairs=y<258?detail.reference:detail.predicted;const pair=pairs.find(([a,b])=>a<=t&&t<=b);if(pair){selectInterval(pair,y<258?'reference':'prediction');return}}pause();playTime=Math.max(0,Math.min(detail.duration,t));draw()};
canvas.onmousemove=e=>{if(!detail)return;const t=Math.max(viewStart,Math.min(viewEnd,timeFor(e)));const nearest=(arr)=>arr.reduce((best,v,i)=>Math.abs(v-t)<Math.abs(arr[best]-t)?i:best,0);const i=nearest(detail.energy_times),j=nearest(detail.wave_times);$('cursor').textContent=`Time ${t.toFixed(4)} s · RMS ${detail.energy_db[i].toFixed(2)} dBFS · Display-bin amplitude ${detail.wave_min[j].toFixed(4)} to ${detail.wave_max[j].toFixed(4)}`};
$('play').onclick=()=>{if(detail)(playing?pause():play().catch(err))};$('selected').onclick=()=>{if(detail&&selection){pause();playTime=selection[0];play(selection[1]).catch(err)}};
$('zoomIn').onclick=()=>zoom(.5);$('zoomOut').onclick=()=>zoom(2);$('fit').onclick=()=>{if(detail){viewStart=0;viewEnd=detail.duration;refreshPlot()}};$('scroll').oninput=()=>{if(detail){const span=viewEnd-viewStart;viewStart=Number($('scroll').value)/1000*(detail.duration-span);viewEnd=viewStart+span;refreshPlot()}};
for(const id of ['previous','next'])$(id).onclick=()=>{if(!list.length)return;let i=list.findIndex(r=>detail&&r.filename===detail.filename);i=Math.max(0,Math.min(list.length-1,i+(id==='next'?1:-1)));load(list[i])};
$('search').oninput=filter;$('sort').onchange=filter;$('method').onchange=()=>{const old=detail?.filename;filter();const r=list.find(r=>r.filename===old)||list[0];if(r)load(r)};window.onresize=draw;
(async()=>{try{catalog=await api('catalog');for(const method of Object.keys(catalog.totals)){const o=document.createElement('option');o.value=o.textContent=method;$('method').append(o)}filter();if(list.length)await load(list[0])}catch(e){err(e)}})();
function tick(){if(playing)draw();requestAnimationFrame(tick)}tick();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--audio-dir', type=Path)
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not all((args.results, args.audio_dir, args.csv)):
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        if not args.results:
            chosen = filedialog.askdirectory(title='Select MAE results folder (contains metrics.csv)')
            if not chosen:
                return
            args.results = Path(chosen)
        if not args.audio_dir:
            chosen = filedialog.askdirectory(title='Select audio folder')
            if not chosen:
                return
            args.audio_dir = Path(chosen)
        if not args.csv:
            chosen = filedialog.askopenfilename(title='Select manual annotation CSV', filetypes=[('CSV', '*.csv')])
            if not chosen:
                return
            args.csv = Path(chosen)
        root.destroy()
    state = Inspection(args.results, args.audio_dir, args.csv)
    token = secrets.token_urlsafe(24)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(state, token))
    url = f'http://127.0.0.1:{server.server_port}/{token}/'
    print(f'Viewer: {url}\nPress Ctrl+C in this terminal to stop. Read-only; no files are changed.', flush=True)
    if not args.no_browser:
        threading.Timer(.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, ImportError) as error:
        raise SystemExit(f'Error: {error}') from error
