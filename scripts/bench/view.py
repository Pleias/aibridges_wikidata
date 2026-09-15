"""Browse every local RLM run through a live, filterable web interface.

The server reads ``results/runs`` on demand. New runs therefore appear without
regenerating an HTML file. Existing ``*_scores.json`` files enrich matching
runs; runs without a score remain visible as ``unscored``.

Usage:
    uv run python scripts/bench/view.py --open
    uv run python scripts/bench/view.py --port 8765
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "runs"
MAX_CHARS = 20_000
CODE_BLOCK = re.compile(r"```(?:python|repl)?\s*\n(.*?)```", re.DOTALL)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def task_index() -> dict[str, dict]:
    tasks = {}
    for path in (ROOT / "tasks").rglob("*.json"):
        task = read_json(path)
        if isinstance(task, dict) and task.get("id"):
            tasks[task["id"]] = task
    return tasks


def score_index() -> dict[str, dict]:
    scores = {}
    for path in (ROOT / "results").glob("*_scores.json"):
        rows = read_json(path, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("run"):
                scores[row["run"]] = {**row, "score_file": path.name}
    return scores


def run_directories() -> list[Path]:
    paths = set()
    for filename in ("metadata.json", "messages.json", "question.txt", "answer.md"):
        paths.update(path.parent for path in RUNS.rglob(filename))
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)


def run_summary(run_dir: Path, tasks: dict, scores: dict) -> dict:
    metadata = read_json(run_dir / "metadata.json", {}) or {}
    score = scores.get(run_dir.name, {})
    task_id = metadata.get("task") or score.get("task")
    task = tasks.get(task_id, {})
    taxonomy = task.get("taxonomy", {})
    question_path = run_dir / "question.txt"
    relative = run_dir.relative_to(RUNS)
    return {
        "path": relative.as_posix(),
        "folder": relative.parent.as_posix(),
        "run": run_dir.name,
        "task": task_id or run_dir.name,
        "question": question_path.read_text() if question_path.exists()
        else task.get("question", ""),
        "category": score.get("category") or taxonomy.get("category") or "unknown",
        "type": score.get("type") or taxonomy.get("type") or "unknown",
        "band": score.get("band") or taxonomy.get("band"),
        "exact": score.get("exact"),
        "status": metadata.get("status") or score.get("status") or "in progress",
        "reads": metadata.get("charged", score.get("reads")),
        "iterations": metadata.get("iterations", score.get("iterations")),
        "sub_queries": metadata.get("sub_queries", 0),
        "total_tokens": metadata.get("total_tokens", score.get("total_tokens")),
        "model": metadata.get("model"),
        "score_file": score.get("score_file"),
        "modified": dt.datetime.fromtimestamp(
            run_dir.stat().st_mtime, dt.UTC
        ).isoformat(),
    }


def segments_of(content: str) -> list[dict]:
    segments, cursor = [], 0
    for match in CODE_BLOCK.finditer(content):
        prose = content[cursor:match.start()].strip()
        if prose:
            segments.append({"kind": "thought", "content": prose})
        source = match.group(1).strip()
        try:
            compile(source, "<viewer-cell>", "exec")
            kind = "code"
        except SyntaxError:
            kind = "artifact"
        segments.append({"kind": kind, "content": source})
        cursor = match.end()
    prose = content[cursor:].strip()
    if prose:
        segments.append({"kind": "thought", "content": prose})
    return segments or [{"kind": "thought", "content": content}]


def turns_of(messages: list[dict]) -> list[dict]:
    turns, current = [], None
    for message in messages:
        if message.get("role") == "system":
            continue
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if len(content) > MAX_CHARS:
            content = content[:MAX_CHARS] + "\n\n[... truncated for display ...]"
        if message.get("role") == "assistant":
            current = {"number": len(turns) + 1,
                       "segments": segments_of(content), "result": None}
            turns.append(current)
        elif message.get("role") == "user" and current is not None:
            current["result"] = content
    return turns


def safe_run_dir(relative: str) -> Path:
    path = (RUNS / relative).resolve()
    if not path.is_relative_to(RUNS.resolve()) or not path.is_dir():
        raise ValueError("unknown run")
    return path


def run_detail(relative: str) -> dict:
    run_dir = safe_run_dir(relative)
    tasks, scores = task_index(), score_index()
    summary = run_summary(run_dir, tasks, scores)
    task = tasks.get(summary["task"], {})
    messages = read_json(run_dir / "messages.json", []) or []
    answer_path = run_dir / "answer.md"
    return {
        **summary,
        "expected": task.get("answer", {}).get("expected"),
        "answer": answer_path.read_text() if answer_path.exists() else "",
        "wrong_fields": scores.get(run_dir.name, {}).get("wrong_fields", []),
        "turns_log": turns_of(messages),
    }


def catalogue() -> dict:
    tasks, scores = task_index(), score_index()
    runs = [run_summary(path, tasks, scores) for path in run_directories()]
    folders = {}
    for run in runs:
        folders[run["folder"]] = folders.get(run["folder"], 0) + 1
    return {
        "runs": runs,
        "folders": [{"path": path, "count": count}
                    for path, count in sorted(folders.items())],
        "refreshed": dt.datetime.now(dt.UTC).isoformat(),
    }


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RLM runs</title><style>
:root{--bg:#f5f5f2;--paper:#fff;--ink:#1b1c19;--muted:#73756e;--line:#dedfd9;
--accent:#315fc5;--ok:#228251;--bad:#bb3b32;--code:#181b18;--soft:#efefeb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
button,input,select{font:inherit;color:inherit}button{cursor:pointer}.app{height:100vh;
display:grid;grid-template-columns:230px minmax(390px,1fr) minmax(480px,1.25fr)}
aside,.runs,.detail{min-height:0;overflow:auto}aside{border-right:1px solid var(--line);
padding:22px 12px}.brand{font-size:18px;font-weight:720;letter-spacing:-.03em;padding:0 10px 20px}
.live{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--ok);
margin-right:7px;box-shadow:0 0 0 4px #22825118}.aside-label,.eyebrow{font-size:10px;
font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:var(--muted)}
.aside-label{padding:9px 10px}.folder{width:100%;border:0;background:transparent;text-align:left;
padding:7px 10px;border-radius:6px;display:flex;gap:8px;align-items:baseline}
.folder span:first-child{flex:1;min-width:0;overflow-wrap:anywhere}
.folder .when{display:block;font-size:10px;color:var(--muted);letter-spacing:.02em}
.folder:hover,.folder.on{background:var(--soft)}
.folder.on{color:var(--accent);font-weight:650}.folder b{font-size:11px;color:var(--muted);flex:none}
.runs{border-right:1px solid var(--line);background:var(--paper)}.toolbar{position:sticky;top:0;
z-index:3;background:#fffffff2;backdrop-filter:blur(12px);padding:18px;border-bottom:1px solid var(--line)}
.search{width:100%;border:1px solid var(--line);border-radius:7px;padding:9px 11px;background:var(--bg)}
.filters{display:flex;gap:6px;margin-top:8px}.filters select{min-width:0;flex:1;border:1px solid var(--line);
border-radius:6px;padding:5px 7px;background:var(--paper)}.count{color:var(--muted);font-size:12px;margin-top:8px}
.run{padding:13px 18px;border-bottom:1px solid var(--line);cursor:pointer;transition:background .12s}
.run:hover,.run.on{background:var(--soft)}.run.on{box-shadow:inset 3px 0 var(--accent)}
.run-top{display:flex;align-items:center;gap:8px}.dot{width:8px;height:8px;border-radius:50%;background:#aaa}
.dot.ok{background:var(--ok)}.dot.bad{background:var(--bad)}.run-id{font:600 12px ui-monospace,
SFMono-Regular,Menlo,monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.question{margin:6px 0;color:#3e403b;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;
overflow:hidden}.meta{color:var(--muted);font-size:11px;display:flex;gap:9px;flex-wrap:wrap}
.detail{padding:27px 30px 80px}.empty{height:100%;display:grid;place-items:center;color:var(--muted)}
.detail-head{padding-bottom:20px;border-bottom:1px solid var(--line)}h1{font-size:20px;margin:4px 0 7px;
letter-spacing:-.025em}.tag{display:inline-block;border:1px solid var(--line);border-radius:20px;
padding:1px 8px;font-size:11px;color:var(--muted);margin-right:4px}h2{font-size:11px;text-transform:uppercase;
letter-spacing:.08em;color:var(--muted);margin:24px 0 8px}.q{border-left:3px solid var(--accent);
padding:10px 13px;background:var(--soft)}pre{white-space:pre-wrap;word-break:break-word;margin:0;
font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}.compare{display:grid;grid-template-columns:1fr 1fr;
gap:10px}.compare>div{padding:11px;background:var(--soft);border-radius:7px}.timeline{border-left:1px solid var(--line)}
.turn{padding:0 0 24px 22px;position:relative}.turn:before{content:"";position:absolute;left:-5px;top:4px;
width:8px;height:8px;border:2px solid var(--accent);background:var(--bg);border-radius:50%}.turn-title{font:650 11px
ui-monospace,monospace;color:var(--accent);margin-bottom:8px}.thought{white-space:pre-wrap;color:var(--muted);margin:7px 0}
.cell{background:var(--code);color:#edf1ec;border-radius:8px;margin:8px 0;overflow:auto}.cell-label{padding:6px 10px;
font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#9da59d;border-bottom:1px solid #303630}
.cell pre{padding:10px 12px;min-width:max-content}.result{border-left:3px solid #92968e;background:var(--soft);padding:9px 11px;
margin-top:8px}.result.error{border-color:var(--bad)}.refresh{font-size:11px;color:var(--muted);padding:18px 10px 0}
@media(max-width:1050px){.app{grid-template-columns:180px 1fr}.detail{position:fixed;inset:0 0 0 180px;
background:var(--bg);z-index:5;display:none}.detail.open{display:block}.close{display:block!important}}
@media(max-width:650px){.app{grid-template-columns:1fr}aside{display:none}.detail{inset:0}.compare{grid-template-columns:1fr}}
.close{display:none;float:right;border:0;background:var(--soft);border-radius:6px;padding:5px 9px}
</style></head><body><div class="app">
<aside><div class="brand"><span class="live"></span>RLM runs</div><div class="aside-label">Folders</div>
<div id="folders"></div><div class="refresh" id="refreshed"></div></aside>
<section class="runs"><div class="toolbar"><input class="search" id="search" placeholder="Search runs, tasks, questions…">
<div class="filters"><select id="verdict"><option value="">All verdicts</option><option value="correct">Correct</option>
<option value="incorrect">Incorrect</option><option value="unscored">Unscored</option></select><select id="category"></select>
<select id="type"></select></div><div class="count" id="count">Loading…</div></div><div id="runs"></div></section>
<section class="detail" id="detail"><div class="empty">Select a run to inspect it.</div></section></div>
<script>
const esc=s=>String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
let data={runs:[],folders:[]},folder="",selected="";
const verdict=r=>r.exact===true?"correct":r.exact===false?"incorrect":"unscored";
function options(id,values,label){const e=document.getElementById(id),old=e.value;e.innerHTML=`<option value="">${label}</option>`+
 [...new Set(values)].sort().map(x=>`<option>${esc(x)}</option>`).join("");e.value=old}
// batch directories are named YYYYMMDD-HHMM_name; showing the stamp above the
// name keeps both readable in a narrow column, where one line clipped exactly
// the part that tells bench_v1 from bench_v4
function folderLabel(path){const m=path.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})_(.*)$/);
 return m?`<span class="when">${m[3]}/${m[2]} ${m[4]}:${m[5]}</span>${esc(m[6])}`:esc(path)}
function drawFolders(){document.getElementById("folders").innerHTML=
 `<button class="folder ${!folder?'on':''}" data-folder=""><span>All runs</span><b>${data.runs.length}</b></button>`+
 data.folders.map(f=>`<button class="folder ${folder===f.path?'on':''}" data-folder="${esc(f.path)}" title="${esc(f.path)}"><span>${folderLabel(f.path)}</span><b>${f.count}</b></button>`).join("")}
function filtered(){const q=document.getElementById("search").value.toLowerCase(),v=document.getElementById("verdict").value,
 c=document.getElementById("category").value,t=document.getElementById("type").value;return data.runs.filter(r=>(!folder||r.folder===folder)&&
 (!v||verdict(r)===v)&&(!c||r.category===c)&&(!t||r.type===t)&&(!q||(r.run+" "+r.task+" "+r.question).toLowerCase().includes(q)))}
function drawRuns(){const rows=filtered();document.getElementById("count").textContent=`${rows.length} runs`;
 document.getElementById("runs").innerHTML=rows.map(r=>`<article class="run ${selected===r.path?'on':''}" data-path="${esc(r.path)}">
 <div class="run-top"><span class="dot ${r.exact===true?'ok':r.exact===false?'bad':''}"></span><span class="run-id">${esc(r.task)}</span></div>
 <div class="question">${esc(r.question||"No question recorded")}</div><div class="meta"><span title="${esc(r.folder)}">${folderLabel(r.folder).replace(/<[^>]+>/g,' ').trim()}</span><span>${esc(r.type)}</span>
 <span>${r.reads??"—"} reads</span><span>${r.iterations??"—"} turns</span><span>${r.sub_queries??0} llm_query</span><span>${esc(r.status)}</span></div></article>`).join("")||`<div class="empty">No matching runs.</div>`}
function segment(s,i){if(s.kind==="code")return `<div class="cell"><div class="cell-label">Python cell ${i+1}</div><pre>${esc(s.content)}</pre></div>`;
 if(s.kind==="artifact")return `<div class="result error"><pre>${esc(s.content)}</pre></div>`;return `<div class="thought">${esc(s.content)}</div>`}
async function openRun(path){selected=path;drawRuns();const d=await fetch(`/api/run?path=${encodeURIComponent(path)}`).then(r=>r.json());
 let got;try{got=JSON.stringify(JSON.parse(d.answer),null,2)}catch{got=d.answer}const expected=d.expected==null?"No score/reference available":JSON.stringify(d.expected,null,2);
 document.getElementById("detail").innerHTML=`<button class="close" onclick="detail.classList.remove('open')">Close</button><header class="detail-head">
 <div class="eyebrow">${esc(d.folder)} · ${esc(verdict(d))}</div><h1>${esc(d.task)}</h1><span class="tag">${esc(d.category)}</span><span class="tag">${esc(d.type)}</span>
 <span class="tag">${d.reads??'—'} reads</span><span class="tag">${d.iterations??'—'} turns</span><span class="tag">${d.sub_queries??0} llm_query</span></header><h2>Question</h2><div class="q">${esc(d.question)}</div>
 <h2>Answer</h2><div class="compare"><div><div class="eyebrow">Expected</div><pre>${esc(expected)}</pre></div><div><div class="eyebrow">Model</div><pre>${esc(got)}</pre></div></div>
 <h2>Execution trace</h2><div class="timeline">${d.turns_log.map(t=>`<div class="turn"><div class="turn-title">Turn ${t.number}</div>${t.segments.map(segment).join("")}
 ${t.result?`<div class="result ${/Error:|Traceback/.test(t.result)?'error':''}"><pre>${esc(t.result)}</pre></div>`:''}</div>`).join("")}</div>`;
 document.getElementById("detail").classList.add("open")}
async function refresh(){const next=await fetch("/api/runs",{cache:"no-store"}).then(r=>r.json());data=next;
 options("category",data.runs.map(r=>r.category),"All categories");options("type",data.runs.map(r=>r.type),"All types");
 drawFolders();drawRuns();document.getElementById("refreshed").textContent="Live · refreshed "+new Date(data.refreshed).toLocaleTimeString()}
document.addEventListener("click",e=>{const f=e.target.closest("[data-folder]");if(f){folder=f.dataset.folder;drawFolders();drawRuns()}
 const r=e.target.closest("[data-path]");if(r)openRun(r.dataset.path)});["search","verdict","category","type"].forEach(id=>document.getElementById(id).addEventListener("input",drawRuns));
refresh();setInterval(refresh,5000);
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        request = urlparse(self.path)
        try:
            if request.path == "/":
                self.send(PAGE.encode(), "text/html; charset=utf-8")
            elif request.path == "/api/runs":
                self.send(json.dumps(catalogue(), ensure_ascii=False).encode(),
                          "application/json; charset=utf-8")
            elif request.path == "/api/run":
                relative = parse_qs(request.query).get("path", [""])[0]
                self.send(json.dumps(run_detail(relative), ensure_ascii=False).encode(),
                          "application/json; charset=utf-8")
            else:
                self.send(b"not found", "text/plain", 404)
        except (ValueError, OSError) as error:
            self.send(json.dumps({"error": str(error)}).encode(),
                      "application/json", 400)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"RLM run browser: {url}")
    print("Watching results/runs — Ctrl-C to stop")
    if args.open:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
