#!/usr/bin/env python3
"""RAKSHA live demo UI (presentation only, never used by the judged run).

    python ui/server.py            -> opens http://127.0.0.1:8765

It runs the unchanged pipeline (`python main.py --input .. --output ..`) as a separate process, exactly as the
evaluator does, then shows the result: summary, original vs sanitised side by side, linked patients across
scripts, quarantine reasons, a paste-and-sanitise box and a secret-rotation proof. Standard library only;
binds to 127.0.0.1; every run uses fresh temporary folders that are deleted on exit.
"""
from __future__ import annotations

import argparse
import atexit
import base64
import csv
import difflib
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
UI = Path(__file__).resolve().parent
WORK = Path(tempfile.mkdtemp(prefix="raksha-ui-"))
atexit.register(shutil.rmtree, WORK, True)
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
MAX_VIEW = 160_000          # characters shown per file in the diff view
MAX_UPLOAD = 60 * 1024 * 1024

DEFAULT_INPUT = next((str(p) for p in (ROOT.parent / "Secret-Wards-ToolKit" / "dev" / "input",
                                        ROOT / "dev" / "input") if p.is_dir()), "")


# ----------------------------------------------------------------------------------------------- helpers
def _decode(b: bytes) -> str:
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return b.decode("utf-16", "replace")
    if len(b) > 4 and b[1:2] == b"\x00" and b[3:4] == b"\x00":
        return b.decode("utf-16-le", "replace")
    try:
        return b.decode("utf-8-sig")
    except UnicodeDecodeError:
        return b.decode("cp1252", "replace")


def _walk(root: Path) -> list[Path]:
    return sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix())


def _script_badge(text: str) -> str:
    if re.search(r"[ऀ-ॿ]", text):
        return "Devanagari"
    if re.search(r"(?i)\b(ka|ki|ke|ne|hai|tha|thi|bhaiya|ji|sahab|didi|bataya|nahi|aaya)\b", text):
        return "Hinglish"
    return "English"


def _pretty_json(text: str) -> str | None:
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except (ValueError, RecursionError):
        return None


_TOK = re.compile(r"\s+|[\wऀ-ॿ]+|[^\w\s]", re.UNICODE)


def _tok_diff(a: str, b: str) -> tuple[list, list]:
    ta, tb = _TOK.findall(a), _TOK.findall(b)
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    A, B = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        sa, sb = "".join(ta[i1:i2]), "".join(tb[j1:j2])
        flag = 0 if op == "equal" else 1
        if sa:
            A.append([sa, flag])
        if sb:
            B.append([sb, flag])
    return A, B


def _diff(a: str, b: str) -> tuple[list, list]:
    """Line-aligned diff with word-level highlights inside changed lines."""
    la, lb = a.splitlines(keepends=True), b.splitlines(keepends=True)
    sm = difflib.SequenceMatcher(None, la, lb, autojunk=False)
    A, B = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            s = "".join(la[i1:i2])
            A.append([s, 0])
            B.append([s, 0])
        elif op == "replace" and (i2 - i1) == (j2 - j1):
            for x, y in zip(la[i1:i2], lb[j1:j2]):
                pa, pb = _tok_diff(x, y)
                A += pa
                B += pb
        else:
            pa, pb = _tok_diff("".join(la[i1:i2]), "".join(lb[j1:j2]))
            A += pa
            B += pb
    return A, B


def _csv_rows(text: str) -> list[list[str]]:
    first = text.split("\n", 1)[0]
    delim = max([",", "\t", ";", "|"], key=first.count)
    try:
        return list(csv.reader(io.StringIO(text, newline=""), delimiter=delim))
    except csv.Error:
        return [ln.split(delim) for ln in text.splitlines()]


def _record_snippet(path: Path, loc: str) -> str:
    if not path.is_file():
        return ""
    text = _decode(path.read_bytes())
    if loc.startswith("row:"):
        rows = _csv_rows(text)
        i = int(loc[4:]) + 1
        return " · ".join(c for c in rows[i] if c.strip()) if i < len(rows) else ""
    if loc.startswith("/"):
        try:
            node = json.loads(text)
            for part in loc.strip("/").split("/"):
                part = part.replace("~1", "/").replace("~0", "~")
                node = node[int(part)] if isinstance(node, list) else node[part]
            return json.dumps(node, ensure_ascii=False)
        except (ValueError, KeyError, IndexError, TypeError, RecursionError):
            return ""
    return " ".join(text.split())


# ----------------------------------------------------------------------------------------------- jobs
def _new_job(input_dir: Path, secret: str, label: str) -> str:
    jid = secrets.token_hex(4)
    out = WORK / f"out_{jid}"
    out.mkdir(parents=True)
    job = {"id": jid, "input": str(input_dir), "output": str(out), "state": "running", "t0": time.time(),
           "seconds": 0.0, "label": label, "secret": secret, "error": ""}
    with LOCK:
        JOBS[jid] = job
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return jid


def _run(job: dict) -> None:
    priv = WORK / f"priv_{job['id']}"
    (priv / "tmp").mkdir(parents=True)
    env = dict(os.environ, RAKSHA_SECRET=job["secret"], TMPDIR=str(priv / "tmp"), TMP=str(priv / "tmp"),
               TEMP=str(priv / "tmp"), PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "main.py", "--input", job["input"], "--output", job["output"]]
    try:
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, timeout=900)
        job["exit"] = p.returncode
        job["stdout"] = p.stdout.decode("utf-8", "replace")[-2000:]
        if p.returncode != 0:
            job["error"] = (p.stderr.decode("utf-8", "replace")[-1500:] or "pipeline failed").strip()
    except subprocess.TimeoutExpired:
        job["error"] = "timed out after 900 s"
    job["seconds"] = round(time.time() - job["t0"], 2)
    job["state"] = "error" if job["error"] else "done"


def _index(job: dict) -> dict:
    """Input file -> report entry (via SHA-256, exactly how the evaluator maps them)."""
    if "_index" in job:
        return job["_index"]
    out = Path(job["output"])
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    by_sha = {}
    for e in report.get("files", []):
        by_sha.setdefault(str(e.get("input_sha256", "")).lower(), e)
    inp = Path(job["input"])
    files = []
    for p in _walk(inp):
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        e = by_sha.get(sha, {})
        rel = p.relative_to(inp).as_posix()
        o = e.get("output")
        files.append({"input": rel, "output": o, "status": e.get("status", "missing"), "reason": e.get("reason", ""),
                      "renamed": bool(o) and o.removeprefix("sanitized/") != rel, "size": p.stat().st_size})
    recs = []
    rp = out / "records.jsonl"
    if rp.is_file():
        recs = [json.loads(x) for x in rp.read_text(encoding="utf-8").splitlines() if x.strip()]
    job["_index"] = {"report": report, "files": files, "records": recs}
    return job["_index"]


def _summary(job: dict) -> dict:
    ix = _index(job)
    rep = ix["report"]
    psns = {r["patient"] for r in ix["records"] if r.get("patient")}
    by_dir: dict[str, int] = {}
    for f in ix["files"]:
        top = f["input"].split("/", 1)[0] if "/" in f["input"] else "(root)"
        by_dir[top] = by_dir.get(top, 0) + 1
    return {"counts": rep.get("counts", {}), "entities": rep.get("entities", {}), "timing": rep.get("timing", {}),
            "resources": rep.get("resources", {}), "files": ix["files"], "patients": len(psns),
            "linked": sum(1 for r in ix["records"] if r.get("patient")), "records": len(ix["records"]),
            "renamed": sum(1 for f in ix["files"] if f["renamed"]), "by_dir": by_dir, "seconds": job["seconds"],
            "input": job["input"], "label": job["label"]}


def _file_view(job: dict, i: int) -> dict:
    ix = _index(job)
    f = ix["files"][i]
    orig = _decode((Path(job["input"]) / f["input"]).read_bytes())
    san = ""
    if f["output"]:
        sp = Path(job["output"]) / f["output"]
        san = _decode(sp.read_bytes()) if sp.is_file() else ""
    if f["input"].lower().endswith(".json") and san:
        po, ps = _pretty_json(orig), _pretty_json(san)
        if po is not None and ps is not None:
            orig, san = po, ps
    trunc = len(orig) > MAX_VIEW or len(san) > MAX_VIEW
    orig, san = orig[:MAX_VIEW], san[:MAX_VIEW]
    if f["status"] != "sanitized":
        binary = "\x00" in orig[:4000] or orig[:4000].count("�") > 20
        A, B = [["(binary / unreadable file: content not shown)" if binary else orig, 0]], []
    else:
        A, B = _diff(orig, san)
    return {"file": f, "orig": A, "san": B, "truncated": trunc,
            "records": sum(1 for r in ix["records"] if r.get("file") == f["output"])}


def _patients(job: dict) -> dict:
    ix = _index(job)
    out_to_in = {f["output"]: f["input"] for f in ix["files"] if f["output"]}
    groups: dict[str, list] = {}
    for r in ix["records"]:
        if r.get("patient"):
            groups.setdefault(r["patient"], []).append(r)
    clusters = []
    for psn, recs in groups.items():
        items, scripts, files = [], set(), set()
        for r in recs[:40]:
            src = out_to_in.get(r["file"], "")
            orig = _record_snippet(Path(job["input"]) / src, r["loc"]) if src else ""
            san = _record_snippet(Path(job["output"]) / r["file"], r.get("out", r["loc"]))
            sc = _script_badge(orig)
            scripts.add(sc)
            files.add(r["file"])
            items.append({"file": r["file"].removeprefix("sanitized/"), "loc": r["loc"], "script": sc,
                          "orig": orig[:220], "san": san[:220]})
        clusters.append({"psn": psn, "n": len(recs), "files": len(files), "scripts": sorted(scripts), "items": items})
    clusters.sort(key=lambda c: (-len(c["scripts"]), -c["files"], -c["n"], c["psn"]))
    unlinked = sum(1 for r in ix["records"] if not r.get("patient"))
    return {"clusters": clusters[:80], "total": len(clusters), "unlinked": unlinked}


def _compare(a: dict, b: dict) -> dict:
    # Renamed files get secret-keyed names, so records are matched through their INPUT file (via SHA-256).
    def by_input(job: dict) -> dict:
        ix = _index(job)
        o2i = {f["output"]: f["input"] for f in ix["files"] if f["output"]}
        return {(o2i.get(r["file"], r["file"]), r["loc"]): r.get("patient") for r in ix["records"]}

    ma, mb = by_input(a), by_input(b)
    both = [k for k in ma if ma[k] and mb.get(k)]
    same = sum(1 for k in both if ma[k] == mb[k])
    pa = {v for v in ma.values() if v}
    pb = {v for v in mb.values() if v}
    def outputs(job: dict) -> dict:
        base = Path(job["output"])
        return {f["input"]: (base / f["output"]).read_bytes() for f in _index(job)["files"]
                if f["output"] and (base / f["output"]).is_file()}

    fa, fb = outputs(a), outputs(b)
    common = set(fa) & set(fb)
    example = None
    for k in sorted(both)[:1]:
        example = {"record": f"{k[0]} {k[1]}", "a": ma[k], "b": mb[k]}
    return {"records": len(both), "same_psn": same, "psn_overlap": len(pa & pb), "psns": len(pa),
            "files": len(common), "identical_files": sum(1 for k in common if fa[k] == fb[k]),
            "linkage_same": _partition(ma) == _partition(mb), "example": example}


def _partition(m: dict) -> set:
    g: dict[str, list] = {}
    for k, v in m.items():
        if v:
            g.setdefault(v, []).append(k)
    return {tuple(sorted(x)) for x in g.values()}


# ----------------------------------------------------------------------------------------------- HTTP
class H(BaseHTTPRequestHandler):
    server_version = "RakshaUI/1"

    def log_message(self, *a) -> None:  # no request logging (keeps the console clean)
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ValueError("upload too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def _job(self, q) -> dict:
        jid = (q.get("job") or [""])[0]
        if jid not in JOBS:
            raise KeyError("unknown run")
        return JOBS[jid]

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (UI / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/config":
                return self._json({"default_input": DEFAULT_INPUT})
            if u.path == "/api/status":
                j = self._job(q)
                return self._json({k: j.get(k) for k in ("id", "state", "error", "seconds", "label")}
                                  | {"elapsed": round(time.time() - j["t0"], 1)})
            if u.path == "/api/summary":
                return self._json(_summary(self._job(q)))
            if u.path == "/api/file":
                return self._json(_file_view(self._job(q), int(q["i"][0])))
            if u.path == "/api/patients":
                return self._json(_patients(self._job(q)))
            if u.path == "/api/compare":
                return self._json(_compare(JOBS[q["a"][0]], JOBS[q["b"][0]]))
            self._json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)

    def do_POST(self) -> None:
        u = urlparse(self.path)
        try:
            body = self._body()
            secret = (body.get("secret") or "demo-secret").strip()
            if u.path == "/api/run":
                inp = Path(body.get("input", "")).expanduser()
                if not inp.is_dir():
                    return self._json({"error": f"folder not found: {inp}"}, 400)
                return self._json({"job": _new_job(inp.resolve(), secret, inp.name)})
            if u.path == "/api/upload":
                d = WORK / f"in_{secrets.token_hex(4)}"
                for f in body.get("files", []):
                    rel = Path(*[p for p in Path(str(f["path"])).parts if p not in ("..", "/", "\\")])
                    dst = (d / rel).resolve()
                    if d.resolve() not in dst.parents:
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(base64.b64decode(f["b64"]))
                if not d.is_dir():
                    return self._json({"error": "no files received"}, 400)
                return self._json({"job": _new_job(d, secret, body.get("label") or "uploaded folder")})
            if u.path == "/api/try":
                ext = {"note": ".txt", "csv": ".csv", "json": ".json"}.get(body.get("kind"), ".txt")
                d = WORK / f"try_{secrets.token_hex(4)}"
                (d / "pasted").mkdir(parents=True)
                (d / "pasted" / f"sample{ext}").write_text(body.get("text", ""), encoding="utf-8")
                return self._json({"job": _new_job(d, secret, "pasted text")})
            if u.path == "/api/rotate":
                base = JOBS[body["job"]]
                return self._json({"job": _new_job(Path(base["input"]), secret, base["label"] + " (new secret)")})
            self._json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)


def main() -> None:
    ap = argparse.ArgumentParser(description="RAKSHA live demo UI")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    url = f"http://127.0.0.1:{a.port}"
    print(f"RAKSHA demo UI running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
