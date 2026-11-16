"""3팀 촬영 영상 모으기 허브 — 같은 와이파이 안에서 핸드폰·PC 로 영상을 올린다 (표준 라이브러리만 사용).

    py scripts\\team_video_hub.py                      # Windows (기본 폴더 D:\\교과10_3팀영상)
    py scripts\\team_video_hub.py --dir E:\\영상 --minutes 0

교과7 tools/share_server.py 의 '공용 보관함'을 영상 모으기에 맞게 줄인 것.
  - 올릴 때 이름을 고르면 <폴더>/<이름>/ 에 핸드폰 원본 파일 이름 그대로 저장한다 (같은 이름은 덮어쓰지 않음).
  - 4자리 번호(기본 3333 고정, --pin 으로 바꿈)를 아는 사람만, 같은 사설망(192.168.x.x 등)에서만 쓸 수 있다. 틀리면 잠시 잠긴다.
  - 영상(mp4·mov 등)과 zip 만 받는다. 파일을 열거나 실행하지 않는다. 지우기 기능은 없다 (서버 PC 에서 직접).
  - 큰 파일도 조금씩 받고 보낸다. 서버 PC 에 2GB 는 남긴다.
  - --minutes 가 지나면 저절로 꺼진다 (기본 240분, 0 이면 끄지 않음).
모인 영상은 Git·공개 저장소에 올리지 않는다 (얼굴이 나오는 촬영 원본).
"""
import argparse
import ipaddress
import json
import os
import re
import secrets
import shutil
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, quote, urlparse

ALLOWED = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".3gp", ".zip"}
NAME_BAD = set('\\/:*?"<>|\r\n\t\0')
MAX_FAILS, LOCK_SECONDS = 8, 60
KEEP_FREE = 2 * 1024 ** 3
CHUNK = 1024 * 1024
DEFAULT_PEOPLE = "S01,S02,S03,S04,S05,S06"


def clean(text, limit):
    text = PurePosixPath(str(text).replace("\\", "/")).name
    text = "".join(c for c in text if c not in NAME_BAD and c >= " ").strip(". ")
    return text if 0 < len(text) <= limit else None


def file_name(name):
    name = clean(name, 150)
    return name if name and Path(name).suffix.lower() in ALLOWED else None


def unique_path(folder, name):
    target = Path(folder) / name
    n = 0
    while target.exists():
        n += 1
        target = target.with_name(f"{Path(name).stem}_{n}{Path(name).suffix}")
    return target


def is_local(ip):
    try:
        a = ipaddress.ip_address(ip.split("%")[0])
    except ValueError:
        return False
    if a.version == 6 and a.ipv4_mapped:
        a = a.ipv4_mapped
    return a.is_private or a.is_loopback or a.is_link_local


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark">
<title>3팀 영상 모으기</title><style>
:root{--bg:#f5f7fb;--card:#fff;--line:#e3e8f0;--text:#1c2433;--muted:#5d6b82;--accent:#3366ff;--soft:#eaf0ff;--ok:#0f8a4b;--bad:#d03b3b}
@media (prefers-color-scheme:dark){:root{--bg:#0e1420;--card:#171f2e;--line:#2a3548;--text:#e8edf6;--muted:#9aa8bf;--accent:#6c8cff;--soft:#1d2a4d;--ok:#4cc58a;--bad:#ff7b7b}}
*{box-sizing:border-box}[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--text);font-family:'Pretendard','Malgun Gothic',system-ui,sans-serif;line-height:1.6}
main{max-width:760px;margin:0 auto;padding:20px 16px 60px}h1{font-size:21px;margin:0}h2{font-size:16px;margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin:14px 0}
.muted{color:var(--muted);font-size:13px}.ok{color:var(--ok)}.bad{color:var(--bad)}
.pin{display:flex;gap:10px;justify-content:center;margin:12px 0}
.pin input{width:54px;height:62px;font-size:28px;text-align:center;border:2px solid var(--line);border-radius:12px;background:var(--bg);color:var(--text)}
select,input.t{font-size:16px;padding:9px 12px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--text)}
.drop{display:block;width:100%;border:2px dashed var(--line);border-radius:14px;padding:28px 12px;text-align:center;cursor:pointer;background:var(--bg);color:inherit;font:inherit}
.drop.over{border-color:var(--accent);background:var(--soft)}.big{font-size:16px;font-weight:700}
.bar{height:7px;background:var(--line);border-radius:9px;overflow:hidden;margin:4px 0}.bar i{display:block;height:100%;width:0;background:var(--accent)}
.q{padding:8px 2px;border-top:1px solid var(--line);font-size:14px}.q:first-child{border-top:0}.qh{display:flex;justify-content:space-between;gap:8px;word-break:break-all}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left}th{background:var(--soft)}
details{margin-top:6px}summary{cursor:pointer;color:var(--accent);font-size:13.5px}ul.f{margin:4px 0 0 18px;padding:0;font-size:13px}
a{color:var(--accent)}button.b{border:0;border-radius:10px;padding:8px 14px;background:var(--accent);color:#fff;font:inherit;font-weight:600;cursor:pointer}
</style></head><body><main>
<h1>3팀 영상 모으기</h1><div class="muted">교과10 동작 영상 · 같은 와이파이 전용 · 핸드폰에서 바로 올려도 됩니다</div>

<div class="card" id="lock"><h2 style="text-align:center">접속 번호 4자리</h2>
<div class="pin" id="pins"><input inputmode="numeric" autocomplete="off"><input inputmode="numeric" autocomplete="off"><input inputmode="numeric" autocomplete="off"><input inputmode="numeric" autocomplete="off"></div>
<div id="pinmsg" class="bad" style="text-align:center;min-height:22px"></div></div>

<div id="app" hidden>
<div class="card"><h2>1. 내 이름</h2>
<select id="who"></select> <input class="t" id="whoNew" placeholder="목록에 없으면 이름 입력" style="width:190px">
<div class="muted" style="margin-top:6px">고른 이름의 폴더에 저장됩니다. 이 브라우저가 기억합니다.</div></div>

<div class="card"><h2>2. 영상 올리기</h2>
<button class="drop" id="drop" type="button"><div class="big">영상을 끌어다 놓거나 눌러서 고르기</div>
<div class="muted">여러 개 한꺼번에 · <b>파일 이름은 바꾸지 말고 핸드폰 원본 그대로</b> · mp4·mov·zip</div></button>
<input type="file" id="file" accept="video/*,.zip" multiple hidden>
<div id="queue"></div></div>

<div class="card"><h2>3. 모인 영상 <span class="muted" id="room"></span></h2>
<table id="tbl"><tr><th>이름</th><th>개수</th><th>용량</th></tr></table>
<div class="muted" style="margin-top:8px">동작 17개 × 앞·옆 = 사람당 34개 이상이 목표입니다. 찍은 순서대로 올리면 촬영 시각으로 동작을 자동으로 묶을 수 있어요.</div>
<button class="b" id="refresh" type="button" style="margin-top:10px">새로고침</button></div>
</div>
</main><script>
const PEOPLE=__PEOPLE__;let PIN="";const $=id=>document.getElementById(id);
const st={get:k=>{try{return localStorage.getItem(k)}catch(e){return null}},set:(k,v)=>{try{localStorage.setItem(k,v)}catch(e){}}};
const mb=n=>n>=1073741824?(n/1073741824).toFixed(2)+" GB":(n/1048576).toFixed(1)+" MB";
const pins=[...document.querySelectorAll("#pins input")];
pins.forEach((el,i)=>{el.oninput=()=>{const ds=el.value.replace(/\D/g,"").split("");$("pinmsg").textContent="";
 el.value=ds[0]||"";let j=i;ds.slice(1).forEach(c=>{if(j<3)pins[++j].value=c});   // 빨리 치거나 붙여넣어도 다음 칸으로 나눠 넣기
 if(el.value)pins[Math.min(j+1,3)].focus();if(pins.every(p=>p.value))unlock(pins.map(p=>p.value).join(""))};
 el.onkeydown=e=>{if(e.key==="Backspace"&&!el.value&&i>0){pins[i-1].focus();pins[i-1].value=""}}});
async function list(){let r;try{r=await fetch("/api/list?pin="+PIN)}catch(e){$("pinmsg").textContent="서버에 연결할 수 없어요";return false}
 if(r.status!==200){const d=await r.json().catch(()=>({}));$("pinmsg").textContent=d.error||"번호가 맞지 않습니다";pins.forEach(p=>p.value="");pins[0].focus();return false}
 const d=await r.json();const t=$("tbl");t.querySelectorAll("tr:not(:first-child)").forEach(x=>x.remove());
 const names=[...new Set([...PEOPLE,...Object.keys(d.people)])];
 names.forEach(n=>{const fs=d.people[n]||[];const tr=t.insertRow();tr.insertCell().textContent=n;
  const c=tr.insertCell();c.textContent=fs.length+"개";const s=tr.insertCell();s.textContent=mb(fs.reduce((a,f)=>a+f.size,0));
  if(fs.length){const det=document.createElement("details");det.innerHTML="<summary>파일 보기</summary>";const ul=document.createElement("ul");ul.className="f";
   fs.forEach(f=>{const li=document.createElement("li");const a=document.createElement("a");a.textContent=f.name;
    a.href="/api/download?pin="+PIN+"&who="+encodeURIComponent(n)+"&name="+encodeURIComponent(f.name);li.append(a," · "+mb(f.size)+" · "+f.time);ul.append(li)});
   det.append(ul);c.append(det)}});
 $("room").textContent="(서버 남은 공간 "+mb(d.room)+")";return true}
async function unlock(p){PIN=p;if(await list()){$("lock").hidden=true;$("app").hidden=false;try{sessionStorage.setItem("pin",p)}catch(e){}}}
const sel=$("who");PEOPLE.forEach(n=>sel.add(new Option(n,n)));const sw=st.get("who");if(sw){if(!PEOPLE.includes(sw))sel.add(new Option(sw,sw));sel.value=sw}
const who=()=>($("whoNew").value.trim()||sel.value);
function up(f,box){return new Promise(res=>{const d=document.createElement("div");d.className="q";
 d.innerHTML='<div class="qh"><span></span><b class="muted">대기</b></div><div class="bar"><i></i></div><div class="muted m"></div>';
 d.querySelector("span").textContent=f.name;box.prepend(d);const s=d.querySelector("b"),bar=d.querySelector("i"),m=d.querySelector(".m");
 const x=new XMLHttpRequest();x.open("PUT","/api/upload?pin="+PIN+"&who="+encodeURIComponent(who())+"&name="+encodeURIComponent(f.name));
 x.upload.onprogress=e=>{if(e.lengthComputable){const p=Math.round(e.loaded*100/e.total);bar.style.width=p+"%";s.textContent=p+"%"}};
 x.onload=()=>{let r={};try{r=JSON.parse(x.responseText)}catch(e){}
  if(x.status===200){bar.style.width="100%";s.className="ok";s.textContent="완료 ✓";m.textContent=r.who+" / "+r.saved}
  else{s.className="bad";s.textContent="실패";m.textContent=r.error||("오류 "+x.status)}res()};
 x.onerror=()=>{s.className="bad";s.textContent="실패";m.textContent="연결이 끊겼습니다";res()};x.send(f)})}
let busy=false;
async function upload(files){files=[...files];if(!files.length)return;if(!who()){alert("이름을 먼저 고르세요");return}
 if(busy){alert("올리는 중인 파일이 끝나면 다시 올려 주세요");return}st.set("who",who());busy=true;
 for(const f of files)await up(f,$("queue"));busy=false;list()}
$("drop").onclick=()=>$("file").click();$("file").onchange=e=>{upload(e.target.files);e.target.value=""};
const dz=$("drop");["dragenter","dragover"].forEach(t=>dz.addEventListener(t,e=>{e.preventDefault();dz.classList.add("over")}));
["dragleave","drop"].forEach(t=>dz.addEventListener(t,e=>{e.preventDefault();dz.classList.remove("over")}));dz.addEventListener("drop",e=>upload(e.dataTransfer.files));
$("refresh").onclick=list;window.onbeforeunload=e=>{if(busy){e.preventDefault();return""}};
try{const p=sessionStorage.getItem("pin");if(p)unlock(p)}catch(e){}
</script></body></html>"""


class Hub(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, base, pin, people, max_gb, log=print):
        super().__init__(addr, Handler)
        self.base, self.pin, self.log = Path(base), pin, log
        self.base.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_gb * 1024 ** 3)
        self.page = PAGE.replace("__PEOPLE__", json.dumps(people, ensure_ascii=False)).encode("utf-8")
        self.fails, self._lock = {}, threading.Lock()

    def check_pin(self, ip, given):
        with self._lock:
            count, until = self.fails.get(ip, (0, 0.0))
            if until > time.monotonic():
                return 429
            if given and secrets.compare_digest(str(given), self.pin):
                self.fails.pop(ip, None)
                return 200
            count += 1
            self.fails[ip] = (0, time.monotonic() + LOCK_SECONDS) if count >= MAX_FAILS else (count, 0.0)
            return 403

    def room(self):
        return max(0, shutil.disk_usage(self.base).free - KEEP_FREE)

    def listing(self):
        out = {}
        for d in sorted(p for p in self.base.iterdir() if p.is_dir() and not p.name.startswith(".")):
            files = sorted((f for f in d.iterdir() if f.is_file() and not f.name.startswith(".")), key=lambda f: f.name)
            out[d.name] = [{"name": f.name, "size": f.stat().st_size,
                            "time": time.strftime("%m-%d %H:%M", time.localtime(f.stat().st_mtime))} for f in files]
        return out


class Handler(BaseHTTPRequestHandler):
    server_version = "TeamVideoHub"

    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, data):
        if code >= 400:
            self.close_connection = True
        try:
            self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"))
        except OSError:
            pass

    def _guard(self, q):
        ip = self.client_address[0]
        if not is_local(ip):
            self._json(403, {"error": "같은 네트워크에서만 접속할 수 있습니다."})
            return False
        code = self.server.check_pin(ip, (q.get("pin") or [""])[0])
        if code != 200:
            self.server.log(f"[거부] {ip} 번호 불일치 또는 잠김")
            self._json(code, {"error": "번호가 맞지 않습니다." if code == 403 else "너무 많이 틀렸습니다. 1분 뒤에 다시 하세요."})
            return False
        return True

    def _target(self, q):
        who = clean((q.get("who") or [""])[0], 30)
        name = file_name((q.get("name") or [""])[0])
        return who, name

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/":
            if not is_local(self.client_address[0]):
                return self._json(403, {"error": "같은 네트워크에서만 접속할 수 있습니다."})
            return self._send(200, self.server.page, "text/html; charset=utf-8")
        if url.path == "/api/list":
            if self._guard(q):
                self._json(200, {"people": self.server.listing(), "room": self.server.room()})
            return
        if url.path == "/api/download":
            if not self._guard(q):
                return
            who, name = self._target(q)
            path = self.server.base / who / name if who and name else None
            if not path or not path.is_file():
                return self._json(404, {"error": "파일을 찾을 수 없습니다."})
            size = path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(name)}")
            self.end_headers()
            try:
                with open(path, "rb") as f:
                    shutil.copyfileobj(f, self.wfile, CHUNK)
            except OSError:
                self.close_connection = True
            return
        self._json(404, {"error": "없는 주소입니다."})

    def do_PUT(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path != "/api/upload":
            return self._json(404, {"error": "없는 주소입니다."})
        if not self._guard(q):
            return
        who, name = self._target(q)
        if not who:
            return self._json(400, {"error": "이름을 쓸 수 없습니다."})
        if not name:
            return self._json(400, {"error": "영상(mp4·mov 등)이나 zip 만 올릴 수 있습니다."})
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._json(411, {"error": "파일 크기를 알 수 없습니다."})
        if length <= 0 or length > self.server.max_bytes:
            return self._json(413, {"error": f"빈 파일이거나 너무 큽니다. (최대 {self.server.max_bytes / 1024 ** 3:g}GB)"})
        if length > self.server.room():
            return self._json(507, {"error": "서버 PC 의 저장 공간이 부족합니다."})
        folder = self.server.base / who
        folder.mkdir(exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".up-", suffix=".part", dir=folder)
        tmp = Path(tmp)
        try:
            remaining = length
            with os.fdopen(fd, "wb") as f:
                while remaining > 0:
                    chunk = self.rfile.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            if remaining:
                raise ValueError("파일이 중간에 끊겼습니다. 다시 올려 주세요.")
            final = unique_path(folder, name)
            tmp.replace(final)
            self.server.log(f"[받음] {self.client_address[0]} {who}/{final.name} ({length / 1048576:.1f} MB)")
            self._json(200, {"who": who, "saved": final.name, "size": length})
        except (ValueError, OSError) as e:
            self.server.log(f"[실패] {self.client_address[0]} {who}/{name}: {e}")
            self._json(400, {"error": str(e)})
        finally:
            tmp.unlink(missing_ok=True)


def local_addresses():
    found = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        found.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        found += [i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)]
    except OSError:
        pass
    return [ip for i, ip in enumerate(found) if ip not in found[:i]
            and ipaddress.ip_address(ip).is_private and not ip.startswith(("127.", "169.254.", "172."))]


def main(argv=None):
    ap = argparse.ArgumentParser(description="3팀 촬영 영상을 같은 와이파이에서 모으는 임시 서버")
    default = Path("D:/교과10_3팀영상") if sys.platform.startswith("win") else Path.home() / "team_videos"
    ap.add_argument("--dir", default=str(default), help="영상을 모을 폴더 (그 안에 사람 이름 폴더가 생김)")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--host", default="0.0.0.0", help="이 PC 안에서만 시험하려면 127.0.0.1")
    ap.add_argument("--people", default=DEFAULT_PEOPLE, help="이름 목록 (쉼표로 구분)")
    ap.add_argument("--max-gb", type=float, default=4, help="파일 한 개 최대 크기(GB)")
    ap.add_argument("--pin", default="3333", help="접속 번호 4자리 (고정). random 이면 켤 때마다 새로 만듦")
    ap.add_argument("--minutes", type=float, default=240, help="이 시간이 지나면 저절로 꺼짐 (0 이면 끄지 않음)")
    a = ap.parse_args(argv)
    people = [p.strip() for p in a.people.split(",") if p.strip()]
    pin = f"{secrets.randbelow(10000):04d}" if a.pin == "random" else a.pin
    if not (len(pin) == 4 and pin.isdigit()):
        print("--pin 은 숫자 4자리여야 합니다.")
        return 2
    try:
        hub = Hub((a.host, a.port), Path(a.dir).expanduser(), pin, people, a.max_gb,
                  log=lambda m: print(f"  {time.strftime('%H:%M:%S')} {m}", flush=True))
    except OSError as e:
        print(f"서버를 시작하지 못했습니다: {e}\n(같은 포트로 켜진 창이 있으면 먼저 닫으세요.)")
        return 2
    print("=" * 60)
    print("  3팀 영상 모으기 서버가 켜졌습니다. 팀원에게 아래 주소와 번호를 알려 주세요.")
    for ip in (local_addresses() if a.host in ("0.0.0.0", "") else [a.host]):
        print(f"      http://{ip}:{a.port}")
    print(f"\n  접속 번호:  {pin}")
    print(f"  저장 폴더:  {hub.base}   (남은 공간 {hub.room() / 1024 ** 3:.0f}GB)")
    if a.minutes > 0:
        print(f"  {a.minutes:g}분 뒤 저절로 꺼집니다. 끝나면 이 창을 닫거나 Ctrl+C.")
        t = threading.Timer(a.minutes * 60, hub.shutdown)
        t.daemon = True
        t.start()
    print("=" * 60, flush=True)
    try:
        hub.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        hub.server_close()
        print("서버를 껐습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
