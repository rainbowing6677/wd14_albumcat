#!/usr/bin/env python3
"""Browser demo: type a Chinese phrase, get the photos that look like it.

A five-minute way to *see* what semantic search does, without the phone app:

    python tools/serve_semantic_demo.py --dir samples
    # open http://127.0.0.1:8801

It reuses the exact same `SemanticEncoder` the service and the phone use, so
what you see here is what `/plan` would return as vectors — same model, same
preprocessing, same normalisation.

Vectors are cached in `<image dir>/.vectors.npz`, keyed by (filename, size).
Re-runs start in about five seconds; add or remove an image and it re-encodes.

Notes
    * Only images already in the folder can be found. This is retrieval, not
      generation: ask for something the library does not contain and you get
      the least-unrelated picture, which is worse than no answer.
    * Scores are cosine similarity: ~0.40+ close, 0.35-0.40 related,
      below that mostly "closest available".
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from albumcat.config import Config  # noqa: E402
from albumcat.semantic import SemanticEncoder, cosine_scores  # noqa: E402

IMAGE_DIR = REPO / "samples"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PORT = 8801

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>相册语义搜索 · demo</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; background:#0F172A; color:#F8FAFC;
         font-family:"Microsoft YaHei","Segoe UI",system-ui,sans-serif; }
  header { position:relative; padding:28px 40px 18px; border-bottom:1px solid #1E293B; }
  h1 { margin:0 0 6px; font-size:26px; font-weight:700; }
  .sub { color:#64748B; font-size:14px; }
  .bar { display:flex; gap:10px; margin-top:18px; max-width:820px; }
  input { flex:1; padding:13px 18px; border-radius:10px; border:1px solid #334155;
          background:#111C2E; color:#F8FAFC; font-size:17px; outline:none; }
  input:focus { border-color:#22D3EE; box-shadow:0 0 0 3px rgba(34,211,238,.15); }
  button { padding:13px 24px; border-radius:10px; border:0; cursor:pointer;
           background:#22D3EE; color:#0F172A; font-size:16px; font-weight:700; }
  button:hover { background:#67E8F9; }
  .chips { display:flex; gap:8px; margin-top:12px; flex-wrap:wrap; }
  .chip { padding:7px 15px; border-radius:20px; border:1px solid #334155;
          background:transparent; color:#94A3B8; cursor:pointer; font-size:14px; }
  .chip:hover { border-color:#22D3EE; color:#22D3EE; }
  main { padding:22px 40px 60px; }
  .status { color:#64748B; font-size:14px; margin-bottom:16px; min-height:20px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:16px; }
  .card { position:relative; border-radius:12px; overflow:hidden;
          background:#111C2E; border:1px solid #1E293B; }
  .card img { width:100%; height:170px; object-fit:cover; display:block; }
  .card .meta { padding:10px 12px; }
  .card .name { font-size:12px; color:#94A3B8; overflow:hidden;
                text-overflow:ellipsis; white-space:nowrap; }
  .score { position:absolute; top:10px; right:10px; padding:5px 11px; border-radius:16px;
           font-size:13px; font-weight:700; font-variant-numeric:tabular-nums;
           background:rgba(15,23,42,.82); backdrop-filter:blur(4px); }
  .s-hi { color:#22C55E; } .s-mid { color:#F97316; } .s-lo { color:#64748B; }
  .bar-bg { height:3px; background:#1E293B; }
  .bar-fg { height:3px; background:#22D3EE; }
  .qr { position:absolute; top:20px; right:28px; display:flex; flex-direction:column;
        align-items:center; gap:6px; }
  .qr img { width:104px; height:104px; border-radius:8px; background:#fff; padding:7px; display:block; }
  .qr span { font-size:12px; color:#64748B; white-space:nowrap; }
  @media (max-width: 760px) {
    header { padding:16px 16px 12px; }
    h1 { font-size:21px; margin-bottom:4px; }
    .sub { font-size:12px; line-height:1.5; }
    .bar { flex-direction:column; max-width:none; margin-top:12px; }
    button { width:100%; padding:12px; }
    .chip { padding:6px 12px; font-size:13px; }
    .qr { display:none; }
    main { padding:14px 16px 40px; }
    .status { font-size:12px; line-height:1.6; }
    .grid { grid-template-columns:repeat(auto-fill,minmax(140px,1fr)); gap:10px; }
    .card img { height:118px; }
    .card .meta { padding:6px 8px; }
    .card .name { font-size:10px; }
    .score { font-size:11px; padding:3px 8px; top:6px; right:6px; }
  }
</style></head><body>
<header>
  <h1>相册语义搜索</h1>
  <div class="sub">输入任意中文描述，按画面相似度排序 —— 不需要照片上有这个标签</div>
  <div class="bar">
    <input id="q" placeholder="试试：沙滩 / 雪山 / 花 / 森林 / 城市夜景" autofocus>
    <button onclick="search()">搜索</button>
  </div>
  {QR_BLOCK}
  <div class="chips">
    <span class="chip" onclick="pick('海边')">海边</span>
    <span class="chip" onclick="pick('教室')">教室</span>
    <span class="chip" onclick="pick('城市夜景')">城市夜景</span>
    <span class="chip" onclick="pick('森林')">森林</span>
    <span class="chip" onclick="pick('雪')">雪</span>
    <span class="chip" onclick="pick('樱花')">樱花</span>
    <span class="chip" onclick="pick('双马尾')">双马尾</span>
    <span class="chip" onclick="pick('蓝头发')">蓝头发</span>
  </div>
</header>
<main>
  <div class="status" id="status"></div>
  <div class="grid" id="grid"></div>
</main>
<script>
const q = document.getElementById('q');
q.addEventListener('keydown', e => { if (e.key === 'Enter') search(); });
function pick(t) { q.value = t; search(); }
function cls(s) { return s >= 0.28 ? 's-hi' : (s >= 0.20 ? 's-mid' : 's-lo'); }
async function search() {
  const text = q.value.trim();
  if (!text) return;
  const status = document.getElementById('status');
  const grid = document.getElementById('grid');
  status.textContent = '正在编码「' + text + '」…';
  const t0 = performance.now();
  const res = await fetch('/api/search?q=' + encodeURIComponent(text));
  const data = await res.json();
  if (!data.ok) { status.textContent = '出错了：' + data.error; return; }
  const ms = (performance.now() - t0).toFixed(0);
  // 不自动判断「命中/未命中」：实测「戴眼镜」0.374 与「太空站」0.375 几乎相同，
  // 绝对分数分不开。改成恒定说明，由人看分数判断。
  status.textContent = '「' + text + '」→ ' + data.results.length + ' 张，排序 ' +
                       data.search_ms + 'ms（共 ' + data.total + ' 张）' +
                       '　·　0.40+ 很像　0.35–0.40 大致相关　更低多半只是最相近的';
  status.style.color = '#64748B';
  if (!data.results.length) {
    grid.innerHTML = '';
    status.textContent = '「' + text + '」没有足够相似的结果 —— 库里可能没有这种画面';
    return;
  }
  grid.innerHTML = data.results.map(r => `
    <div class="card">
      <img src="/img?i=${r.i}" loading="lazy">
      <span class="score ${cls(r.score)}">${r.score.toFixed(3)}</span>
      <div class="meta"><div class="name">${r.name}</div></div>
      <div class="bar-bg"><div class="bar-fg" style="width:${Math.max(2, r.score * 100)}%"></div></div>
    </div>`).join('');
}
window.addEventListener('load', () => {
  document.getElementById('status').textContent = '共 {COUNT} 张图已建好向量，输入一个词试试';
});
</script></body></html>"""


def _lan_ip() -> str | None:
    """Best-effort LAN address, so the page can show a scannable URL."""
    import socket

    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        ip = probe.getsockname()[0]
        probe.close()
        return ip
    except Exception:  # noqa: BLE001
        return None


def _qr_block(url: str) -> str:
    """QR code as a data URI. Optional: without `qrcode` we just show the URL."""
    if not url:
        return ""
    try:
        import base64
        import io

        import qrcode

        image = qrcode.make(url)
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return (
            '<div class="qr">'
            f'<img src="data:image/png;base64,{encoded}" alt="{url}">'
            "<span>手机扫码打开</span>"
            "</div>"
        )
    except Exception:  # noqa: BLE001
        return ""


def build_thumbs(files: list[Path], thumb_dir: Path) -> None:
    thumb_dir.mkdir(exist_ok=True)
    for path in files:
        out = thumb_dir / (path.stem + ".jpg")
        if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
            continue
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((720, 720), Image.LANCZOS)
            im.save(out, "JPEG", quality=82)


def main() -> int:
    parser = argparse.ArgumentParser(description="semantic search demo (browser)")
    parser.add_argument("--dir", type=Path, default=IMAGE_DIR, help="image folder (default: samples/)")
    parser.add_argument("--port", type=int, default=PORT, help=f"端口（默认 {PORT}）")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 让同一局域网里的手机也能打开")
    parser.add_argument("--qr-url", default=None, help="二维码指向的地址；默认用探测到的局域网 IP")
    args = parser.parse_args()

    image_dir = args.dir.resolve()
    thumb_dir = image_dir / ".thumbs"
    port = args.port

    config = Config.load(REPO / "config.toml")
    encoder = SemanticEncoder(config.semantic)
    info = encoder.info()
    if not encoder.available:
        print("语义模型没加载成功，原因：", info.reason)
        print("先导出模型：python tools\\export_chinese_clip_onnx.py")
        return 1
    if not encoder.text_available:
        print("警告：只有图像塔，没有文本塔 —— 无法编码查询词。")
        return 1

    lan_ip = _lan_ip() if args.host in ("0.0.0.0", "::") else None
    qr_url = args.qr_url or (f"http://{lan_ip}:{port}" if lan_ip else "")
    qr_block = _qr_block(qr_url)

    files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in EXTS) if image_dir.exists() else []
    if not files:
        print(f"{image_dir} 里没有图片。")
        print("把几张图片拷进这个目录，或者用 --dir 指向自己的图库，例如：")
        print("  python tools\\serve_semantic_demo.py --dir D:/pics")
        print("建议至少 10 张、内容有差异的图，否则排序看不出区别。")
        return 1

    cache = image_dir / ".vectors.npz"
    signature = np.array([f"{f.name}:{f.stat().st_size}" for f in files])
    matrix = None
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=True) as data:
                if "signature" in data and np.array_equal(data["signature"], signature):
                    matrix = data["matrix"]
                    print(f"向量缓存命中（{len(files)} 张），跳过编码")
        except Exception as exc:  # noqa: BLE001
            print(f"缓存读取失败，重新编码：{exc}")

    if matrix is None:
        print(f"编码 {len(files)} 张图……（首次会慢，之后走缓存）")
        vectors = []
        for i, path in enumerate(files, 1):
            try:
                with Image.open(path) as im:
                    vectors.append(encoder.encode_image(im.convert("RGB")))
                print(f"  [{i}/{len(files)}] {path.name}")
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}/{len(files)}] 跳过 {path.name}：{exc}")
                vectors.append(np.zeros(info.dim, dtype=np.float32))
        matrix = np.stack(vectors)
        try:
            np.savez(cache, matrix=matrix, signature=signature)
            print(f"向量已缓存到 {cache.name}，下次启动秒开")
        except Exception as exc:  # noqa: BLE001
            print(f"缓存写入失败（不影响演示）：{exc}")

    build_thumbs(files, thumb_dir)

    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "albumcat-demo"

        def log_message(self, fmt, *args):  # noqa: A003
            pass

        def _json(self, payload: dict, code: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                body = INDEX_HTML.replace("{COUNT}", str(len(files))).replace("{QR_BLOCK}", qr_block).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/api/search":
                text = (parse_qs(parsed.query).get("q") or [""])[0].strip()
                if not text:
                    self._json({"ok": False, "error": "查询是空的"}, 400)
                    return
                try:
                    import time

                    started = time.perf_counter()
                    with lock:
                        query = encoder.encode_text(text)
                    scores = cosine_scores(query, matrix)
                    order = np.argsort(-scores)
                    results = [
                        {
                            "i": int(i),
                            "name": files[i].name,
                            "score": round(float(scores[i]), 4),
                        }
                        for i in order[:16]
                    ]
                    self._json(
                        {
                            "ok": True,
                            "query": text,
                            "total": len(files),
                            "results": results,
                            "search_ms": round((time.perf_counter() - started) * 1000, 1),
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)
                return

            if parsed.path == "/img":
                raw = (parse_qs(parsed.query).get("i") or [""])[0]
                try:
                    index = int(raw)
                    path = thumb_dir / (files[index].stem + ".jpg")
                    data = path.read_bytes()
                except Exception:  # noqa: BLE001
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            self.send_error(404)

    if args.host in ("0.0.0.0", "::"):
        import socket

        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("8.8.8.8", 80))
            lan_ip = probe.getsockname()[0]
            probe.close()
        except Exception:  # noqa: BLE001
            lan_ip = "<本机局域网 IP>"
        print("\n服务已启动：")
        print(f"  本机     http://127.0.0.1:{port}")
        print(f"  同局域网  {qr_url}   （手机浏览器打开这个，或扫页面右上角的码）")
        print("  手机连的是 Wi-Fi 时，IP 可能和上面这个不同：")
        print("    用 --qr-url http://<那个IP>:端口 覆盖二维码地址")
        print("  首次从手机访问若打不开，多半是 Windows 防火墙拦了入站连接。")
    else:
        print(f"\n服务已启动： http://127.0.0.1:{port}")
    print("按 Ctrl+C 停止\n")
    ThreadingHTTPServer((args.host, port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
