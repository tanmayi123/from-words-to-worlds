"""Rewind, live: type a sentence, watch it paint, edit it, branch from any step.

Run from the project folder (milestone-1/exp1) on a machine with a CUDA GPU:

    python -m w2w.live            then open http://127.0.0.1:8765 (it opens by itself)

The server listens on 127.0.0.1 only, so nothing outside this computer can reach it. It
serves viewer/live.html and streams each painting step to the page as it happens
(Server-Sent Events), one painting at a time.

  GET /api/paint?prompt=...&seed=...                  a new painting from noise
  GET /api/branch?parent=ID&at=K|auto&prompt=...      continue painting ID from after step K
                                                       with a new sentence

Every picture shown passes Stable Diffusion's safety checker first; a flagged frame is
sent black.

`at=auto` picks the rewind point from how the sentence changed, using what tasks 10 and
11 measured: a colour change needs to start almost from the beginning (after step 3), a
swapped object about after step 6, an added detail (hat, sunglasses, snow) about after
step 9, and a sentence that changes in more than three words is treated as a new picture.
"""
import argparse
import base64
import difflib
import io
import json
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import torch.nn.functional as F

from .attention import is_content
from .paths import ROOT
from .rewind import Painter

COLOURS = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "grey",
           "gray", "golden", "gold", "silver", "beige", "cyan", "magenta", "teal", "navy", "violet"}
MAP = 32


def sentence_words(tokenizer, prompt):
    """The prompt's words in order, each with its token slots, so the page can light up whole words.

    CLIP's tokenizer marks the last piece of each word with '</w>'; pieces are grouped on it.
    """
    ids = tokenizer(prompt, padding="max_length", max_length=77, truncation=True)["input_ids"]
    words, cur, slots = [], "", []
    for i, tid in enumerate(ids):
        if tid == tokenizer.bos_token_id:
            continue
        if tid == tokenizer.eos_token_id:
            break
        piece = tokenizer.convert_ids_to_tokens(tid)
        cur += piece.replace("</w>", "")
        slots.append(i)
        if piece.endswith("</w>"):
            words.append({"text": cur, "slots": slots, "content": is_content(cur)})
            cur, slots = "", []
    if slots:
        words.append({"text": cur, "slots": slots, "content": is_content(cur)})
    return words


def auto_rewind(old: str, new: str):
    """(step, kind) for an edit from `old` to `new`. See the module docstring."""
    a, b = old.lower().split(), new.lower().split()
    ops = [op for op in difflib.SequenceMatcher(a=a, b=b).get_opcodes() if op[0] != "equal"]
    changed = sum(max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in ops)
    added = {w.strip(",.") for tag, _, _, j1, j2 in ops if tag in ("insert", "replace") for w in b[j1:j2]}
    if not ops:
        return 0, "same sentence"
    if changed > 3:
        return 0, "new picture"
    if added & COLOURS:
        return 3, "colour"
    if all(tag == "replace" and (i2 - i1) == (j2 - j1) for tag, i1, i2, j1, j2 in ops):
        return 6, "swap"
    return 9, "added detail"


class Live:
    def __init__(self, painter: Painter):
        self.p = painter
        self.tok = painter.pipe.tokenizer
        self.timelines = {}          # id -> (Timeline, parent id or None)
        self.lock = threading.Lock()

    def resolve(self, tid, at):
        """The timeline that actually holds the latent before step `at` (a branch only holds
        steps from its own start; earlier steps live in its parent)."""
        tl, parent = self.timelines[tid]
        while at < tl.start and parent is not None:
            tl, parent = self.timelines[parent]
        return tl

    def frame_event(self, step, preview, maps, words):
        buf = io.BytesIO()
        preview.resize((384, 384)).save(buf, format="JPEG", quality=82)
        out = {"step": step, "img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}
        if maps is not None:
            m = maps.float()
            mass = [float(m[w["slots"]].mean()) if w["content"] else 0.0 for w in words]
            total = sum(mass) or 1.0
            heat = []
            for w in words:
                if not w["content"]:
                    heat.append(None)
                    continue
                h = F.interpolate(m[w["slots"]].mean(0)[None, None], size=(MAP, MAP), mode="area")[0, 0].numpy()
                h = (h - h.min()) / (h.max() - h.min() + 1e-8)
                heat.append(base64.b64encode((h * 255).round().astype(np.uint8).tobytes()).decode())
            out["share"] = [round(x / total, 4) for x in mass]
            out["heat"] = heat
        return out

    def run(self, send, prompt, seed, parent=None, at=0, kind=None):
        words = sentence_words(self.tok, prompt)
        tid = uuid.uuid4().hex[:8]
        src = self.resolve(parent, at) if parent else None
        send("start", {"id": tid, "prompt": prompt, "seed": seed, "parent": parent, "at": at, "kind": kind,
                       "steps": self.p.steps, "map": MAP,
                       "words": [{"text": w["text"], "content": w["content"]} for w in words]})
        with self.lock:
            tl = self.p.paint(prompt, seed, branch_from=src, at=at, safe=True,
                              on_step=lambda i, prev, maps: send("step", self.frame_event(i, prev, maps, words)))
        tl.previews.clear()          # frames already went to the page; keep only what rewinding needs
        tl.maps = None
        self.timelines[tid] = (tl, parent)
        send("done", {"id": tid})


def make_handler(live: Live, page: bytes):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def send(event, data):
                self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()
            return send

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(page)
                return
            if url.path not in ("/api/paint", "/api/branch"):
                self.send_error(404)
                return
            send = self._sse()
            try:
                prompt = q.get("prompt", "").strip()[:300]
                if not prompt:
                    raise ValueError("Type a sentence first.")
                if url.path == "/api/paint":
                    live.run(send, prompt, int(q.get("seed", "0")))
                else:
                    parent = q.get("parent", "")
                    if parent not in live.timelines:
                        raise ValueError("That painting is no longer in memory. Paint it again.")
                    kind = None
                    at = q.get("at", "auto")
                    if at == "auto":
                        at, kind = auto_rewind(live.timelines[parent][0].prompt, prompt)
                    at = max(0, min(int(at), live.p.steps - 1))
                    live.run(send, prompt, live.timelines[parent][0].seed, parent=parent, at=at, kind=kind)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # report to the page instead of dropping the stream
                try:
                    send("failed", {"message": str(e)})
                except (BrokenPipeError, ConnectionResetError):
                    pass
    return Handler


def main():
    ap = argparse.ArgumentParser(description="Rewind, live")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default=None, help="testing only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    from diffusers import StableDiffusionPipeline
    from .generate import SD_ID
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(args.model or SD_ID, dtype=dtype).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    live = Live(Painter(pipe, args.device, steps=args.steps))
    page = (ROOT / "viewer" / "live.html").read_bytes()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(live, page))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Rewind is running at {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
