"""Real-time streaming dashboard.

A background thread captures a live Bybit *public* feed for the currently selected
symbol, runs label-free regime inference + a BULL/BEAR direction read + an
incremental paper maker on every snapshot, and pushes them to the browser over
Server-Sent Events. The coin is switchable live from the UI. Monitor only — no
orders are ever sent (Constitution Rule 3: WebSocket on the data path, no REST).

    lob_core web --live                 # stream BTCUSDT live (default)
    lob_core web --live --symbol ETHUSDT
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .schema import N_LEVELS, Regime, featurize_from_raw, snapshot_to_row, snapshot_to_vector

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"
DEFAULT_URL = "wss://stream.bybit.com/v5/public/spot"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT"]
_TICKS = {"BTCUSDT": 0.1, "ETHUSDT": 0.01, "SOLUSDT": 0.01,
          "XRPUSDT": 0.0001, "DOGEUSDT": 0.00001, "BNBUSDT": 0.01}

STREAM: _Stream | None = None   # singleton, created by serve_live()
_MAX_STREAMS = 24                 # cap concurrent SSE clients (Slowloris guard)
_active_streams = 0
_streams_lock = threading.Lock()


class _LiveMaker:
    """Incremental mirror of execution.simulator.run_backtest's per-step loop —
    a true cumulative paper PnL for the live stream (no rolling-window baseline)."""

    def __init__(self):
        from .execution.risk import RiskGate
        from .strategy.avellaneda import AvellanedaStoikovMaker
        from .strategy.maker import Quote
        self._Quote = Quote
        self.strategy = AvellanedaStoikovMaker()
        self.risk = RiskGate()
        self.inv = 0.0
        self.cash = 0.0
        self.q = Quote(None, 0.0, None, 0.0)
        self.last_mid = 0.0

    def step(self, row: dict, regime: int):
        mid = float(row["mid"])
        bb = mid - float(row["bid_px_0"])
        ba = mid + float(row["ask_px_0"])
        if not (math.isfinite(mid) and math.isfinite(bb) and math.isfinite(ba)):
            return self.cash + self.inv * self.last_mid, self.inv
        self.last_mid = mid
        tb, ts = float(row["trade_buy"]), float(row["trade_sell"])
        tb = tb if math.isfinite(tb) else 0.0
        ts = ts if math.isfinite(ts) else 0.0
        q = self.q
        if q.ask_px is not None and q.ask_sz > 0 and tb > 0 and q.ask_px <= ba + 1e-9:
            f = min(q.ask_sz, tb)
            self.inv -= f
            self.cash += f * q.ask_px
        if q.bid_px is not None and q.bid_sz > 0 and ts > 0 and q.bid_px >= bb - 1e-9:
            f = min(q.bid_sz, ts)
            self.inv += f
            self.cash -= f * q.bid_px
        pnl = self.cash + self.inv * mid
        if not self.risk.update(pnl):
            self.q = self._Quote(None, 0.0, None, 0.0)
        else:
            self.q = self.risk.clamp(self.strategy.decide(regime, bb, ba, self.inv), self.inv)
        return pnl, self.inv


class _Stream:
    def __init__(self, predictor, metrics, model_src):
        self.predictor = predictor
        self.metrics = metrics
        self.model_src = model_src
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.seq = 0
        self.latest = None
        self.scatter = None
        self.symbol = SYMBOLS[0]
        self.want = SYMBOLS[0]
        self.running = True
        self.error = None
        self._reset_state()

    def _reset_state(self):
        self.maker = _LiveMaker()
        self._mids = deque(maxlen=25)
        self._lat = deque(maxlen=400)
        self._reg = deque(maxlen=400)
        self._since_scatter = 0

    def set_symbol(self, sym: str):
        if sym in SYMBOLS:
            with self.lock:
                if sym != self.want:
                    self.want = sym
                    self._reset_state()

    def stop(self):
        with self.cond:
            self.running = False
            self.cond.notify_all()

    # ---- producer (capture thread) ----
    def _process(self, snap, sym):
        if sym != self.want:                       # symbol already changed — drop
            return
        # LiveOrderBook stores px offsets in TICK units; convert to price units so
        # the maker and the ladder (mid ∓ offset) are correct for every coin. The
        # synthetic path already uses tick=1, so this is a no-op there.
        tick = _TICKS.get(sym, 0.1)
        row = snapshot_to_row(snap)
        for k in range(N_LEVELS):
            row[f"bid_px_{k}"] = float(row[f"bid_px_{k}"]) * tick
            row[f"ask_px_{k}"] = float(row[f"ask_px_{k}"]) * tick

        X, _ = featurize_from_raw(snapshot_to_vector(snap).reshape(1, -1))
        regime, z = int(Regime.RANGE), None
        if self.predictor is not None:
            try:
                regime = int(self.predictor.predict_features(X)[0])
                from .models.embedder import embed
                z = np.asarray(embed(self.predictor.model, X, self.predictor.stats)[0])
            except Exception as e:
                # Fail SAFE: a broken inference must NOT silently pick RANGE (the
                # least-cautious regime) and drop the maker's stand-aside veto.
                # Mirror the bridge's non-finite → TOXIC rule (microstructure.py).
                logger.warning("live regime inference failed, failing safe to TOXIC: %s", e)
                regime, z = int(Regime.TOXIC), None
        mid = float(snap.mid)

        # Publish atomically w.r.t. the symbol: the maker step, rolling-buffer
        # mutation and publish all happen under the lock, and we re-check the symbol
        # so a /set that lands mid-compute drops this straggler instead of polluting
        # the new coin's freshly-reset state.
        with self.cond:
            if sym != self.want:
                return
            pnl, inv = self.maker.step(row, regime)
            self._mids.append(mid)
            arr = np.fromiter(self._mids, float)
            drift = mid - arr[0]
            thr = max(1e-9, 0.4 * float(arr.std())) if len(arr) > 4 else 1e18
            dirv = 1 if drift > thr else (-1 if drift < -thr else 0)
            if z is not None:
                self._lat.append(z)
                self._reg.append(regime)
            self._since_scatter += 1
            if self._since_scatter >= 15 and len(self._lat) >= 20:
                self.scatter = self._compute_scatter()
                self._since_scatter = 0
            self.latest = {
                "dir": dirv, "mid": round(mid, 6),
                "bpx": [round(float(row[f"bid_px_{k}"]), 6) for k in range(N_LEVELS)],
                "bsz": [round(float(row[f"bid_sz_{k}"]), 3) for k in range(N_LEVELS)],
                "apx": [round(float(row[f"ask_px_{k}"]), 6) for k in range(N_LEVELS)],
                "asz": [round(float(row[f"ask_sz_{k}"]), 3) for k in range(N_LEVELS)],
                "cxl": round(sum(float(row[f"bid_cxl_{k}"]) + float(row[f"ask_cxl_{k}"])
                                 for k in range(N_LEVELS)), 1),
                "tf": round(float(row["trade_buy"] - row["trade_sell"]), 2),
                "rp": regime, "rt": regime, "pnl": round(pnl, 1), "inv": round(inv, 3),
                "sym": sym,
            }
            self.seq += 1
            self.cond.notify_all()

    def _compute_scatter(self):
        try:
            from sklearn.decomposition import PCA
            Z = np.array(self._lat)
            proj = PCA(n_components=2, random_state=0).fit_transform(Z)
            return {"x": [round(float(v), 3) for v in proj[:, 0]],
                    "y": [round(float(v), 3) for v in proj[:, 1]],
                    "r": [int(v) for v in self._reg]}
        except Exception:
            return self.scatter

    async def capture_forever(self):
        import websockets

        from .ingest.bybit import DepthMsg, TradeMsg, build_subscribe, parse_message
        from .ingest.orderbook import LiveOrderBook
        while self.running:
            sym = self.want
            self.symbol = sym
            book = LiveOrderBook(tick_size=_TICKS.get(sym, 0.1))
            try:
                async with websockets.connect(DEFAULT_URL, ping_interval=20,
                                              open_timeout=15) as ws:
                    await ws.send(json.dumps(build_subscribe(sym, 50)))
                    self.error = None
                    while self.running and self.want == sym:
                        try:                           # bounded wait so a switch in a
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except asyncio.TimeoutError:    # quiet market is noticed promptly
                            continue
                        try:
                            msg = json.loads(raw)
                        except (ValueError, TypeError):
                            continue
                        ev = parse_message(msg)
                        if isinstance(ev, DepthMsg):
                            book.apply_depth(ev.bids, ev.asks, ev.is_snapshot)
                            if book.ready():
                                self._process(book.snapshot(), sym)
                        elif isinstance(ev, TradeMsg):
                            book.apply_trades(ev.trades)
            except Exception as e:                       # network drop / bad symbol
                self.error = repr(e)
                await asyncio.sleep(1.0)


def _capture_thread(stream: _Stream):
    try:
        asyncio.run(stream.capture_forever())
    except Exception as e:
        stream.error = repr(e)


class _Handler(BaseHTTPRequestHandler):
    timeout = 30                 # drop stalled requests (Slowloris guard)

    def log_message(self, *a):   # quiet
        pass

    def handle_one_request(self):
        # a browser preconnect / client closing mid-request resets the socket —
        # benign, so swallow it instead of dumping a traceback.
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            html = (WEB_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif u.path == "/config.json":
            self._json({
                "mode": "live", "symbols": SYMBOLS, "symbol": STREAM.want,
                "regime_names": {str(r.value): r.name for r in Regime},
                "metrics": STREAM.metrics, "model": STREAM.model_src,
            })
        elif u.path == "/set":
            sym = parse_qs(u.query).get("symbol", [""])[0]
            STREAM.set_symbol(sym)
            self._json({"symbol": STREAM.want})
        elif u.path == "/stream":
            # read-only: the coin is changed only via /set, so a stale EventSource
            # reconnect can't silently switch the active symbol back.
            self._stream()
        else:
            self.send_error(404)

    def _stream(self):
        global _active_streams
        with _streams_lock:
            if _active_streams >= _MAX_STREAMS:
                self.send_error(503, "too many streams")
                return
            _active_streams += 1
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last = -1
            while STREAM.running:
                with STREAM.cond:
                    STREAM.cond.wait_for(
                        lambda last=last: STREAM.seq != last or not STREAM.running, timeout=5)
                    fresh = STREAM.seq != last
                    if fresh:
                        last = STREAM.seq
                        payload = {"snap": STREAM.latest, "scatter": STREAM.scatter,
                                   "symbol": STREAM.symbol, "error": STREAM.error}
                if not STREAM.running:
                    break
                if fresh:
                    self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                else:
                    self.wfile.write(b": keepalive\n\n")   # heartbeat / disconnect probe
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ValueError, OSError):
            pass
        finally:
            with _streams_lock:
                _active_streams -= 1


def _load_predictor():
    """Live data → prefer the real-Bybit-trained model; fall back to synthetic."""
    from .real import has_real_model, real_model_paths
    from .regime.predict import RegimePredictor
    if has_real_model():
        try:
            return RegimePredictor.load(*real_model_paths()), "real-Bybit-trained", True
        except Exception:
            pass
    if Path("artifacts/lob_encoder.safetensors").exists() and Path("artifacts/regime_model.npz").exists():
        try:
            return RegimePredictor.load(), "synthetic-trained", False
        except Exception:
            pass
    return None, "no model (run `lob_core real` or `--mode synthetic`)", False


def serve_live(port: int = 8000, symbol: str = "BTCUSDT", real: bool = True,
               open_browser: bool = True) -> int:
    global STREAM
    from .web.build import _load_metrics
    predictor, src, is_real = _load_predictor()
    STREAM = _Stream(predictor, _load_metrics(is_real), src)
    STREAM.want = symbol if symbol in SYMBOLS else SYMBOLS[0]
    STREAM.symbol = STREAM.want

    print(f"connecting live Bybit stream · {STREAM.want} · model: {src}")
    threading.Thread(target=_capture_thread, args=(STREAM,), daemon=True).start()

    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    except OSError as e:
        print(f"could not start server on port {port}: {e}\n"
              f"the port may be in use — try `lob_core web --live --port {port + 1}`.")
        STREAM.stop()
        return 1
    httpd.daemon_threads = True
    url = f"http://127.0.0.1:{port}/"
    print(f"\n  LOB-Core LIVE dashboard > {url}\n  coins: {', '.join(SYMBOLS)}\n  (Ctrl+C to stop)\n")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        STREAM.stop()
        httpd.shutdown()
    return 0
