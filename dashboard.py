from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
import json
import asyncio
import random
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(input_dim * patch_size, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        B, T, C = x.shape
        n_patches = T // self.patch_size
        x = x[:, :n_patches * self.patch_size].reshape(B, n_patches, C * self.patch_size)
        x = self.projection(x)
        return self.layer_norm(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.tril(torch.ones(2048, 2048)).view(1, 1, 2048, 2048))
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ff = FeedForward(hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x

class CandleTransformerV6(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, num_layers=4, num_heads=4, patch_size=4, max_seq_len=2048, pred_len=5):
        super().__init__()
        self.patch_embedding = PatchEmbedding(input_dim, hidden_dim, patch_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len // patch_size, hidden_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.pred_head = nn.Linear(hidden_dim, pred_len * 3 * 2)
        self.pred_len = pred_len
        self.patch_size = patch_size
        self.input_dim = input_dim

    def forward(self, x):
        B, T, C = x.shape
        x = self.patch_embedding(x)
        T_patch = x.shape[1]
        x = x + self.pos_embedding[:, :T_patch]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        
        last_hidden = x[:, -1]
        out = self.pred_head(last_hidden)
        out = out.view(B, self.pred_len, 3, 2)
        
        mean = out[:, :, :, 0]
        log_std = out[:, :, :, 1].clamp(-4, 2)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, x, temperature=1.0):
        mean, std = self.forward(x)
        std = std * temperature
        eps = torch.randn_like(mean)
        return mean + eps * std, mean, std

# Load data
csv_path = r'C:\Users\Mann Patel\Downloads\kronos_making\XAUUSD_M1_202101040105_202512312324.csv'
df = pd.read_csv(csv_path, sep='\t')
df.columns = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'VOL', 'SPREAD']

ohlcv = df[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL']].values.astype(np.float32)
dates = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y.%m.%d %H:%M:%S')

# Build features
ret_1 = np.log(ohlcv[1:, 3] / ohlcv[:-1, 3])
ret_1 = np.concatenate([[0], ret_1])

ret_5 = np.log(ohlcv[5:, 3] / ohlcv[:-5, 3])
ret_5 = np.concatenate([np.zeros(5), ret_5])

hl_range = (ohlcv[:, 1] - ohlcv[:, 2]) / ohlcv[:, 3]
oc_range = np.abs(ohlcv[:, 0] - ohlcv[:, 3]) / ohlcv[:, 3]

vol_20 = pd.Series(ret_1).rolling(20, min_periods=1).std().values
vol_20 = np.nan_to_num(vol_20, nan=0.0)

vol = np.log1p(ohlcv[:, 4:5])

delta = np.diff(ohlcv[:, 3], prepend=ohlcv[0, 3])
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean().values
avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean().values
rs = avg_gain / (avg_loss + 1e-8)
rsi = 100 - (100 / (1 + rs))
rsi = (rsi - 50) / 50

ema_12 = pd.Series(ohlcv[:, 3]).ewm(span=12, adjust=False).mean().values
ema_26 = pd.Series(ohlcv[:, 3]).ewm(span=26, adjust=False).mean().values
macd = (ema_12 - ema_26) / ohlcv[:, 3]
macd_signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
macd_hist = macd - macd_signal

bb_mid = pd.Series(ohlcv[:, 3]).rolling(20, min_periods=1).mean().values
bb_std = pd.Series(ohlcv[:, 3]).rolling(20, min_periods=1).std().values
bb_std = np.nan_to_num(bb_std, nan=0.0)
bb_upper = bb_mid + 2 * bb_std
bb_lower = bb_mid - 2 * bb_std
bb_pos = (ohlcv[:, 3] - bb_lower) / (bb_upper - bb_lower + 1e-8)
bb_pos = (bb_pos - 0.5) * 2

tr1 = ohlcv[:, 1] - ohlcv[:, 2]
tr2 = np.abs(ohlcv[:, 1] - np.roll(ohlcv[:, 3], 1))
tr3 = np.abs(ohlcv[:, 2] - np.roll(ohlcv[:, 3], 1))
tr = np.maximum(np.maximum(tr1, tr2), tr3)
tr[0] = tr1[0]
atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
atr_norm = atr / ohlcv[:, 3]

dt_index = pd.DatetimeIndex(dates)
hour = dt_index.hour.values.astype(np.float32) / 12.0 - 1.0
day_of_week = dt_index.dayofweek.values.astype(np.float32) / 3.0 - 1.0

features = np.column_stack([
    ret_1.reshape(-1, 1),
    ret_5.reshape(-1, 1),
    hl_range.reshape(-1, 1),
    oc_range.reshape(-1, 1),
    vol,
    vol_20.reshape(-1, 1),
    rsi.reshape(-1, 1),
    macd.reshape(-1, 1),
    macd_signal.reshape(-1, 1),
    macd_hist.reshape(-1, 1),
    bb_pos.reshape(-1, 1),
    atr_norm.reshape(-1, 1),
    hour.reshape(-1, 1),
    day_of_week.reshape(-1, 1),
])

mean = features.mean(axis=0)
std = features.std(axis=0) + 1e-8
features_norm = ((features - mean) / std).astype(np.float32)

# Load checkpoint
checkpoint = torch.load('candle_transformer_v6_best.pth', map_location=device, weights_only=False)
target_mean = checkpoint['target_mean']
target_std = checkpoint['target_std']

seq_len = 128
pred_len = 5
model = CandleTransformerV6(input_dim=14, hidden_dim=128, num_layers=4, num_heads=4, patch_size=4, max_seq_len=128, pred_len=pred_len).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

TEMPERATURE = 0.3  # Optimal from evaluation

def reconstruct_ohlc(last_o, last_h, last_l, last_c, pred_components):
    pred_close_ret = pred_components[0]
    pred_hl_ret = pred_components[1]
    pred_body_ret = pred_components[2]
    
    pred_c = last_c * np.exp(pred_close_ret)
    pred_hl = (last_h - last_l) * np.exp(pred_hl_ret)
    pred_body = np.abs(last_c - last_o) * np.exp(pred_body_ret) * np.sign(last_c - last_o) if (last_c - last_o) != 0 else 0
    
    if pred_body >= 0:
        pred_o = pred_c - pred_body
    else:
        pred_o = pred_c + pred_body
    
    pred_h = max(pred_c, pred_o) + pred_hl / 2
    pred_l = min(pred_c, pred_o) - pred_hl / 2
    
    return pred_o, pred_h, pred_l, pred_c

def get_random_chunk():
    max_start = len(ohlcv) - seq_len - pred_len - 1
    if max_start <= seq_len:
        return None, None, None
    start = random.randint(seq_len, max_start)
    
    # Historical context
    hist_start = start - seq_len
    hist_end = start
    
    historical = []
    for i in range(hist_start, hist_end):
        historical.append({
            'time': dates.iloc[i].timestamp(),
            'open': float(ohlcv[i, 0]),
            'high': float(ohlcv[i, 1]),
            'low': float(ohlcv[i, 2]),
            'close': float(ohlcv[i, 3]),
            'volume': float(ohlcv[i, 4])
        })
    
    # Ground truth
    ground_truth = []
    for i in range(start, start + pred_len):
        ground_truth.append({
            'time': dates.iloc[i].timestamp(),
            'open': float(ohlcv[i, 0]),
            'high': float(ohlcv[i, 1]),
            'low': float(ohlcv[i, 2]),
            'close': float(ohlcv[i, 3]),
            'volume': float(ohlcv[i, 4])
        })
    
    # Predictions with stochastic sampling
    context = features_norm[hist_start:hist_end].copy()
    context_tensor = torch.from_numpy(context).unsqueeze(0).to(device, dtype=torch.float32)
    
    with torch.no_grad():
        sampled, pred_mean, pred_std = model.sample(context_tensor, temperature=TEMPERATURE)
        pred_components = sampled.cpu().numpy()[0]
    
    # Denormalize
    pred_components = pred_components * target_std + target_mean
    
    last_o = ohlcv[start - 1, 0]
    last_h = ohlcv[start - 1, 1]
    last_l = ohlcv[start - 1, 2]
    last_c = ohlcv[start - 1, 3]
    
    predictions = []
    for step in range(pred_len):
        pred_o, pred_h, pred_l, pred_c = reconstruct_ohlc(
            last_o, last_h, last_l, last_c, pred_components[step]
        )
        last_o, last_h, last_l, last_c = pred_o, pred_h, pred_l, pred_c
        
        next_dt = dates.iloc[start + step]
        predictions.append({
            'time': next_dt.timestamp(),
            'open': float(pred_o),
            'high': float(pred_h),
            'low': float(pred_l),
            'close': float(pred_c),
            'volume': 0.0,
            'predicted': True
        })
    
    return historical, ground_truth, predictions

connected_clients = set()

async def broadcast_predictions():
    while True:
        try:
            hist, gt, preds = get_random_chunk()
            if hist is not None:
                message = json.dumps({
                    'historical': hist,
                    'ground_truth': gt,
                    'predictions': preds,
                    'timestamp': datetime.now().isoformat()
                })
                for ws in connected_clients.copy():
                    try:
                        await ws.send_text(message)
                    except:
                        connected_clients.discard(ws)
        except Exception as e:
            print(f"Broadcast error: {e}")
        await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(broadcast_predictions())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        hist, gt, preds = get_random_chunk()
        if hist is not None:
            await websocket.send_text(json.dumps({
                'historical': hist,
                'ground_truth': gt,
                'predictions': preds,
                'timestamp': datetime.now().isoformat()
            }))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.discard(websocket)

@app.get("/")
async def root():
    return HTMLResponse(html_content)

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>XAUUSD M1 - V6 Probabilistic Predictions</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        body { margin: 0; padding: 20px; background: #131722; color: #d1d4dc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 10px; font-weight: 500; }
        .model-info { text-align: center; margin-bottom: 10px; color: #787b86; font-size: 13px; }
        .metrics { display: flex; justify-content: center; gap: 20px; margin-bottom: 15px; flex-wrap: wrap; }
        .metric { background: #1e222d; padding: 10px 20px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #0ecb8e; }
        .metric-label { font-size: 12px; color: #787b86; }
        .status { text-align: center; margin-bottom: 20px; padding: 10px; background: #1e222d; border-radius: 8px; }
        .status.connected { border-left: 4px solid #089981; }
        .status.disconnected { border-left: 4px solid #f23645; }
        #chart { width: 100%; height: 700px; background: #131722; border-radius: 8px; }
        .legend { display: flex; justify-content: center; gap: 20px; margin-top: 15px; flex-wrap: wrap; }
        .legend-item { display: flex; align-items: center; gap: 8px; font-size: 14px; }
        .legend-color { width: 20px; height: 12px; border-radius: 2px; }
        .historical-up { background: #089981; }
        .historical-down { background: #f23645; }
        .predicted-up { background: #0ecb8e; opacity: 0.7; border: 1px dashed #089981; }
        .predicted-down { background: #ff5b6b; opacity: 0.7; border: 1px dashed #f23645; }
        .truth-up { background: #089981; opacity: 0.5; }
        .truth-down { background: #f23645; opacity: 0.5; }
        .last-update { text-align: center; margin-top: 10px; font-size: 12px; color: #787b86; }
    </style>
</head>
<body>
    <div class="container">
        <h1>XAUUSD M1 - V6 Probabilistic Predictions</h1>
        <div class="model-info">V6 Model: 128-dim, 4 layers, 4 heads, predicts mean+std for (close_ret, hl_ret, body_ret) | Temp=0.3 | 500K samples | Val NLL: 0.238</div>
        <div class="metrics">
            <div class="metric"><div class="metric-value" id="mseVal">--</div><div class="metric-label">MSE</div></div>
            <div class="metric"><div class="metric-value" id="maeVal">--</div><div class="metric-label">MAE</div></div>
            <div class="metric"><div class="metric-value" id="dirVal">--</div><div class="metric-label">Dir Acc</div></div>
        </div>
        <div id="status" class="status disconnected">Connecting...</div>
        <div id="chart"></div>
        <div class="legend">
            <div class="legend-item"><div class="legend-color historical-up"></div><span>Historical (64 bars context)</span></div>
            <div class="legend-item"><div class="legend-color truth-up"></div><span>Ground Truth</span></div>
            <div class="legend-item"><div class="legend-color predicted-up"></div><span>Predicted (5 steps, stochastic)</span></div>
        </div>
        <div class="last-update" id="lastUpdate">Last update: --</div>
    </div>

    <script>
        const chart = LightweightCharts.createChart(document.getElementById('chart'), {
            width: document.getElementById('chart').clientWidth,
            height: 700,
            layout: { background: { color: '#131722' }, textColor: '#d1d4dc' },
            grid: { vertLines: { color: '#2a2e39' }, horzLines: { color: '#2a2e39' } },
            crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
            rightPriceScale: { borderColor: '#2a2e39' },
            timeScale: { borderColor: '#2a2e39', timeVisible: true, secondsVisible: false },
        });

        const histSeries = chart.addCandlestickSeries({
            upColor: '#089981', downColor: '#f23645',
            borderUpColor: '#089981', borderDownColor: '#f23645',
            wickUpColor: '#089981', wickDownColor: '#f23645',
        });

        const truthSeries = chart.addCandlestickSeries({
            upColor: '#089981', downColor: '#f23645',
            borderUpColor: '#089981', borderDownColor: '#f23645',
            wickUpColor: '#089981', wickDownColor: '#f23645',
        });

        const predSeries = chart.addCandlestickSeries({
            upColor: '#0ecb8e', downColor: '#ff5b6b',
            borderUpColor: '#0ecb8e', borderDownColor: '#ff5b6b',
            wickUpColor: '#0ecb8e', wickDownColor: '#ff5b6b',
        });

        const statusEl = document.getElementById('status');
        const lastUpdateEl = document.getElementById('lastUpdate');
        const mseEl = document.getElementById('mseVal');
        const maeEl = document.getElementById('maeVal');
        const dirEl = document.getElementById('dirVal');
        let ws;

        function connect() {
            ws = new WebSocket(`ws://${location.host}/ws`);
            ws.onopen = () => {
                statusEl.textContent = 'Connected - New random chunk every 60s (V6 Probabilistic, temp=0.3)';
                statusEl.className = 'status connected';
            };
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updateChart(data.historical, data.ground_truth, data.predictions);
                lastUpdateEl.textContent = `Last update: ${new Date(data.timestamp).toLocaleTimeString()}`;
                if (data.ground_truth && data.predictions) {
                    calcMetrics(data.ground_truth, data.predictions);
                }
            };
            ws.onclose = () => {
                statusEl.textContent = 'Disconnected - Reconnecting...';
                statusEl.className = 'status disconnected';
                setTimeout(connect, 3000);
            };
            ws.onerror = () => ws.close();
        }

        function calcMetrics(truth, preds) {
            if (!truth || !preds || truth.length !== preds.length) return;
            let mse = 0, mae = 0, dirCorrect = 0;
            for (let i = 0; i < truth.length; i++) {
                const diff = preds[i].close - truth[i].close;
                mse += diff * diff;
                mae += Math.abs(diff);
                if (i > 0) {
                    const predDir = preds[i].close - preds[i-1].close;
                    const actualDir = truth[i].close - truth[i-1].close;
                    if ((predDir > 0 && actualDir > 0) || (predDir < 0 && actualDir < 0)) dirCorrect++;
                }
            }
            mse /= truth.length;
            mae /= truth.length;
            const dirAcc = truth.length > 1 ? dirCorrect / (truth.length - 1) : 0;
            mseEl.textContent = mse.toFixed(2);
            maeEl.textContent = mae.toFixed(2);
            dirEl.textContent = (dirAcc * 100).toFixed(1) + '%';
        }

        function updateChart(historical, ground_truth, predictions) {
            const histData = historical.map(c => ({
                time: c.time,
                open: c.open, high: c.high, low: c.low, close: c.close
            }));
            const truthData = ground_truth.map(c => ({
                time: c.time,
                open: c.open, high: c.high, low: c.low, close: c.close
            }));
            const predData = predictions.map(c => ({
                time: c.time,
                open: c.open, high: c.high, low: c.low, close: c.close
            }));
            histSeries.setData(histData);
            truthSeries.setData(truthData);
            predSeries.setData(predData);
            chart.timeScale().fitContent();
        }

        window.addEventListener('resize', () => {
            chart.applyOptions({ width: document.getElementById('chart').clientWidth });
        });

        connect();
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)