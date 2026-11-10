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

# ============================================================
# MODEL ARCHITECTURES
# ============================================================

class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, patch_size=4, use_ln=False):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(input_dim * patch_size, hidden_dim)
        self.use_ln = use_ln
        if use_ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        B, T, C = x.shape
        n_patches = T // self.patch_size
        x = x[:, :n_patches * self.patch_size].reshape(B, n_patches, C * self.patch_size)
        x = self.projection(x)
        if self.use_ln:
            x = self.layer_norm(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, max_seq_len=2048):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Use non-persistent buffer so it doesn't get saved/loaded
        self.register_buffer('mask', torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len), persistent=False)
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
    def __init__(self, hidden_dim, dropout=0.1, v1_style=False):
        super().__init__()
        if v1_style:
            # V1 original architecture: Linear -> GELU -> Linear (no middle dropout)
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
        else:
            # V3/V5/V6 architecture: Linear -> GELU -> Dropout -> Linear
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, max_seq_len=2048, v1_style=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ff = FeedForward(hidden_dim, dropout, v1_style)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.ff(self.ln2(x))
        return x

# V1-V3: Simple head predicting OHLCV directly
class CandleTransformerV1(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=3, num_heads=2, patch_size=2, max_seq_len=32, pred_len=5):
        super().__init__()
        self.patch_embedding = PatchEmbedding(input_dim, hidden_dim, patch_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len // patch_size, hidden_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, num_heads, dropout=0.0, max_seq_len=max_seq_len, v1_style=True) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 10)  # Original V1: single head with 10 outputs
        self.patch_size = patch_size
        self.pred_len = pred_len
    def forward(self, x):
        B, T, C = x.shape
        x = self.patch_embedding(x)
        T_patch = x.shape[1]
        x = x + self.pos_embedding[:, :T_patch]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        last_hidden = x[:, -1]
        out = self.head(last_hidden)  # [B, 10]
        # Interpret as 2 steps of 5 features, or 1 step of 10
        # We'll just repeat for pred_len
        return out.unsqueeze(1).repeat(1, self.pred_len, 1)[:, :, :5]

# V3: Return-based with multi-head prediction (matching checkpoint architecture)
class CandleTransformerV3(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, num_layers=4, num_heads=4, patch_size=4, max_seq_len=128, pred_len=5):
        super().__init__()
        self.patch_embedding = PatchEmbedding(input_dim, hidden_dim, patch_size, use_ln=True)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len // patch_size, hidden_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, num_heads, dropout=0.1, max_seq_len=max_seq_len, v1_style=False) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.pred_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(pred_len)])
        self.patch_size = patch_size
        self.pred_len = pred_len
    def forward(self, x):
        B, T, C = x.shape
        x = self.patch_embedding(x)
        T_patch = x.shape[1]
        x = x + self.pos_embedding[:, :T_patch]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        last_hidden = x[:, -1]
        preds = []
        for head in self.pred_heads:
            preds.append(head(last_hidden))
        return torch.stack(preds, dim=1)  # [B, pred_len, 1]

# Add V3 model creation and loading
class CandleTransformerV5(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, num_layers=4, num_heads=4, patch_size=4, max_seq_len=128, pred_len=5):
        super().__init__()
        self.patch_embedding = PatchEmbedding(input_dim, hidden_dim, patch_size, use_ln=True)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len // patch_size, hidden_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, num_heads, dropout=0.1, max_seq_len=max_seq_len) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.pred_heads = nn.ModuleList([nn.Linear(hidden_dim, 3) for _ in range(pred_len)])
        self.patch_size = patch_size
        self.pred_len = pred_len
    def forward(self, x):
        B, T, C = x.shape
        x = self.patch_embedding(x)
        T_patch = x.shape[1]
        x = x + self.pos_embedding[:, :T_patch]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        last_hidden = x[:, -1]
        preds = []
        for head in self.pred_heads:
            preds.append(head(last_hidden))
        return torch.stack(preds, dim=1)

# V6: Probabilistic (mean + std)
class CandleTransformerV6(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, num_layers=4, num_heads=4, patch_size=4, max_seq_len=128, pred_len=5):
        super().__init__()
        self.patch_embedding = PatchEmbedding(input_dim, hidden_dim, patch_size, use_ln=True)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len // patch_size, hidden_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, num_heads, dropout=0.1, max_seq_len=max_seq_len) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.pred_head = nn.Linear(hidden_dim, pred_len * 3 * 2)
        self.pred_len = pred_len
        self.patch_size = patch_size
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

# ============================================================
# DATA LOADING & FEATURES
# ============================================================

csv_path = r'C:\Users\Mann Patel\Downloads\kronos_making\XAUUSD_M1_202101040105_202512312324.csv'
df = pd.read_csv(csv_path, sep='\t')
df.columns = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'VOL', 'SPREAD']

ohlcv = df[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL']].values.astype(np.float32)
dates = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y.%m.%d %H:%M:%S')

# V1 features (5 dim: OHLCV only - raw model input)
features_v1 = ohlcv.astype(np.float32)
mean_v1 = features_v1.mean(axis=0)
std_v1 = features_v1.std(axis=0) + 1e-8
features_norm_v1 = ((features_v1 - mean_v1) / std_v1).astype(np.float32)

# V3/V5/V6 features (14 dim: stationary indicators)
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

features_v3 = np.column_stack([
    ret_1.reshape(-1, 1), ret_5.reshape(-1, 1), hl_range.reshape(-1, 1),
    oc_range.reshape(-1, 1), vol, vol_20.reshape(-1, 1), rsi.reshape(-1, 1),
    macd.reshape(-1, 1), macd_signal.reshape(-1, 1), macd_hist.reshape(-1, 1),
    bb_pos.reshape(-1, 1), atr_norm.reshape(-1, 1), hour.reshape(-1, 1), day_of_week.reshape(-1, 1),
])
mean_v3 = features_v3.mean(axis=0)
std_v3 = features_v3.std(axis=0) + 1e-8
features_norm_v3 = ((features_v3 - mean_v3) / std_v3).astype(np.float32)

# ============================================================
# LOAD CHECKPOINTS
# ============================================================

print("Loading model checkpoints...")

# V1
ckpt_v1 = torch.load('candle_transformer.pth', map_location=device, weights_only=False)
model_v1 = CandleTransformerV1(input_dim=5, hidden_dim=128, num_layers=3, num_heads=2, patch_size=2, max_seq_len=32, pred_len=5).to(device)
# Remove mask keys from checkpoint before loading
ckpt_v1_filtered = {k: v for k, v in ckpt_v1.items() if 'attn.mask' not in k}
model_v1.load_state_dict(ckpt_v1_filtered)
model_v1.eval()

# V3
ckpt_v3 = torch.load('candle_transformer_v3_best.pth', map_location=device, weights_only=False)
# Remove mask keys
ckpt_v3_filtered = {k: v for k, v in ckpt_v3.items() if 'attn.mask' not in k}
model_v3 = CandleTransformerV3(input_dim=14, hidden_dim=128, num_layers=4, num_heads=4, patch_size=4, max_seq_len=128, pred_len=5).to(device)
model_v3.load_state_dict(ckpt_v3_filtered)
model_v3.eval()
target_mean_v3 = np.zeros(5)
target_std_v3 = np.ones(5)

# V5
ckpt_v5 = torch.load('candle_transformer_v5_best.pth', map_location=device, weights_only=False)
ckpt_v5_filtered = {k: v for k, v in ckpt_v5['model'].items() if 'attn.mask' not in k}
model_v5 = CandleTransformerV5(input_dim=14, hidden_dim=128, num_layers=4, num_heads=4, patch_size=4, max_seq_len=128, pred_len=5).to(device)
model_v5.load_state_dict(ckpt_v5_filtered)
model_v5.eval()
target_mean_v5 = ckpt_v5['target_mean']
target_std_v5 = ckpt_v5['target_std']

# V6
ckpt_v6 = torch.load('candle_transformer_v6_best.pth', map_location=device, weights_only=False)
ckpt_v6_filtered = {k: v for k, v in ckpt_v6['model'].items() if 'attn.mask' not in k}
model_v6 = CandleTransformerV6(input_dim=14, hidden_dim=128, num_layers=4, num_heads=4, patch_size=4, max_seq_len=128, pred_len=5).to(device)
model_v6.load_state_dict(ckpt_v6_filtered)
model_v6.eval()
target_mean_v6 = ckpt_v6['target_mean']
target_std_v6 = ckpt_v6['target_std']

print("All models loaded!")

# ============================================================
# RECONSTRUCTION FUNCTIONS
# ============================================================

def reconstruct_v1(pred_ohlcv):
    """V1 predicts OHLCV directly (normalized)"""
    return pred_ohlcv[0], pred_ohlcv[1], pred_ohlcv[2], pred_ohlcv[3]

def reconstruct_v3(pred_ret):
    """V3 predicts log-returns, reconstruct from last close"""
    pred_c = pred_ret[0]  # denormalized return
    # This is simplified - V3 used different approach
    return pred_c, pred_c, pred_c, pred_c  # placeholder

def reconstruct_v5(last_o, last_h, last_l, last_c, pred_components):
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

def reconstruct_v6(last_o, last_h, last_l, last_c, pred_components):
    return reconstruct_v5(last_o, last_h, last_l, last_c, pred_components)

# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

seq_len_v1 = 32
seq_len_v3 = 128
pred_len = 5
TEMPERATURE = 0.3

def predict_v1(start_idx):
    hist_start = start_idx - seq_len_v1
    context = features_norm_v1[hist_start:start_idx]
    context_tensor = torch.from_numpy(context).unsqueeze(0).to(device, dtype=torch.float32)
    with torch.no_grad():
        pred = model_v1(context_tensor).cpu().numpy()[0]  # [5, 5]
    pred_denorm = pred * std_v1[:5] + mean_v1[:5]
    predictions = []
    for step in range(pred_len):
        o, h, l, c, _ = pred_denorm[step]
        predictions.append({'open': float(o), 'high': float(h), 'low': float(l), 'close': float(c)})
    return predictions

def predict_v3(start_idx):
    hist_start = start_idx - seq_len_v3
    context = features_norm_v3[hist_start:start_idx]
    context_tensor = torch.from_numpy(context).unsqueeze(0).to(device, dtype=torch.float32)
    with torch.no_grad():
        pred = model_v3(context_tensor).cpu().numpy()[0]  # [pred_len, 1]
    pred_denorm = pred * target_std_v3 + target_mean_v3
    predictions = []
    last_c = ohlcv[start_idx - 1, 3]
    for step in range(pred_len):
        ret = pred_denorm[step, 0]  # Extract scalar
        pred_c = last_c * np.exp(ret)
        pred_hl = 0.5 * (np.exp(np.abs(ret)) - 1) * last_c  # rough estimate
        pred_o = last_c
        pred_h = max(pred_c, pred_o) + pred_hl
        pred_l = min(pred_c, pred_o) - pred_hl
        predictions.append({'open': float(pred_o), 'high': float(pred_h), 'low': float(pred_l), 'close': float(pred_c)})
        last_c = pred_c
    return predictions

def predict_v5(start_idx):
    hist_start = start_idx - seq_len_v3
    context = features_norm_v3[hist_start:start_idx]
    context_tensor = torch.from_numpy(context).unsqueeze(0).to(device, dtype=torch.float32)
    with torch.no_grad():
        pred = model_v5(context_tensor).cpu().numpy()[0]
    pred_denorm = pred * target_std_v5 + target_mean_v5
    predictions = []
    last_o = ohlcv[start_idx - 1, 0]
    last_h = ohlcv[start_idx - 1, 1]
    last_l = ohlcv[start_idx - 1, 2]
    last_c = ohlcv[start_idx - 1, 3]
    for step in range(pred_len):
        o, h, l, c = reconstruct_v5(last_o, last_h, last_l, last_c, pred_denorm[step])
        predictions.append({'open': float(o), 'high': float(h), 'low': float(l), 'close': float(c)})
        last_o, last_h, last_l, last_c = o, h, l, c
    return predictions

def predict_v6(start_idx):
    hist_start = start_idx - seq_len_v3
    context = features_norm_v3[hist_start:start_idx]
    context_tensor = torch.from_numpy(context).unsqueeze(0).to(device, dtype=torch.float32)
    with torch.no_grad():
        sampled, _, _ = model_v6.sample(context_tensor, temperature=TEMPERATURE)
        pred = sampled.cpu().numpy()[0]
    pred_denorm = pred * target_std_v6 + target_mean_v6
    predictions = []
    last_o = ohlcv[start_idx - 1, 0]
    last_h = ohlcv[start_idx - 1, 1]
    last_l = ohlcv[start_idx - 1, 2]
    last_c = ohlcv[start_idx - 1, 3]
    for step in range(pred_len):
        o, h, l, c = reconstruct_v6(last_o, last_h, last_l, last_c, pred_denorm[step])
        predictions.append({'open': float(o), 'high': float(h), 'low': float(l), 'close': float(c)})
        last_o, last_h, last_l, last_c = o, h, l, c
    return predictions

# ============================================================
# GET RANDOM CHUNK WITH ALL PREDICTIONS
# ============================================================

def get_all_predictions():
    max_start = len(ohlcv) - seq_len_v3 - pred_len - 1
    if max_start <= seq_len_v3:
        return None
    start = random.randint(seq_len_v3, max_start)
    
    hist_start = start - seq_len_v3
    hist_end = start
    
    # Historical (64 bars for display)
    historical = []
    for i in range(start - 64, start):
        historical.append({
            'time': int(dates.iloc[i].timestamp()),
            'open': float(ohlcv[i, 0]), 'high': float(ohlcv[i, 1]),
            'low': float(ohlcv[i, 2]), 'close': float(ohlcv[i, 3]),
            'volume': float(ohlcv[i, 4])
        })
    
    # Ground truth
    ground_truth = []
    for i in range(start, start + pred_len):
        ground_truth.append({
            'time': int(dates.iloc[i].timestamp()),
            'open': float(ohlcv[i, 0]), 'high': float(ohlcv[i, 1]),
            'low': float(ohlcv[i, 2]), 'close': float(ohlcv[i, 3]),
            'volume': float(ohlcv[i, 4])
        })
    
    # All model predictions
    preds_v1 = predict_v1(start)
    preds_v3 = predict_v3(start)
    preds_v5 = predict_v5(start)
    preds_v6 = predict_v6(start)
    
    def add_time(preds, start_idx):
        return [{'time': int(dates.iloc[start_idx + i].timestamp()), **p} for i, p in enumerate(preds)]
    
    return {
        'historical': historical,
        'ground_truth': ground_truth,
        'V1': add_time(preds_v1, start),
        'V3': add_time(preds_v3, start),
        'V5': add_time(preds_v5, start),
        'V6': add_time(preds_v6, start),
        'timestamp': datetime.now().isoformat()
    }

# ============================================================
# FASTAPI APP
# ============================================================

connected_clients = set()

async def broadcast_predictions():
    while True:
        try:
            data = get_all_predictions()
            if data:
                message = json.dumps(data)
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
        data = get_all_predictions()
        if data:
            await websocket.send_text(json.dumps(data))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.discard(websocket)

@app.get("/")
async def root():
    return HTMLResponse(html_content)

# ============================================================
# HTML WITH MULTIPLE CHARTS
# ============================================================

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>XAUUSD M1 - All Models Comparison</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        body { margin: 0; padding: 20px; background: #131722; color: #d1d4dc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .container { max-width: 1800px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 5px; font-weight: 500; font-size: 28px; }
        .subtitle { text-align: center; margin-bottom: 20px; color: #787b86; font-size: 14px; }
        .metrics { display: flex; justify-content: center; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }
        .metric { background: #1e222d; padding: 8px 16px; border-radius: 8px; text-align: center; min-width: 80px; }
        .metric-value { font-size: 18px; font-weight: bold; color: #0ecb8e; }
        .metric-label { font-size: 11px; color: #787b86; }
        .status { text-align: center; margin-bottom: 20px; padding: 10px; background: #1e222d; border-radius: 8px; font-size: 13px; }
        .status.connected { border-left: 4px solid #089981; }
        .status.disconnected { border-left: 4px solid #f23645; }
        .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .chart-container { background: #1e222d; border-radius: 8px; padding: 15px; }
        .chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #2a2e39; }
        .chart-title { font-size: 16px; font-weight: 500; }
        .chart-info { font-size: 12px; color: #787b86; }
        .chart-canvas { width: 100%; height: 380px; }
        .legend { display: flex; justify-content: center; gap: 15px; margin-top: 15px; flex-wrap: wrap; font-size: 12px; }
        .legend-item { display: flex; align-items: center; gap: 6px; }
        .legend-color { width: 16px; height: 10px; border-radius: 2px; }
        .historical-up { background: #089981; }
        .historical-down { background: #f23645; }
        .truth-up { background: #089981; opacity: 0.5; }
        .truth-down { background: #f23645; opacity: 0.5; }
        .pred-up { background: #0ecb8e; opacity: 0.8; }
        .pred-down { background: #ff5b6b; opacity: 0.8; }
        .last-update { text-align: center; margin-top: 15px; font-size: 12px; color: #787b86; }
        @media (max-width: 1200px) {
            .charts-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>XAUUSD M1 - All Models Comparison</h1>
        <div class="subtitle">V1→V6: From deterministic OHLCV to probabilistic 3-component forecasting</div>
        <div class="metrics">
            <div class="metric"><div class="metric-value" id="mseV1">--</div><div class="metric-label">V1 MSE</div></div>
            <div class="metric"><div class="metric-value" id="mseV3">--</div><div class="metric-label">V3 MSE</div></div>
            <div class="metric"><div class="metric-value" id="mseV5">--</div><div class="metric-label">V5 MSE</div></div>
            <div class="metric"><div class="metric-value" id="mseV6">--</div><div class="metric-label">V6 MSE</div></div>
            <div class="metric"><div class="metric-value" id="maeV6">--</div><div class="metric-label">V6 MAE</div></div>
            <div class="metric"><div class="metric-value" id="dirV6">--</div><div class="metric-label">V6 Dir%</div></div>
        </div>
        <div id="status" class="status disconnected">Connecting...</div>
        
        <div class="charts-grid">
            <!-- V1 Chart -->
            <div class="chart-container">
                <div class="chart-header">
                    <span class="chart-title">V1 - Original (64-dim, 2L, 2H, patch=2)</span>
                    <span class="chart-info">OHLCV direct | 9 features | 50K samples</span>
                </div>
                <div id="chartV1" class="chart-canvas"></div>
            </div>
            
            <!-- V3 Chart -->
            <div class="chart-container">
                <div class="chart-header">
                    <span class="chart-title">V3 - Return-Based (128-dim, 4L, 4H)</span>
                    <span class="chart-info">Log-returns | 14 features | Multi-head</span>
                </div>
                <div id="chartV3" class="chart-canvas"></div>
            </div>
            
            <!-- V5 Chart -->
            <div class="chart-container">
                <div class="chart-header">
                    <span class="chart-title">V5 - 3-Component Deterministic</span>
                    <span class="chart-info">close_ret + hl_ret + body_ret | 128/4/4</span>
                </div>
                <div id="chartV5" class="chart-canvas"></div>
            </div>
            
            <!-- V6 Chart -->
            <div class="chart-container">
                <div class="chart-header">
                    <span class="chart-title">V6 - Probabilistic (mean+std, temp=0.3)</span>
                    <span class="chart-info">Stochastic sampling | 3-component distribution</span>
                </div>
                <div id="chartV6" class="chart-canvas"></div>
            </div>
        </div>
        
        <div class="legend">
            <div class="legend-item"><div class="legend-color historical-up"></div><span>Historical (64 bars)</span></div>
            <div class="legend-item"><div class="legend-color truth-up"></div><span>Ground Truth</span></div>
            <div class="legend-item"><div class="legend-color pred-up"></div><span>Predicted (5 steps)</span></div>
        </div>
        <div class="last-update" id="lastUpdate">Last update: --</div>
    </div>

    <script>
            function initCharts() {
                // Model configurations
                const models = ['V1', 'V3', 'V5', 'V6'];
                const charts = {};
                const series = { hist: {}, truth: {}, pred: {} };
        
                // Create charts
                models.forEach(m => {
                    const chart = LightweightCharts.createChart(document.getElementById('chart' + m), {
                        width: document.getElementById('chart' + m).clientWidth,
                        height: 380,
                        layout: { background: { color: '#131722' }, textColor: '#d1d4dc' },
                        grid: { vertLines: { color: '#2a2e39' }, horzLines: { color: '#2a2e39' } },
                        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
                        rightPriceScale: { borderColor: '#2a2e39', autoScale: true },
                        timeScale: { borderColor: '#2a2e39', timeVisible: true, secondsVisible: false },
                    });
                    charts[m] = chart;
        
                    series.hist[m] = chart.addCandlestickSeries({
                        upColor: '#089981', downColor: '#f23645',
                        borderUpColor: '#089981', borderDownColor: '#f23645',
                        wickUpColor: '#089981', wickDownColor: '#f23645',
                    });
        
                    series.truth[m] = chart.addCandlestickSeries({
                        upColor: '#089981', downColor: '#f23645',
                        borderUpColor: '#089981', borderDownColor: '#f23645',
                        wickUpColor: '#089981', wickDownColor: '#f23645',
                    });
        
                    series.pred[m] = chart.addCandlestickSeries({
                        upColor: '#0ecb8e', downColor: '#ff5b6b',
                        borderUpColor: '#0ecb8e', borderDownColor: '#ff5b6b',
                        wickUpColor: '#0ecb8e', wickDownColor: '#ff5b6b',
                    });
                });
        
                const statusEl = document.getElementById('status');
                const lastUpdateEl = document.getElementById('lastUpdate');
                const mseEls = { V1: document.getElementById('mseV1'), V3: document.getElementById('mseV3'), V5: document.getElementById('mseV5'), V6: document.getElementById('mseV6') };
                const maeV6El = document.getElementById('maeV6');
                const dirV6El = document.getElementById('dirV6');
                let ws;
        
                function connect() {
                    ws = new WebSocket(`ws://${location.host}/ws`);
                    ws.onopen = () => {
                        statusEl.textContent = 'Connected - New random chunk every 60s (All 4 models)';
                        statusEl.className = 'status connected';
                    };
                    ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        updateAllCharts(data);
                        lastUpdateEl.textContent = `Last update: ${new Date(data.timestamp).toLocaleTimeString()}`;
                        if (data.ground_truth) {
                            calcAllMetrics(data.ground_truth, data);
                        }
                    };
                    ws.onclose = () => {
                        statusEl.textContent = 'Disconnected - Reconnecting...';
                        statusEl.className = 'status disconnected';
                        setTimeout(connect, 3000);
                    };
                    ws.onerror = () => ws.close();
                }
        
                function calcAllMetrics(truth, preds) {
                    const models = ['V1', 'V3', 'V5', 'V6'];
                    models.forEach(m => {
                        if (!preds[m]) return;
                        let mse = 0, mae = 0, dirCorrect = 0;
                        for (let i = 0; i < truth.length; i++) {
                            const diff = preds[m][i].close - truth[i].close;
                            mse += diff * diff;
                            mae += Math.abs(diff);
                            if (i > 0) {
                                const predDir = preds[m][i].close - preds[m][i-1].close;
                                const actualDir = truth[i].close - truth[i-1].close;
                                if ((predDir > 0 && actualDir > 0) || (predDir < 0 && actualDir < 0)) dirCorrect++;
                            }
                        }
                        mse /= truth.length;
                        mae /= truth.length;
                        mseEls[m].textContent = mse.toFixed(2);
                        if (m === 'V6') {
                            maeV6El.textContent = mae.toFixed(2);
                            const dirAcc = truth.length > 1 ? dirCorrect / (truth.length - 1) : 0;
                            dirV6El.textContent = (dirAcc * 100).toFixed(1) + '%';
                        }
                    });
                }
        
                function updateAllCharts(data) {
                                console.log('updateAllCharts called with:', Object.keys(data));
                                try {
                                    const histData = data.historical.map(c => ({
                                        time: c.time, open: c.open, high: c.high, low: c.low, close: c.close
                                    }));
                                    const truthData = data.ground_truth.map(c => ({
                                        time: c.time, open: c.open, high: c.high, low: c.low, close: c.close
                                    }));
            
                                    models.forEach(m => {
                                        if (!data[m]) return;
                                        const predData = data[m].map(c => ({
                                            time: c.time, open: c.open, high: c.high, low: c.low, close: c.close
                                        }));
                                        console.log('Setting data for', m, 'hist:', histData.length, 'truth:', truthData.length, 'pred:', predData.length);
                                        series.hist[m].setData(histData);
                                        series.truth[m].setData(truthData);
                                        series.pred[m].setData(predData);
                                        charts[m].timeScale().fitContent();
                                    });
                                    console.log('updateAllCharts completed');
                                } catch(e) {
                                    console.error('updateAllCharts error:', e);
                                }
                            }
            
                            // Expose for debugging
                            window.updateAllCharts = updateAllCharts;
            
                            window.addEventListener('resize', () => {
                    models.forEach(m => {
                        charts[m].applyOptions({ width: document.getElementById('chart' + m).clientWidth });
                    });
                });
        
                // Attach to window for WebSocket access
                window.charts = charts;
                window.series = series;
                window.models = models;
        
                connect();
            }
        
            // Run immediately if DOM is already loaded, otherwise wait for DOMContentLoaded
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', initCharts);
            } else {
                initCharts();
            }
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)