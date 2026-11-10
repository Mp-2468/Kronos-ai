import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

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

checkpoint = torch.load('candle_transformer_v6_best.pth', map_location=device, weights_only=False)
target_mean = checkpoint['target_mean']
target_std = checkpoint['target_std']

seq_len = 128
pred_len = 5
model = CandleTransformerV6(input_dim=14, hidden_dim=128, num_layers=4, num_heads=4, patch_size=4, max_seq_len=128, pred_len=pred_len).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

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

test_start = len(ohlcv) - 200 - seq_len
n_tests = 200

errors = { 'mse': [], 'mae': [], 'dir_acc': [] }
step_errors = { 'mse': [[] for _ in range(pred_len)], 'mae': [[] for _ in range(pred_len)], 'dir': [[] for _ in range(pred_len)] }

print("Testing V6 model (probabilistic: mean+std, temperature=0.5)...")

for i in range(n_tests):
    start = test_start + i
    hist_start = start - seq_len
    hist_end = start
    
    context = features_norm[hist_start:hist_end].copy()
    context_tensor = torch.from_numpy(context).unsqueeze(0).to(device, dtype=torch.float32)
    
    with torch.no_grad():
        sampled, pred_mean, pred_std = model.sample(context_tensor, temperature=0.5)
        pred_components = sampled.cpu().numpy()[0]  # [pred_len, 3]
    
    # Denormalize
    pred_components = pred_components * target_std + target_mean
    
    pred_prices = []
    last_o = ohlcv[start - 1, 0]
    last_h = ohlcv[start - 1, 1]
    last_l = ohlcv[start - 1, 2]
    last_c = ohlcv[start - 1, 3]
    
    for step in range(pred_len):
        pred_o, pred_h, pred_l, pred_c = reconstruct_ohlc(
            last_o, last_h, last_l, last_c, pred_components[step]
        )
        pred_prices.append([pred_o, pred_h, pred_l, pred_c])
        last_o, last_h, last_l, last_c = pred_o, pred_h, pred_l, pred_c
    
    actual_prices = ohlcv[start:start + pred_len, :4]
    
    mse = np.mean((np.array(pred_prices) - actual_prices) ** 2)
    mae = np.mean(np.abs(np.array(pred_prices) - actual_prices))
    
    pred_dir = np.sign(np.array([p[3] - p[0] for p in pred_prices]))
    actual_dir = np.sign(actual_prices[:, 3] - actual_prices[:, 0])
    dir_acc = np.mean(pred_dir == actual_dir)
    
    errors['mse'].append(mse)
    errors['mae'].append(mae)
    errors['dir_acc'].append(dir_acc)
    
    for s in range(pred_len):
        step_errors['mse'][s].append((pred_prices[s][3] - actual_prices[s, 3]) ** 2)
        step_errors['mae'][s].append(abs(pred_prices[s][3] - actual_prices[s, 3]))
        pred_d = np.sign(pred_prices[s][3] - pred_prices[s][0])
        actual_d = np.sign(actual_prices[s, 3] - actual_prices[s, 0])
        step_errors['dir'][s].append(pred_d == actual_d)

print(f'\nV6 Model - Evaluation on {n_tests} sequential tests (temp=0.5):')
print(f'  Overall MSE:  {np.mean(errors["mse"]):.4f} ± {np.std(errors["mse"]):.4f}')
print(f'  Overall MAE:  {np.mean(errors["mae"]):.4f} ± {np.std(errors["mae"]):.4f}')
print(f'  Dir Accuracy: {np.mean(errors["dir_acc"]):.4f} ± {np.std(errors["dir_acc"]):.4f}')

print('\nPer-step breakdown:')
for s in range(pred_len):
    print(f'  Step {s+1}: MSE={np.mean(step_errors["mse"][s]):.4f}, MAE={np.mean(step_errors["mae"][s]):.4f}, DirAcc={np.mean(step_errors["dir"][s]):.4f}')

# Test different temperatures
for temp in [0.3, 0.7, 1.0]:
    temp_errors = []
    for i in range(min(50, n_tests)):
        start = test_start + i
        hist_start = start - seq_len
        hist_end = start
        context = features_norm[hist_start:hist_end].copy()
        context_tensor = torch.from_numpy(context).unsqueeze(0).to(device, dtype=torch.float32)
        
        with torch.no_grad():
            sampled, _, _ = model.sample(context_tensor, temperature=temp)
            pred_components = sampled.cpu().numpy()[0]
        
        pred_components = pred_components * target_std + target_mean
        pred_prices = []
        last_o = ohlcv[start - 1, 0]
        last_h = ohlcv[start - 1, 1]
        last_l = ohlcv[start - 1, 2]
        last_c = ohlcv[start - 1, 3]
        
        for step in range(pred_len):
            pred_o, pred_h, pred_l, pred_c = reconstruct_ohlc(last_o, last_h, last_l, last_c, pred_components[step])
            pred_prices.append([pred_o, pred_h, pred_l, pred_c])
            last_o, last_h, last_l, last_c = pred_o, pred_h, pred_l, pred_c
        
        actual_prices = ohlcv[start:start + pred_len, :4]
        mse = np.mean((np.array(pred_prices) - actual_prices) ** 2)
        temp_errors.append(mse)
    
    print(f'\nTemperature {temp}: Mean MSE = {np.mean(temp_errors):.4f} ± {np.std(temp_errors):.4f}')

# Sample predictions
print('\nSample predictions (last 5 tests, temp=0.5):')
for i in range(max(0, n_tests-5), n_tests):
    start = test_start + i
    hist_start = start - seq_len
    hist_end = start
    context = features_norm[hist_start:hist_end].copy()
    context_tensor = torch.from_numpy(context).unsqueeze(0).to(device, dtype=torch.float32)
    with torch.no_grad():
        sampled, _, _ = model.sample(context_tensor, temperature=0.5)
        pred_components = sampled.cpu().numpy()[0]
    pred_components = pred_components * target_std + target_mean
    
    print(f"\nTest {i+1}:")
    pred_prices = []
    last_o = ohlcv[start - 1, 0]
    last_h = ohlcv[start - 1, 1]
    last_l = ohlcv[start - 1, 2]
    last_c = ohlcv[start - 1, 3]
    
    for step in range(pred_len):
        pred_o, pred_h, pred_l, pred_c = reconstruct_ohlc(last_o, last_h, last_l, last_c, pred_components[step])
        pred_prices.append([pred_o, pred_h, pred_l, pred_c])
        last_o, last_h, last_l, last_c = pred_o, pred_h, pred_l, pred_c
    
    actual_prices = ohlcv[start:start + pred_len, :4]
    for step in range(pred_len):
        print(f"  Step {step+1}: Pred O={pred_prices[step][0]:.2f} H={pred_prices[step][1]:.2f} L={pred_prices[step][2]:.2f} C={pred_prices[step][3]:.2f} | Actual O={actual_prices[step,0]:.2f} H={actual_prices[step,1]:.2f} L={actual_prices[step,2]:.2f} C={actual_prices[step,3]:.2f}")