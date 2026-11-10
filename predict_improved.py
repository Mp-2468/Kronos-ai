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

class CandleTransformer(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=256, num_layers=6, num_heads=8, patch_size=4, max_seq_len=2048, output_dim=5):
        super().__init__()
        self.patch_embedding = PatchEmbedding(input_dim, hidden_dim, patch_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len // patch_size, hidden_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim * patch_size)
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.input_dim = input_dim
    def forward(self, x):
        B, T, C = x.shape
        x = self.patch_embedding(x)
        T_patch = x.shape[1]
        x = x + self.pos_embedding[:, :T_patch]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.head(x)
        x = x.reshape(B, T_patch, self.patch_size, self.output_dim)
        x = x.reshape(B, T_patch * self.patch_size, self.output_dim)
        return x[:, :T]

csv_path = r'C:\Users\Mann Patel\Downloads\kronos_making\XAUUSD_M1_202101040105_202512312324.csv'
df = pd.read_csv(csv_path, sep='\t')
df.columns = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'VOL', 'SPREAD']

ohlcv = df[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL']].values.astype(np.float32)
log_ret = np.log(ohlcv[1:, 3] / ohlcv[:-1, 3])
log_ret = np.concatenate([[0], log_ret])
hl_range = (ohlcv[:, 1] - ohlcv[:, 2]) / ohlcv[:, 3]
oc_range = np.abs(ohlcv[:, 0] - ohlcv[:, 3]) / ohlcv[:, 3]
vol = np.log1p(ohlcv[:, 4:5])

features = np.column_stack([ohlcv, log_ret.reshape(-1, 1), hl_range.reshape(-1, 1), oc_range.reshape(-1, 1), vol])

mean = features.mean(axis=0)
std = features.std(axis=0) + 1e-8
features_norm = ((features - mean) / std).astype(np.float32)

target_mean = mean[:5]
target_std = std[:5]

dates = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y.%m.%d %H:%M:%S')

seq_len = 64
model = CandleTransformer(input_dim=9, hidden_dim=256, num_layers=6, num_heads=8, patch_size=4, max_seq_len=64, output_dim=5).to(device)
model.load_state_dict(torch.load('candle_transformer_best.pth', map_location=device, weights_only=True))
model.eval()

context = features_norm[-seq_len:].copy()
context_tensor = torch.from_numpy(context).unsqueeze(0).to(device, dtype=torch.float32)
last_dt = dates.iloc[-1]

n_steps = 5
predictions = []

with torch.no_grad():
    for step in range(n_steps):
        pred_norm = model(context_tensor)
        next_step_norm = pred_norm[0, -1].cpu().numpy()
        next_step = next_step_norm * target_std + target_mean
        predictions.append(next_step)
        # Build next feature vector for autoregressive step (9 features matching training)
        next_ohlcv = next_step
        next_log_ret = np.log(next_step[3] / ohlcv[-1, 3]) if step == 0 else np.log(next_step[3] / predictions[-2][3])
        next_hl_range = (next_step[1] - next_step[2]) / next_step[3]
        next_oc_range = abs(next_step[0] - next_step[3]) / next_step[3]
        next_vol = np.log1p(next_step[4])
        next_features = np.concatenate([next_ohlcv, [next_log_ret, next_hl_range, next_oc_range, next_vol]])
        next_features_norm = ((next_features - mean) / std).astype(np.float32)
        context_tensor = torch.cat([context_tensor[:, 1:], torch.from_numpy(next_features_norm).unsqueeze(0).unsqueeze(0).to(device)], dim=1)

print('5-Step Autoregressive Forecast (Improved Model):')
for i, p in enumerate(predictions):
    next_dt = last_dt + timedelta(minutes=i+1)
    print(f'  {next_dt.strftime("%Y.%m.%d %H:%M:%S")} | O={p[0]:.5f} H={p[1]:.5f} L={p[2]:.5f} C={p[3]:.5f} V={p[4]:.2f}')