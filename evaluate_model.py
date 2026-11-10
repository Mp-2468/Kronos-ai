import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math

device = torch.device('cuda')

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
    def __init__(self, input_dim=9, hidden_dim=256, num_layers=6, num_heads=8, patch_size=4, max_seq_len=2048, output_dim=5):
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

seq_len = 64
model = CandleTransformer(input_dim=9, hidden_dim=256, num_layers=6, num_heads=8, patch_size=4, max_seq_len=64, output_dim=5).to(device)
model.load_state_dict(torch.load('candle_transformer_best.pth', map_location=device, weights_only=True))
model.eval()

n_chunks = 1000
n_steps = 5
errors = { 'mse': [], 'mae': [], 'dir_acc': [], 'close_mse': [] }

for chunk_idx in range(n_chunks):
    max_start = len(ohlcv) - seq_len - n_steps - 1
    start = np.random.randint(seq_len, max_start)
    hist_start = start - seq_len
    hist_end = start
    
    context = features_norm[hist_start:hist_end].copy()
    context_tensor = torch.from_numpy(context).unsqueeze(0).to(device, dtype=torch.float32)
    
    preds = []
    with torch.no_grad():
        for step in range(n_steps):
            pred_norm = model(context_tensor)
            next_step_norm = pred_norm[0, -1].cpu().numpy()
            next_step = next_step_norm * target_std + target_mean
            preds.append(next_step)
            next_ohlcv = next_step
            next_log_ret = np.log(next_step[3] / ohlcv[start - 1, 3]) if step == 0 else np.log(next_step[3] / preds[-1][3])
            next_hl_range = (next_step[1] - next_step[2]) / next_step[3]
            next_oc_range = abs(next_step[0] - next_step[3]) / next_step[3]
            next_vol = np.log1p(next_step[4])
            next_features = np.concatenate([next_ohlcv, [next_log_ret, next_hl_range, next_oc_range, next_vol]])
            next_features_norm = ((next_features - mean) / std).astype(np.float32)
            context_tensor = torch.cat([context_tensor[:, 1:], torch.from_numpy(next_features_norm).unsqueeze(0).unsqueeze(0).to(device)], dim=1)
    
    preds = np.array(preds)
    actual = ohlcv[start:start + n_steps]
    
    mse = np.mean((preds - actual) ** 2)
    mae = np.mean(np.abs(preds - actual))
    close_mse = np.mean((preds[:, 3] - actual[:, 3]) ** 2)
    
    pred_dir = np.sign(preds[:, 3] - preds[:, 0])
    actual_dir = np.sign(actual[:, 3] - actual[:, 0])
    dir_acc = np.mean(pred_dir == actual_dir)
    
    errors['mse'].append(mse)
    errors['mae'].append(mae)
    errors['close_mse'].append(close_mse)
    errors['dir_acc'].append(dir_acc)

print(f'Evaluation on {n_chunks} random chunks ({n_steps}-step ahead):')
print(f'  Overall MSE:  {np.mean(errors["mse"]):.4f} ± {np.std(errors["mse"]):.4f}')
print(f'  Overall MAE:  {np.mean(errors["mae"]):.4f} ± {np.std(errors["mae"]):.4f}')
print(f'  Close MSE:    {np.mean(errors["close_mse"]):.4f} ± {np.std(errors["close_mse"]):.4f}')
print(f'  Dir Accuracy: {np.mean(errors["dir_acc"]):.4f} ± {np.std(errors["dir_acc"]):.4f}')

step_errors = { 'mse': [[] for _ in range(n_steps)], 'mae': [[] for _ in range(n_steps)], 'dir': [[] for _ in range(n_steps)] }

for chunk_idx in range(200):
    max_start = len(ohlcv) - seq_len - n_steps - 1
    start = np.random.randint(seq_len, max_start)
    hist_start = start - seq_len
    hist_end = start
    
    context = features_norm[hist_start:hist_end].copy()
    context_tensor = torch.from_numpy(context).unsqueeze(0).to(device, dtype=torch.float32)
    
    preds = []
    with torch.no_grad():
        for step in range(n_steps):
            pred_norm = model(context_tensor)
            next_step_norm = pred_norm[0, -1].cpu().numpy()
            next_step = next_step_norm * target_std + target_mean
            preds.append(next_step)
            next_ohlcv = next_step
            next_log_ret = np.log(next_step[3] / ohlcv[start - 1, 3]) if step == 0 else np.log(next_step[3] / preds[-1][3])
            next_hl_range = (next_step[1] - next_step[2]) / next_step[3]
            next_oc_range = abs(next_step[0] - next_step[3]) / next_step[3]
            next_vol = np.log1p(next_step[4])
            next_features = np.concatenate([next_ohlcv, [next_log_ret, next_hl_range, next_oc_range, next_vol]])
            next_features_norm = ((next_features - mean) / std).astype(np.float32)
            context_tensor = torch.cat([context_tensor[:, 1:], torch.from_numpy(next_features_norm).unsqueeze(0).unsqueeze(0).to(device)], dim=1)
    
    preds = np.array(preds)
    actual = ohlcv[start:start + n_steps]
    
    for s in range(n_steps):
        step_errors['mse'][s].append(np.mean((preds[s] - actual[s]) ** 2))
        step_errors['mae'][s].append(np.mean(np.abs(preds[s] - actual[s])))
        pred_dir = np.sign(preds[s, 3] - preds[s, 0])
        actual_dir = np.sign(actual[s, 3] - actual[s, 0])
        step_errors['dir'][s].append(pred_dir == actual_dir)

print('\nPer-step breakdown:')
for s in range(n_steps):
    print(f'  Step {s+1}: MSE={np.mean(step_errors["mse"][s]):.4f}, MAE={np.mean(step_errors["mae"][s]):.4f}, DirAcc={np.mean(step_errors["dir"][s]):.4f}')