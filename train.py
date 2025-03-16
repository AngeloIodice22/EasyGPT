import os, re, math, json, torch
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.checkpoint import checkpoint
from safetensors.torch import save_file, load_file
from torch import nn
from tqdm import tqdm

# === 工具函數 ===
def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_texts(file_path):
    with open(file_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# === 配置類 ===
class ChatConfig:
    def __init__(self, vocab_size, max_seq_length, hidden_size=256, num_layers=4, num_heads=8, rope_dim=32, 
            feed_forward_dim=640, window_size=512, dropout=0.1, num_experts=4, expert_loss_weight=0.01):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.rope_dim = rope_dim
        self.feed_forward_dim = feed_forward_dim
        self.window_size = window_size
        self.dropout = dropout
        self.num_experts = num_experts
        self.expert_loss_weight = expert_loss_weight

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        save_json(self.__dict__, os.path.join(path, "config.json"))

    @classmethod
    def from_pretrained(cls, path):
        return cls(**load_json(os.path.join(path, "config.json")))

# === ROPE 函數 ===
def apply_rotary(x, cos, sin, rope_dim):
    x_rot = x[..., :rope_dim].reshape(*x.shape[:-1], rope_dim // 2, 2)
    x1, x2 = x_rot[..., 0], x_rot[..., 1]
    x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).reshape(*x.shape[:-1], rope_dim)
    return torch.cat([x_rotated, x[..., rope_dim:]], dim=-1)

def rotary_embedding(q, k, rope_dim):
    B, num_heads, T, _ = q.shape
    theta = torch.arange(0, rope_dim, 2, dtype=q.dtype, device=q.device) / rope_dim
    inv_freq = 1.0 / (10000 ** theta)
    positions = torch.arange(T, device=q.device, dtype=q.dtype).unsqueeze(0)
    freqs = positions.unsqueeze(-1) * inv_freq
    cos = torch.cos(freqs).unsqueeze(1)
    sin = torch.sin(freqs).unsqueeze(1)
    return apply_rotary(q, cos, sin, rope_dim), apply_rotary(k, cos, sin, rope_dim)

# === 正弦位置編碼 ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(1)].unsqueeze(0))

    @staticmethod
    def get_adaptive_causal_mask(T, window_size, device):
        effective_window = min(window_size, T - 1) if T <= 1024 else min(window_size, T // 2)
        idx = torch.arange(T, device=device).unsqueeze(0)
        return (idx.repeat(T, 1) > idx.T) | ((idx.T - idx) > effective_window)

# === GEGLU 模塊 ===
class GEGLU(nn.Module):
    def __init__(self, hidden_size, feed_forward_dim, dropout):
        super().__init__()
        self.fc = nn.Linear(hidden_size, feed_forward_dim * 2)
        self.out = nn.Linear(feed_forward_dim, hidden_size)

    def forward(self, x):
        x1, x2 = self.fc(x).chunk(2, dim=-1)
        return self.out(x1 * F.gelu(x2))

# === MoE 前饋 ===
class MoEFeedForward(nn.Module):
    def __init__(self, hidden_size, expert_dim, dropout, num_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([GEGLU(hidden_size, expert_dim, dropout) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_size, num_experts)
        self.dropout = nn.Dropout(dropout)
        self.num_experts = num_experts

    def forward(self, x):
        gate_scores = torch.softmax(self.gate(x), dim=-1).unsqueeze(-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        output = self.dropout((gate_scores * expert_outputs).sum(dim=2))
        avg_gate = gate_scores.mean(dim=(0, 1))
        utilization_loss = ((avg_gate - 1.0 / self.num_experts) ** 2).sum()
        return output, utilization_loss

# === 自注意力模塊 ===
class SelfAttention(nn.Module):
    def __init__(self, config, global_token_indices=None, allow_global_bidirectional=False):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.rope_dim = config.rope_dim
        self.qkv_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim * 3)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.allow_global_bidirectional = allow_global_bidirectional
        self.global_token_indices = global_token_indices if global_token_indices is not None else [2, 3, 4, 5]
        nn.init.xavier_normal_(self.qkv_proj.weight)
        nn.init.constant_(self.qkv_proj.bias, 0)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                   for t in torch.chunk(qkv, 3, dim=-1)]
        if self.rope_dim > 0:
            q, k = rotary_embedding(q, k, self.rope_dim)
        global_mask = torch.zeros(T, dtype=torch.bool, device=x.device)
        valid_indices = [idx for idx in self.global_token_indices if idx < T]
        if valid_indices:
            global_mask[valid_indices] = True
        attn_local = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask_val = torch.finfo(attn_local.dtype).min if attn_local.dtype == torch.float16 else -1e9
        if mask is not None:
            attn_local = attn_local.masked_fill(mask, mask_val)
        attn_local = self.attn_dropout(torch.softmax(attn_local, dim=-1))
        out_local = torch.matmul(attn_local, v)
        global_q = q[:, :, global_mask]
        attn_global = torch.matmul(global_q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if not self.allow_global_bidirectional:
            pos = torch.nonzero(global_mask, as_tuple=False).squeeze(-1)
            row_idx = pos.view(1, 1, -1, 1)
            col_idx = torch.arange(T, device=x.device).view(1, 1, 1, T)
            attn_global = attn_global.masked_fill(col_idx > row_idx, mask_val)
        attn_global = self.attn_dropout(torch.softmax(attn_global, dim=-1))
        out_local[:, :, global_mask] = torch.matmul(attn_global, v)
        attn_out = out_local.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(attn_out)

# === Transformer 塊 ===
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.self_attn = SelfAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.ffn = MoEFeedForward(config.hidden_size, config.feed_forward_dim, config.dropout, config.num_experts)
        self.dropout = nn.Dropout(config.dropout)

    def _forward_impl(self, x, mask):
        x = x + self.dropout(self.self_attn(self.norm1(x), mask))
        ffn_out, moe_loss = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out), moe_loss

    def forward(self, x, mask=None):
        return checkpoint(self._forward_impl, x, mask, use_reentrant=False)

# === 聊天模型 ===
class ChatModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.config = config

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed(input_ids)
        B, T, _ = x.size()
        causal_mask = PositionalEncoding.get_adaptive_causal_mask(T, self.config.window_size, x.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        if attention_mask is not None:
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2).expand(B, 1, T, T)
            mask = causal_mask | pad_mask
        else:
            mask = causal_mask
        moe_losses = 0
        for block in self.blocks:
            x, loss = block(x, mask)
            moe_losses += loss
        moe_losses /= len(self.blocks)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)
            loss += self.config.expert_loss_weight * moe_losses
        return {"loss": loss, "logits": logits}

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        model_to_save = self.module if hasattr(self, 'module') else self
        save_file(model_to_save.state_dict(), os.path.join(path, "model.safetensors"))

    @classmethod
    def from_pretrained(cls, model_path, config):
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        model = cls(config)
        model.load_state_dict(new_state_dict)
        return model

# === 標記器 ===
class ChatTokenizer:
    def __init__(self, vocab_path=None):
        self.vocab = {"<|padding|>": 0, "<|unknown|>": 1, "<|user|>": 2,
        "<|think|>": 3,  "<|assistant|>": 4, "<|end|>": 5, " ": 6, "\\n": 7}
        self.special_tokens = sorted(self.vocab.keys(), key=lambda k: self.vocab[k])
        self.pattern = re.compile(
            f'({"|".join(map(re.escape, self.special_tokens))})'
            r'|([\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\U00020000-\U0002EBEF])'
            r'|([a-zA-Z]+)' r'|( +)' r'|([0-9])' r'|(_)' r'|([^\s])'
            r'|([!@#$%^&*()\-+=\[\]{}\\|;:\'",.<>/?`~])', re.UNICODE)
        if vocab_path and os.path.exists(vocab_path):
            self.load(vocab_path)

    def tokenize(self, text):
        return [m.group() for m in self.pattern.finditer(text)]

    def convert_tokens_to_ids(self, tokens, update_vocab=True):
        return [self.vocab.setdefault(token, len(self.vocab)) if token not in self.vocab and update_vocab
                else self.vocab.get(token, self.vocab["<|unknown|>"]) for token in tokens]

    def build_vocab(self, texts, max_vocab_size=None):
        token_freq = Counter(token for text in texts for token in self.tokenize(text) if token not in self.special_tokens)
        sorted_tokens = sorted(token_freq.items(), key=lambda x: (-x[1], x[0]))
        if max_vocab_size:
            sorted_tokens = sorted_tokens[:max_vocab_size - len(self.special_tokens)]
        self.vocab.update({token: idx + len(self.special_tokens) for idx, (token, _) in enumerate(sorted_tokens)})

    def __call__(self, text, max_length, truncation=True, padding="max_length", update_vocab=False):
        ids = self.convert_tokens_to_ids(self.tokenize(text), update_vocab)
        ids = ids[:max_length] if truncation else ids
        ids += [self.vocab["<|padding|>"]] * (max_length - len(ids))
        mask = [1 if i != self.vocab["<|padding|>"] else 0 for i in ids]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        save_json({"vocab": self.vocab}, os.path.join(path, "tokenizer.json"))

    def load(self, path):
        self.vocab = load_json(os.path.join(path, "tokenizer.json"))["vocab"]

    @property
    def reverse_vocab(self):
        return {i: token for token, i in self.vocab.items()}

    @classmethod
    def from_pretrained(cls, path):
        tokenizer = cls()
        tokenizer.load(path)
        return tokenizer

# === 數據集 ===
class ChatDataset(Dataset):
    def __init__(self, tokenizer, max_length, texts):
        self.data = [text for text in texts if text and len(tokenizer.tokenize(text)) > 1]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.data[idx], self.max_length, update_vocab=False)
        input_ids = encoding["input_ids"].squeeze()
        return {
            "input_ids": input_ids[:-1],
            "attention_mask": encoding["attention_mask"].squeeze()[:-1],
            "labels": input_ids[1:]}

# === 訓練週期 ===
def run_epoch(model, data_loader, device, pad_id, desc, optimizer=None, accum_steps=2):
    total_loss, total_correct, total_tokens = 0, 0, 0
    scaler = torch.amp.GradScaler() if optimizer is not None else None
    for step, batch in enumerate(tqdm(data_loader, desc=desc, leave=False)):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if optimizer:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(**batch)
                loss = outputs["loss"] / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs["loss"]
        total_loss += loss.item() * (accum_steps if optimizer else 1)
        preds = outputs["logits"].argmax(dim=-1)
        mask = batch["labels"] != pad_id
        total_correct += ((preds == batch["labels"]) * mask).sum().item()
        total_tokens += mask.sum().item()
    return total_loss / len(data_loader), total_correct / total_tokens if total_tokens > 0 else 0

# === 訓練階段 ===
def stage_train(stage_name, model, tokenizer, config, file_data, epochs=30, batch_size=4, val_split=0.1):
    bs = batch_size * (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    texts = read_texts(file_data)
    dataset = ChatDataset(tokenizer, config.max_seq_length, texts)
    indices = torch.randperm(len(dataset)).tolist()
    split_idx = int(len(dataset) * (1 - val_split))
    train_data, val_data = Subset(dataset, indices[:split_idx]), Subset(dataset, indices[split_idx:])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(8, os.cpu_count() or 1)
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, persistent_workers=True,
    num_workers=num_workers, pin_memory=True, pin_memory_device=str(device), prefetch_factor=3)
    val_loader = DataLoader(val_data, batch_size=bs, shuffle=False, persistent_workers=True,
    num_workers=num_workers, pin_memory=True, pin_memory_device=str(device), prefetch_factor=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    pad_id = tokenizer.vocab["<|padding|>"]

    for epoch in range(epochs):
        model.train()
        ep_str = f"{stage_name} EP: {epoch+1}/{epochs}"
        train_loss, train_acc = run_epoch(model, train_loader, device, pad_id, f"{ep_str}, Training", optimizer)
        model.eval()
        val_loss, val_acc = run_epoch(model, val_loader, device, pad_id, f"{ep_str}, Validation")
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{ep_str}, LR: {current_lr:.6f}, TL: {train_loss:.6f}, TA: {train_acc:.6f}, VL: {val_loss:.6f}, VA: {val_acc:.6f}")
        save_path = os.path.join("./model", f"{stage_name}_epoch_{epoch+1}")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save(save_path)
        config.save_pretrained(save_path)

if __name__ == "__main__":
    file_data = "./data/dialogues.txt"
    tokenizer = ChatTokenizer()
    tokenizer.build_vocab(read_texts(file_data))
    config = ChatConfig(vocab_size=len(tokenizer.vocab), max_seq_length=1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    model = ChatModel(config)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    model = model.to(device)
    stage_train("Pretrain", model, tokenizer, config, file_data)
