import os, re, math, json, torch
import torch.nn.functional as F
from collections import Counter
from bitsandbytes.optim import Adam8bit
from torch.utils.data import Dataset, DataLoader, Subset
from safetensors.torch import save_file, load_file
from torch import nn
from tqdm import tqdm

# ===== 工具函數 =====
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

# ===== 配置類 =====
class ChatConfig:
    def __init__(self, vocab_size, max_seq_length, hidden_size=384, num_layers=3, num_heads=6, rope_dim=64, 
            feed_forward_dim=960, window_size=512, dropout=0.1, num_experts=4, expert_loss_weight=0.01):
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

# ===== ROPE 位置編碼 =====
class RotaryEmbedding(nn.Module):
    def __init__(self, rope_dim, max_seq_length=1024, base=10000):
        super().__init__()
        self.rope_dim = rope_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        self.register_buffer("inv_freq", inv_freq)
        positions = torch.arange(max_seq_length, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        self.register_buffer("cos_cache", torch.cos(freqs)[None, None, :, :])
        self.register_buffer("sin_cache", torch.sin(freqs)[None, None, :, :])

    def forward(self, x):
        B, num_heads, T, _ = x.shape
        return self.cos_cache[:, :, :T, :], self.sin_cache[:, :, :T, :]

def apply_rotary(x, cos, sin, rope_dim):
    x_rot = x[..., :rope_dim].reshape(*x.shape[:-1], rope_dim // 2, 2)
    x1, x2 = x_rot[..., 0], x_rot[..., 1]
    x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).reshape(*x.shape[:-1], rope_dim)
    return torch.cat([x_rotated, x[..., rope_dim:]], dim=-1)

# ===== 正弦位置編碼與漸進式視窗 =====
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
        i = torch.arange(T, device=device).unsqueeze(1)
        j = torch.arange(T, device=device).unsqueeze(0)
        mask = (j > i) | ((i - j) >= window_size)
        return mask

# ===== GEGLU 模塊 =====
class GEGLU(nn.Module):
    def __init__(self, hidden_size, feed_forward_dim, dropout):
        super().__init__()
        self.fc = nn.Linear(hidden_size, feed_forward_dim * 2)
        self.out = nn.Linear(feed_forward_dim, hidden_size)

    def forward(self, x):
        x1, x2 = self.fc(x).chunk(2, dim=-1)
        return self.out(x1 * F.gelu(x2))

# ===== MoE 前饋專家 =====
class MoEFeedForward(nn.Module):
    def __init__(self, hidden_size, expert_dim, dropout, num_experts, top_k=1):
        super().__init__()
        self.experts = nn.ModuleList([GEGLU(hidden_size, expert_dim, dropout) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_size, num_experts)
        self.dropout = nn.Dropout(dropout)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x):
        raw_gate = self.gate(x)
        gate_values, topk_indices = torch.topk(raw_gate, k=self.top_k, dim=-1)
        gate_scores = F.softmax(gate_values, dim=-1).squeeze(-1)
        output = torch.zeros_like(x)
        expert_usage = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)
        for expert_idx in range(self.num_experts):
            mask = (topk_indices.squeeze(-1) == expert_idx)
            if mask.any():
                x_expert = x[mask]
                expert_out = self.experts[expert_idx](x_expert)
                output[mask] = expert_out * gate_scores[mask].unsqueeze(-1)
                expert_usage[expert_idx] = gate_scores[mask].mean()
        utilization_loss = ((expert_usage - 1.0 / self.num_experts) ** 2).sum() + 1e-8
        return self.dropout(output), utilization_loss

# ===== 自注意力模塊 =====
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
        if self.rope_dim > 0:
            self.rotary_emb = RotaryEmbedding(self.rope_dim, max_seq_length=config.max_seq_length)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) 
                   for t in torch.chunk(qkv, 3, dim=-1)]
        if self.rope_dim > 0:
            cos, sin = self.rotary_emb(q)
            q = apply_rotary(q, cos, sin, self.rope_dim)
            k = apply_rotary(k, cos, sin, self.rope_dim)
        indices = torch.tensor(self.global_token_indices, device=x.device)
        valid_indices = indices[indices < T]
        global_mask = torch.zeros(T, dtype=torch.bool, device=x.device)
        if valid_indices.numel() > 0:
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

# ===== Transformer 塊 =====
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.self_attn = SelfAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.ffn = MoEFeedForward(config.hidden_size, config.feed_forward_dim, config.dropout, config.num_experts)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), mask))
        ffn_out, moe_loss = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out), moe_loss

# ===== 聊天模型 =====
class ChatModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
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

class ChatTokenizer:
    def __init__(self, vocab_path=None, use_bpe=True, split_length=8):
        self.use_bpe = use_bpe
        self.split_length = split_length
        self.vocab = {"<|padding|>": 0, "<|unknown|>": 1, "<|user|>": 2,
            "<|think|>": 3, "<|assistant|>": 4, "<|end|>": 5, " ": 6, "\\n": 7}
        self.special_tokens = sorted(self.vocab, key=self.vocab.get)
        self.pattern = re.compile(
            f'({"|".join(map(re.escape, self.special_tokens))})'
            r'|([a-zA-Z]+)|( +)|([0-9])|(_)|([^\s])', re.UNICODE)
        if vocab_path and os.path.exists(vocab_path):
            self.load(vocab_path)

    def build_trie(self, candidates):
        trie = {}
        for cand in candidates:
            node = trie
            for ch in cand:
                node = node.setdefault(ch, {})
            node['$'] = True
        return trie

    def smart_segment_with_trie(self, token, trie):
        n = len(token)
        dp = [None] * (n + 1)
        dp[0] = []
        for i in range(n):
            if dp[i] is None:
                continue
            node = trie
            for j in range(i, n):
                node = node.get(token[j])
                if node is None:
                    break
                if node.get('$'):
                    new_seg = dp[i] + [token[i:j+1]]
                    if dp[j+1] is None or len(new_seg) < len(dp[j+1]):
                        dp[j+1] = new_seg
            new_seg = dp[i] + [token[i]]
            if dp[i+1] is None or len(new_seg) < len(dp[i+1]):
                dp[i+1] = new_seg
        segmentation = dp[n]
        return segmentation if segmentation and segmentation != [token] else None

    def tokenize(self, text):
        pattern = self.pattern
        tokens = [m.group() for m in pattern.finditer(text)]
        if not self.use_bpe:
            return tokens
        candidates = {t for t in tokens if t.isalpha() and len(t) <= self.split_length}
        trie = self.build_trie(candidates)
        cache, output = {}, []
        for t in tokens:
            if t.isalpha() and len(t) > self.split_length:
                seg = cache.setdefault(t, self.smart_segment_with_trie(t, trie))
                output.extend(seg if seg else [t])
            else:
                output.append(t)
        return output

    def convert_tokens_to_ids(self, tokens, update_vocab=True):
        vocab = self.vocab
        unk = vocab["<|unknown|>"]
        if update_vocab:
            return [vocab.setdefault(t, len(vocab)) for t in tokens]
        else:
            return [vocab.get(t, unk) for t in tokens]

    def build_vocab(self, texts, max_vocab_size=None):
        total_text = 0
        freq = Counter()
        pbar = tqdm(texts, desc="Tokenizing")
        for text in pbar:
            for tok in self.tokenize(text):
                if tok not in self.special_tokens:
                    freq[tok] += 1
            total_text += len(text)
            pbar.set_postfix({"text": len(text), "texts": total_text, "tokens": len(freq)})
        tokens_sorted = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        if max_vocab_size:
            tokens_sorted = tokens_sorted[:max_vocab_size - len(self.special_tokens)]
        self.vocab.update({t: i + len(self.special_tokens) for i, (t, _) in enumerate(tokens_sorted)})

    def __call__(self, text, max_length, truncation=True, padding="max_length", update_vocab=False):
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens, update_vocab)
        if truncation:
            ids = ids[:max_length]
        if padding == "max_length":
            pad_id = self.vocab["<|padding|>"]
            ids += [pad_id] * (max_length - len(ids))
        mask = [1 if i != self.vocab["<|padding|>"] else 0 for i in ids]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        data = {"use_bpe": self.use_bpe, "split_length": self.split_length, "vocab": self.vocab}
        save_json(data, os.path.join(path, "tokenizer.json"))

    def load(self, path):
        data = load_json(os.path.join(path, "tokenizer.json"))
        self.vocab = data.get("vocab", self.vocab)
        self.use_bpe = data.get("use_bpe", True)
        self.split_length = data.get("split_length", 8)

    @property
    def reverse_vocab(self):
        return {i: t for t, i in self.vocab.items()}

    @classmethod
    def from_pretrained(cls, path):
        tokenizer = cls()
        tokenizer.load(path)
        return tokenizer

# ===== 數據集 =====
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

# ===== 訓練週期 =====
def run_epoch(model, data_loader, device, pad_id, desc, epoch, optimizer=None, accum_steps=1):
    total_loss, total_correct, total_tokens = 0, 0, 0
    scaler = torch.amp.GradScaler(device='cuda') if (optimizer is not None and device.type == "cuda") else None
    pbar = tqdm(data_loader, desc=desc)
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if optimizer is not None:
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(**batch)
                    loss = outputs["loss"] / accum_steps
            else:
                outputs = model(**batch)
                loss = outputs["loss"] / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs["loss"]
        total_loss += loss.item() * (accum_steps if optimizer is not None else 1)
        preds = outputs["logits"].argmax(dim=-1)
        mask = batch["labels"] != pad_id
        total_correct += ((preds == batch["labels"]) * mask).sum().item()
        total_tokens += mask.sum().item()
        batch_acc = ((preds == batch["labels"]) * mask).sum().item() / mask.sum().item()
        pbar.set_postfix({"epoch": f"{epoch+1}", "loss": f"{loss.item():.6f}", "acc": f"{batch_acc:.6f}"})
    return total_loss / len(data_loader), total_correct / total_tokens if total_tokens > 0 else 0

# ===== 訓練階段 =====
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
    num_workers=num_workers, pin_memory=True, pin_memory_device=str(device), prefetch_factor=5)
    val_loader = DataLoader(val_data, batch_size=bs, shuffle=False, persistent_workers=True,
    num_workers=num_workers, pin_memory=True, pin_memory_device=str(device), prefetch_factor=5)
    optimizer = Adam8bit(model.parameters(), lr=1e-3, weight_decay=0.1, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=1)
    pad_id = tokenizer.vocab["<|padding|>"]
    original_window = config.window_size

    for epoch in range(epochs):
        dynamic_window = max(1, int(original_window * (epoch + 1) / epochs))
        model.config.window_size = dynamic_window
        model.train()
        tl, ta = run_epoch(model, train_loader, device, pad_id, f"Training  ", epoch, optimizer)
        model.eval()
        vl, va = run_epoch(model, val_loader, device, pad_id, f"Verifying ", epoch)
        scheduler.step(vl)
        lr = optimizer.param_groups[0]['lr']
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
    stage_train("Pretrain", model.to(device), tokenizer, config, file_data)
