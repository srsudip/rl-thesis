"""
Transformer-based Pathway Recommender with LoRA Continual Learning
===================================================================

Architecture
------------
Each student's grade history is treated as a sequence of tokens:

    [CLS] [G4 scores] [G5 scores] ... [G9 scores]
      ↓         ↓           ↓              ↓
    Transformer Encoder (2 layers, 4 heads, d_model=64)
      ↓
    [CLS] representation
      ↓               ↓
  pathway head     action head
  (3 classes)     (9 classes)

LoRA Adapters (Low-Rank Adaptation)
------------------------------------
Applied to Q and V projections in every attention layer.
Rank r=8 means each LoRA adapter adds only 2×(64×8)=1,024 params per layer.

During normal training  : base weights + LoRA trained together.
During HITL fine-tuning : base weights FROZEN, only LoRA A/B matrices updated.
                          Teacher overrides cannot overwrite learned knowledge.

Comparison with DQN
--------------------
- DQN:         RL, one grade transition at a time, replay buffer for HITL
- Transformer: Supervised, full G4-G9 trajectory at once, LoRA for HITL
- Both expose the same recommend() interface so the dashboard can run both.
"""

import numpy as np
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    # Stub out nn.Module so class definitions at module level don't crash
    # on machines without PyTorch.  Any actual *use* raises ImportError.
    class _StubModule:
        def __init_subclass__(cls, **kwargs):
            pass
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch is required for Transformer model")

    class _StubNN:
        Module   = _StubModule
        Linear   = _StubModule
        LayerNorm = _StubModule
        Dropout  = _StubModule
        Embedding = _StubModule
        Parameter = object

        @staticmethod
        def utils(*a, **kw): pass

    class _StubF:
        @staticmethod
        def softmax(*a, **kw): pass
        @staticmethod
        def cross_entropy(*a, **kw): pass

    class _StubTorch:
        FloatTensor = object
        LongTensor  = object
        device      = object
        @staticmethod
        def no_grad(): return _NullCtx()
        @staticmethod
        def zeros(*a, **kw): return None
        @staticmethod
        def ones(*a, **kw): return None

    class _NullCtx:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    nn    = _StubNN()
    F     = _StubF()
    torch = _StubTorch()

from config import MODELS_DIR
from src.rl.dqn_coaching import ACTIONS, NUM_ACTIONS

PATHWAY_ORDER  = ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']
SUBJECT_ORDER  = ['MATH', 'ENG', 'KIS_KSL', 'INT_SCI', 'AGRI',
                  'SOC_STUD', 'REL_CRE', 'CRE_ARTS', 'PRE_TECH']
GRADE_ORDER    = [4, 5, 6, 7, 8, 9]
N_SUBJECTS     = len(SUBJECT_ORDER)   # 9
N_GRADES       = len(GRADE_ORDER)     # 6
N_PATHWAYS     = len(PATHWAY_ORDER)   # 3

# ─────────────────────────────────────────────────────────────────────────────
#  LoRA linear layer
# ─────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Linear layer with a low-rank adapter: y = W·x + (B·A·x) * scale

    W is the pretrained weight (frozen during HITL fine-tuning).
    A (in_features × r) and B (r × out_features) are the trainable adapters.
    scale = alpha / r  (Hu et al. 2021 LoRA paper convention)
    """

    def __init__(self, in_features: int, out_features: int,
                 r: int = 8, alpha: int = 16, bias: bool = True):
        super().__init__()
        self.r     = r
        self.scale = alpha / r

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_  = nn.Parameter(torch.zeros(out_features)) if bias else None

        # LoRA adapters — initialised so BA=0 at the start
        self.lora_A = nn.Parameter(torch.randn(in_features, r)  * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        base = F.linear(x, self.weight, self.bias_)
        lora = (x @ self.lora_A @ self.lora_B) * self.scale
        return base + lora

    def freeze_base(self):
        """Freeze base weight for HITL fine-tuning (only LoRA trains)."""
        self.weight.requires_grad_(False)
        if self.bias_ is not None:
            self.bias_.requires_grad_(False)

    def unfreeze_base(self):
        self.weight.requires_grad_(True)
        if self.bias_ is not None:
            self.bias_.requires_grad_(True)


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-head attention with LoRA on Q and V
# ─────────────────────────────────────────────────────────────────────────────

class LoRAMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, r: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads

        # Q and V use LoRA; K and output projection are plain linear
        self.W_q = LoRALinear(d_model, d_model, r=r)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = LoRALinear(d_model, d_model, r=r)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale   = self.d_head ** -0.5

    def forward(self, x: 'torch.Tensor',
                mask: 'torch.Tensor | None' = None) -> tuple:
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out), attn

    def freeze_lora_base(self):
        self.W_q.freeze_base()
        self.W_v.freeze_base()

    def unfreeze_lora_base(self):
        self.W_q.unfreeze_base()
        self.W_v.unfreeze_base()


# ─────────────────────────────────────────────────────────────────────────────
#  Transformer encoder layer
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 ff_dim: int, r: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn   = LoRAMultiHeadAttention(d_model, n_heads, r=r, dropout=dropout)
        self.ff     = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: 'torch.Tensor') -> tuple:
        attn_out, attn_weights = self.attn(x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
#  Full Pathway Transformer
# ─────────────────────────────────────────────────────────────────────────────

class PathwayTransformer(nn.Module):
    """
    2-layer Transformer encoder for pathway + coaching-action prediction.

    Input  : (batch, N_GRADES+1, N_SUBJECTS)  — +1 for the [CLS] token
    Output : pathway logits (3), action logits (9)
    """

    def __init__(self, n_subjects: int = N_SUBJECTS,
                 n_grades: int  = N_GRADES,
                 d_model: int   = 64,
                 n_heads: int   = 4,
                 n_layers: int  = 2,
                 ff_dim: int    = 128,
                 r: int         = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.n_grades  = n_grades
        self.d_model   = d_model
        seq_len        = n_grades + 1   # +1 for [CLS]

        # Input projection: raw scores → d_model
        self.input_proj = nn.Linear(n_subjects, d_model)

        # Learnable [CLS] token and positional embeddings
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed  = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        self.drop = nn.Dropout(dropout)

        # Encoder stack
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, ff_dim, r=r, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output heads
        self.pathway_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, N_PATHWAYS),
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, NUM_ACTIONS),
        )

    def forward(self, x: 'torch.Tensor') -> tuple:
        """
        x : (B, N_GRADES, N_SUBJECTS)
        Returns pathway_logits (B,3), action_logits (B,9),
                attention_weights list (one per layer)
        """
        B = x.shape[0]
        h = self.input_proj(x)                         # (B, 6, d_model)
        cls = self.cls_token.expand(B, -1, -1)         # (B, 1, d_model)
        h = torch.cat([cls, h], dim=1)                 # (B, 7, d_model)
        h = self.drop(h + self.pos_embed)

        all_attn = []
        for layer in self.layers:
            h, attn = layer(h)
            all_attn.append(attn)

        cls_out = h[:, 0]                              # (B, d_model) — [CLS] token
        return self.pathway_head(cls_out), self.action_head(cls_out), all_attn

    def freeze_for_hitl(self):
        """Freeze all base weights; only LoRA adapters remain trainable."""
        for p in self.parameters():
            p.requires_grad_(False)
        # Unfreeze LoRA A/B matrices only
        for layer in self.layers:
            for p in [layer.attn.W_q.lora_A, layer.attn.W_q.lora_B,
                      layer.attn.W_v.lora_A, layer.attn.W_v.lora_B]:
                p.requires_grad_(True)

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
#  Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(assessments_df, student_ids,
                    grade_order=GRADE_ORDER,
                    subject_order=SUBJECT_ORDER) -> np.ndarray:
    """
    Convert assessments DataFrame → (N, n_grades, n_subjects) float32 array.
    Missing grades/subjects default to 0.5 (neutral mid-score).
    """
    import pandas as pd
    seqs = np.full((len(student_ids), len(grade_order), len(subject_order)),
                   0.5, dtype=np.float32)

    for i, sid in enumerate(student_ids):
        sdf = assessments_df[assessments_df['student_id'] == sid]
        for gi, grade in enumerate(grade_order):
            gdf = sdf[sdf['grade'] == grade]
            if gdf.empty:
                continue
            row = gdf.iloc[0]
            for si, subj in enumerate(subject_order):
                col = f'{subj}_score'
                if col in row.index and not (
                        hasattr(row[col], '__float__') and
                        np.isnan(float(row[col]))):
                    try:
                        seqs[i, gi, si] = float(row[col]) / 100.0
                    except (TypeError, ValueError):
                        pass
    return seqs


def build_labels(pathways_df, student_ids) -> np.ndarray:
    """Map student pathway labels → integer indices (STEM=0, SS=1, ARTS=2)."""
    pw2idx = {pw: i for i, pw in enumerate(PATHWAY_ORDER)}
    labels = np.zeros(len(student_ids), dtype=np.int64)
    for i, sid in enumerate(student_ids):
        row = pathways_df[pathways_df['student_id'] == sid]
        if not row.empty:
            labels[i] = pw2idx.get(row.iloc[0]['recommended_pathway'], 0)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
#  Agent wrapper  (same interface as PathwayRecommendationAgent)
# ─────────────────────────────────────────────────────────────────────────────

class PathwayTransformerAgent:
    """
    Wrapper around PathwayTransformer that exposes the same public interface
    as PathwayRecommendationAgent so both can be called interchangeably from
    the dashboard.

    Training  : supervised cross-entropy, fast (seconds not minutes)
    HITL      : LoRA fine-tuning — base weights frozen, adapters updated
    Inference : single forward pass, returns pathway + attention heatmap
    """

    def __init__(self, verbose: bool = True):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for PathwayTransformerAgent")

        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model   = PathwayTransformer().to(self.device)
        self.verbose = verbose
        self._trained = False
        self.accuracy = None

        total   = sum(p.numel() for p in self.model.parameters())
        trainable = self.model.count_trainable()
        if verbose:
            print(f"  Transformer initialized: {total:,} params total, "
                  f"{trainable:,} trainable")
            print(f"  LoRA rank=8 on Q,V in each of 2 attention layers")

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, assessments_df, pathways_df,
              epochs: int = 60,
              lr: float = 1e-3,
              batch_size: int = 64) -> dict:
        """
        Supervised training on (grade_sequence → pathway_label) pairs.

        Returns history dict with 'accuracy' and 'loss' lists.
        """
        import pandas as pd

        student_ids = sorted(pathways_df['student_id'].unique())
        X = build_sequences(assessments_df, student_ids)
        y = build_labels(pathways_df, student_ids)

        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)

        self.model.unfreeze_all()
        self.model.train()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                      weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)

        history = {'loss': [], 'accuracy': []}
        N = len(student_ids)

        if self.verbose:
            print(f"  Training Transformer: {N} students, {epochs} epochs")

        for epoch in range(epochs):
            # Shuffle
            perm  = torch.randperm(N)
            X_t   = X_t[perm]
            y_t   = y_t[perm]

            epoch_loss = 0.0
            correct    = 0
            n_batches  = max(1, N // batch_size)

            for b in range(n_batches):
                xb = X_t[b * batch_size: (b + 1) * batch_size]
                yb = y_t[b * batch_size: (b + 1) * batch_size]

                optimizer.zero_grad()
                pw_logits, _, _ = self.model(xb)
                loss = F.cross_entropy(pw_logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                correct    += (pw_logits.argmax(1) == yb).sum().item()

            scheduler.step()
            acc = correct / N
            history['loss'].append(epoch_loss / n_batches)
            history['accuracy'].append(acc)

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d}/{epochs}  "
                      f"loss={epoch_loss/n_batches:.4f}  acc={acc:.1%}")

        self._trained = True
        self.accuracy = history['accuracy'][-1]
        if self.verbose:
            print(f"  ✓ Training complete — accuracy: {self.accuracy:.1%}")
        return history

    # ── HITL fine-tuning via LoRA ─────────────────────────────────────────────

    def fine_tune_hitl(self, overrides: list,
                       assessments_df,
                       epochs: int = 30,
                       lr: float = 5e-4) -> dict:
        """
        Incorporate teacher overrides using LoRA fine-tuning.

        overrides : list of {'student_id': int, 'desired_pathway': str}
        Base model weights are FROZEN — only LoRA adapters update.
        Prevents catastrophic forgetting of the original training.
        """
        if not overrides:
            return {'success': False, 'error': 'No overrides provided'}

        pw2idx = {pw: i for i, pw in enumerate(PATHWAY_ORDER)}
        student_ids = [o['student_id'] for o in overrides]
        labels = np.array([pw2idx.get(o['desired_pathway'], 0) for o in overrides],
                          dtype=np.int64)

        X = build_sequences(assessments_df, student_ids)
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(labels).to(self.device)

        # Freeze base weights — only LoRA adapters train
        self.model.freeze_for_hitl()
        lora_params = self.model.count_trainable()
        if self.verbose:
            print(f"  LoRA fine-tuning: {len(overrides)} overrides, "
                  f"{lora_params} trainable params (base frozen)")

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

        self.model.train()
        history = {'loss': [], 'accuracy': []}

        for epoch in range(epochs):
            optimizer.zero_grad()
            pw_logits, _, _ = self.model(X_t)
            loss = F.cross_entropy(pw_logits, y_t)
            loss.backward()
            optimizer.step()

            acc = (pw_logits.argmax(1) == y_t).float().mean().item()
            history['loss'].append(loss.item())
            history['accuracy'].append(acc)

        # Restore full trainability for future normal training
        self.model.unfreeze_all()

        if self.verbose:
            print(f"  ✓ LoRA fine-tuning complete — "
                  f"override accuracy: {history['accuracy'][-1]:.1%}")

        return {'success': True,
                'override_accuracy': history['accuracy'][-1],
                'history': history}

    # ── Inference ────────────────────────────────────────────────────────────

    def recommend(self, student_id: int, assessments_df) -> dict:
        """
        Generate recommendation for a single student.
        Returns the same structure as PathwayRecommendationAgent.recommend()
        plus 'attention_weights' for explainability.
        """
        self.model.eval()

        X = build_sequences(assessments_df, [student_id])
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            pw_logits, act_logits, attn_list = self.model(X_t)

        pw_probs  = F.softmax(pw_logits[0], dim=-1).cpu().numpy()
        act_probs = F.softmax(act_logits[0], dim=-1).cpu().numpy()

        best_pw_idx  = int(np.argmax(pw_probs))
        best_act_idx = int(np.argmax(act_probs))
        recommended  = PATHWAY_ORDER[best_pw_idx]
        confidence   = float(pw_probs[best_pw_idx])

        # Attention on the last layer, averaged over heads, CLS row (token 0)
        # Shape of attn: (B, heads, seq, seq) → take [0] → mean heads → row 0
        last_attn = attn_list[-1][0].mean(0)[0].cpu().numpy()  # (seq_len,)
        # last_attn[0] = CLS→CLS (self), last_attn[1:] = CLS→grade tokens
        grade_attention = last_attn[1:].tolist()               # one per grade

        ranking = sorted(range(N_PATHWAYS), key=lambda i: pw_probs[i], reverse=True)

        return {
            'recommended_pathway': recommended,
            'confidence': round(confidence, 3),
            'confidence_scores': {
                PATHWAY_ORDER[i]: round(float(pw_probs[i]), 3)
                for i in range(N_PATHWAYS)
            },
            'pathway_ranking': [PATHWAY_ORDER[i] for i in ranking],
            'recommended_action': ACTIONS[best_act_idx].name,
            'action_confidence': round(float(act_probs[best_act_idx]), 3),
            'attention_weights': {
                f'G{GRADE_ORDER[i]}': round(grade_attention[i], 4)
                for i in range(N_GRADES)
            },
            'model': 'Transformer+LoRA',
            'is_override': False,
        }

    # ── Batch evaluation ─────────────────────────────────────────────────────

    def evaluate(self, assessments_df, pathways_df) -> dict:
        """Evaluate accuracy on all students. Returns per-pathway breakdown."""
        student_ids = sorted(pathways_df['student_id'].unique())
        X = build_sequences(assessments_df, student_ids)
        y = build_labels(pathways_df, student_ids)

        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            pw_logits, _, _ = self.model(X_t)

        preds = pw_logits.argmax(1).cpu().numpy()

        overall_acc = float((preds == y).mean())
        pw_acc = {}
        for i, pw in enumerate(PATHWAY_ORDER):
            mask = y == i
            if mask.sum() > 0:
                pw_acc[pw] = float((preds[mask] == y[mask]).mean())

        self.accuracy = overall_acc
        return {
            'overall_accuracy': overall_acc,
            'pathway_accuracy': pw_acc,
            'n_students': len(student_ids),
        }

    # ── Persist ──────────────────────────────────────────────────────────────

    def save(self, path: Path = None):
        if path is None:
            path = MODELS_DIR / 'transformer_pathway_model'
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'accuracy': self.accuracy,
        }, str(path) + '.pt')
        if self.verbose:
            print(f"  Transformer saved to {path}")

    def load(self, path: Path = None):
        if path is None:
            path = MODELS_DIR / 'transformer_pathway_model'
        path = Path(path)
        ckpt = torch.load(str(path) + '.pt', weights_only=False,
                          map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.accuracy = ckpt.get('accuracy')
        self._trained = True
        if self.verbose:
            print(f"  Transformer loaded from {path} "
                  f"(accuracy={self.accuracy:.1%})" if self.accuracy else
                  f"  Transformer loaded from {path}")
