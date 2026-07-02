"""
Cleopatra model in PyTorch.

Every intermediate 1-D representation (input epigenomics + positional
encoding, initial dense projection, each Conv1D output, each self-attention
output) is collected and concatenated in DenseNet fashion before the
pairwise combination step.  Pairwise features are concatenated with an
RBF distance embedding and passed through a small 2-D conv stack that
predicts log(observed/expected) contact values.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def insulation_kernel_numpy(size=41):
    assert size % 2 == 1
    mat = np.ones((size, size))
    half = size // 2
    mat[:half, half + 1:] = -1
    mat[half + 1:, :half] = -1
    return mat.astype(np.float32) * 1e-3


class RBFDistanceEncoding(nn.Module):
    """Radial Basis Function encoding: Gaussian kernels placed uniformly across
    the distance range, followed by a learned linear projection.  Smooth by
    construction -- widely used in molecular property prediction (SchNet)."""

    def __init__(self, distance_dim, max_distance=800, n_centers=64, width_factor=1.0):
        super().__init__()
        centers = torch.linspace(0, max_distance, n_centers)
        self.register_buffer('centers', centers)
        spacing = (centers[1] - centers[0]).item() if n_centers > 1 else float(max_distance)
        self.width = spacing * width_factor
        self.proj = nn.Linear(n_centers, distance_dim)

    def forward(self, distance):
        d = distance.float().unsqueeze(-1)                                     # (..., 1)
        rbf = torch.exp(-((d - self.centers) ** 2) / (2 * self.width ** 2))    # (..., n_centers)
        return self.proj(rbf)                                                  # (..., dim)


class RepeatVector3D(nn.Module):
    """Repeats the input n times along a new axis.
    Input: (N, L, C) -> Output: (N, L, n, C)
    """

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x.unsqueeze(2).repeat(1, 1, self.n, 1)


def custom_insulation_loss(y_true, y_pred, ins_kernel):
    """
    Insulation loss: match diagonal of convolution with insulation kernel.
    Uses VALID padding to match original TensorFlow implementation.
    y_true, y_pred: (N, H, W)
    ins_kernel: (1, 1, kh, kw) for conv2d
    """
    if y_pred.dim() == 3:
        y_true = y_true.unsqueeze(1)
        y_pred = y_pred.unsqueeze(1)
    conv_true = F.conv2d(y_true, ins_kernel, padding=0)
    conv_pred = F.conv2d(y_pred, ins_kernel, padding=0)
    conv_true = conv_true.squeeze(1)
    conv_pred = conv_pred.squeeze(1)
    diag_true = torch.diagonal(conv_true, dim1=1, dim2=2)
    diag_pred = torch.diagonal(conv_pred, dim1=1, dim2=2)
    loss = F.mse_loss(diag_true, diag_pred, reduction='mean')
    return loss


def combined_loss_fn(y_true, y_pred, ins_kernel, beta=10.0):
    mse_loss = F.mse_loss(y_true, y_pred, reduction='mean')
    ins_loss = custom_insulation_loss(y_true, y_pred, ins_kernel)
    return mse_loss + beta * ins_loss


class SelfAttentionBlock(nn.Module):
    """
    Self-attention block with asymmetric internal dimension.

    Input/output are Conv_dim-wide.  Internally, Q/K/V are projected to SA_dim
    so that per-head dimension = SA_dim // num_heads (e.g. 96 // 8 = 12).
    Output projection maps SA_dim back to Conv_dim, followed by BatchNorm.
    """

    def __init__(self, conv_dim, sa_dim, num_heads):
        super().__init__()
        assert sa_dim % num_heads == 0
        self.sa_dim = sa_dim
        self.num_heads = num_heads
        self.head_dim = sa_dim // num_heads

        self.q_proj = nn.Linear(conv_dim, sa_dim)
        self.k_proj = nn.Linear(conv_dim, sa_dim)
        self.v_proj = nn.Linear(conv_dim, sa_dim)
        self.out_proj = nn.Linear(sa_dim, conv_dim)
        self.bn = nn.BatchNorm1d(conv_dim)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.sa_dim)

        y = self.out_proj(attn_out)
        y = self.bn(y.transpose(1, 2)).transpose(1, 2)
        return y


class Cleopatra(nn.Module):
    """
    Cleopatra model (DenseNet-style concatenation of intermediate 1-D
    representations, followed by a 2-D refinement stack).
    """

    def __init__(
        self,
        nBins=1250,
        nMarks=14,
        verbose=1,
        lr=0.0001,
        positional_dim=8,
        n_distance=800,
        distance_dim=8,
        n_SA_layers=2,
        SA_dim=96,   # kept for API parity; controls MHA key_dim indirectly
        SA_head=8,
        n_Conv_layers=3,
        Conv_dim=64,
        Conv_kernel=21,
        n_final_layers=3,
        final_layer_dim=24,
        final_layer_kernel=3,
        insulation_loss=True,
        beta=1.0,
        freeze_distance_embedding=False,
        rbf_n_centers=64,
        rbf_width_factor=1.0,
    ):
        super().__init__()
        assert SA_dim % SA_head == 0
        assert n_SA_layers > 0
        assert n_Conv_layers > 0
        assert n_final_layers > 0

        self.nBins = nBins
        self.insulation_loss = insulation_loss
        self.beta = beta
        self._verbose = verbose

        # Insulation kernel (fixed)
        ins_mat = insulation_kernel_numpy()
        ins_kernel = torch.from_numpy(ins_mat).reshape(1, 1, ins_mat.shape[0], ins_mat.shape[1])
        self.register_buffer('ins_kernel', ins_kernel)

        input_dim = nMarks + positional_dim
        self.input_dim = input_dim

        # ── 1-D processing ──────────────────────────────────────────────────
        # Initial dense projection: input_dim → Conv_dim
        self.conv_0 = nn.Linear(input_dim, Conv_dim)
        self.conv_0_bn = nn.BatchNorm1d(Conv_dim)

        # Conv1D layers (sequential; each takes the previous output)
        self.conv_layers = nn.ModuleList()
        self.conv_bns = nn.ModuleList()
        for _ in range(n_Conv_layers):
            self.conv_layers.append(
                nn.Conv1d(Conv_dim, Conv_dim, kernel_size=Conv_kernel, padding=Conv_kernel // 2)
            )
            self.conv_bns.append(nn.BatchNorm1d(Conv_dim))

        # Self-attention layers: operate on Conv_dim (no separate projection)
        self.sa_blocks = nn.ModuleList()
        for _ in range(n_SA_layers):
            self.sa_blocks.append(SelfAttentionBlock(Conv_dim, SA_dim, SA_head))

        # ── DenseNet concatenation ───────────────────────────────────────────
        # Collected tensors: epi_data + conv_0 + n_Conv_layers × Conv_dim
        #                    + n_SA_layers × Conv_dim
        conc_dim = input_dim + Conv_dim * (1 + n_Conv_layers + n_SA_layers)
        self.conc_dim = conc_dim

        # ── Pairwise heads (take concatenated representation) ────────────────
        self.pair1 = nn.Conv1d(conc_dim, final_layer_dim, kernel_size=1)
        self.pair2 = nn.Conv1d(conc_dim, final_layer_dim, kernel_size=1)
        self.repeat_3d = RepeatVector3D(nBins)

        # Distance embedding (RBF kernels + linear projection -- smooth by construction)
        self.dist_embed = RBFDistanceEncoding(distance_dim, max_distance=n_distance, n_centers=rbf_n_centers, width_factor=rbf_width_factor)

        # Freeze distance embedding if requested
        if freeze_distance_embedding:
            for p in self.dist_embed.parameters():
                p.requires_grad = False

        # ── Final 2-D conv stack (identical to original Cleo) ────────────────
        pair_dim = final_layer_dim + distance_dim
        final_in = pair_dim
        self.final_convs = nn.ModuleList()
        self.final_bns = nn.ModuleList()
        for _ in range(n_final_layers - 1):
            self.final_convs.append(
                nn.Conv2d(final_in, final_layer_dim, kernel_size=final_layer_kernel,
                          padding=final_layer_kernel // 2)
            )
            self.final_bns.append(nn.BatchNorm2d(final_layer_dim))
            final_in = final_layer_dim
        self.final_last = nn.Conv2d(final_in, 1, kernel_size=1)

        if verbose:
            self._print_summary()

    def _print_summary(self):
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Cleo (DenseNet) model: {n_params:,} parameters  |  conc_dim={self.conc_dim}")

    def forward(self, epi_inp, positional, distance0, weights):
        """
        epi_inp:    (N, nBins, nMarks)
        positional: (N, nBins, positional_dim)
        distance0:  (N, nBins, nBins)
        weights:    (N, nBins, nBins)
        """
        N = epi_inp.size(0)

        epi_data = torch.cat([epi_inp, positional], dim=-1)  # (N, nBins, input_dim)

        # ── 1-D processing: collect every intermediate representation ────────
        collected = [epi_data]

        # Dense projection + ReLU + BN
        x = F.relu(self.conv_0(epi_data))                          # (N, nBins, Conv_dim)
        x = self.conv_0_bn(x.transpose(1, 2)).transpose(1, 2)
        collected.append(x)

        # Conv1D blocks (no residual addition)
        for conv, bn in zip(self.conv_layers, self.conv_bns):
            x_t = x.transpose(1, 2)           # (N, Conv_dim, nBins)
            x_t = conv(x_t)
            x_t = bn(x_t)
            x = F.relu(x_t).transpose(1, 2)   # (N, nBins, Conv_dim)
            collected.append(x)

        # SA blocks: each block handles projections internally, output is Conv_dim-wide
        for sa_block in self.sa_blocks:
            x = sa_block(x)                                           # (N, nBins, Conv_dim)
            collected.append(x)

        # DenseNet concat: (N, nBins, conc_dim)
        conc = torch.cat(collected, dim=-1)

        # ── Pairwise combination ─────────────────────────────────────────────
        h = conc.transpose(1, 2)                              # (N, conc_dim, nBins)
        p1 = F.relu(self.pair1(h)).transpose(1, 2)           # (N, nBins, final_layer_dim)
        p2 = F.relu(self.pair2(h)).transpose(1, 2)
        p1_stack = self.repeat_3d(p1)                         # (N, nBins, nBins, final_layer_dim)
        p2_stack = self.repeat_3d(p2)
        p2_stack_T = p2_stack.permute(0, 2, 1, 3)
        adds = p1_stack + p2_stack_T                          # (N, nBins, nBins, final_layer_dim)

        # Distance embedding
        distance_emb = self.dist_embed(distance0)             # (N, nBins, nBins, distance_dim)

        final = torch.cat([adds, distance_emb], dim=-1)       # (N, nBins, nBins, pair_dim)

        # ── Final 2-D conv stack ─────────────────────────────────────────────
        final = final.permute(0, 3, 1, 2)                     # (N, pair_dim, nBins, nBins)
        for conv2d, bn2d in zip(self.final_convs, self.final_bns):
            final = conv2d(final)
            final = F.relu(final)
            final = bn2d(final)
        final = self.final_last(final)
        final = F.relu(final)
        final = final.permute(0, 2, 3, 1).squeeze(-1)         # (N, nBins, nBins)

        # Symmetrize and apply weights
        out = final * weights
        out = 0.5 * (out + out.transpose(1, 2))
        return out

    def train_step(self, batch, target, optimizer, device):
        self.train()
        optimizer.zero_grad()
        epi, pos_enc, distance, weights = batch
        epi = epi.to(device)
        pos_enc = pos_enc.to(device)
        distance = distance.to(device)
        weights = weights.to(device)
        target = target.to(device)

        pred = self.forward(epi, pos_enc, distance, weights)
        if self.insulation_loss:
            loss = combined_loss_fn(target, pred, self.ins_kernel, beta=self.beta)
        else:
            loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate_batch(self, batch, target, device):
        self.eval()
        epi, pos_enc, distance, weights = batch
        epi = epi.to(device)
        pos_enc = pos_enc.to(device)
        distance = distance.to(device)
        weights = weights.to(device)
        target = target.to(device)

        pred = self.forward(epi, pos_enc, distance, weights)
        return F.mse_loss(pred, target).item()

    @torch.no_grad()
    def predict_batch(self, batch, device):
        self.eval()
        epi, pos_enc, distance, weights = batch
        epi = epi.to(device)
        pos_enc = pos_enc.to(device)
        distance = distance.to(device)
        weights = weights.to(device)
        pred = self.forward(epi, pos_enc, distance, weights)
        return pred.cpu().numpy()

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path, device=None, strict=True):
        state = torch.load(path, map_location=device or next(self.parameters()).device)
        if strict:
            self.load_state_dict(state, strict=True)
        else:
            # Partial loading: skip mismatched keys (e.g. dist_embed with different n_centers)
            model_state = self.state_dict()
            filtered = {k: v for k, v in state.items()
                        if k in model_state and v.shape == model_state[k].shape}
            skipped = [k for k in state if k not in filtered]
            if skipped:
                print(f"Skipped loading keys (shape mismatch): {skipped}")
            model_state.update(filtered)
            self.load_state_dict(model_state)
