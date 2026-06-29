# Derivations

Hand-derived gradients for every parameterized operation in `src/transformer/`,
each paired with a numerical check (central differences, relative error
`< 1e-5`) that lives in `tests/`.

The convention throughout: scalars in lowercase, vectors in bold lowercase,
matrices in uppercase. Gradients of a scalar loss `L` with respect to a tensor
`X` are written `dX` (the same shape as `X`). All sums are over indices made
explicit by the bounds.

## Contents

| # | Section | Implementation | Numerical check |
|---|---------|----------------|-----------------|
| 1 | [Linear and ReLU](#1-linear-and-relu-the-building-blocks) | [`linear.py`](../src/transformer/linear.py) | [`test_linear.py`](../tests/test_linear.py) |
| 2 | [Softmax and cross-entropy](#2-softmax-and-cross-entropy-the-combined-gradient) | [`linear.py`](../src/transformer/linear.py) | [`test_linear.py`](../tests/test_linear.py) |
| 3 | [LayerNorm backward](#3-layernorm-backward) | [`layernorm.py`](../src/transformer/layernorm.py) | [`test_layernorm.py`](../tests/test_layernorm.py) |
| 4 | [Attention gradients](#4-scaled-dot-product-attention-gradients-in-matrix-form) | [`attention.py`](../src/transformer/attention.py) | [`test_attention.py`](../tests/test_attention.py) |
| 5 | [Multi-head reshape](#5-multi-head-reshape-in-tensor-index-notation) | [`mha.py`](../src/transformer/mha.py) | [`test_mha.py`](../tests/test_mha.py) |
| 6 | [Adam + bias correction](#6-adam-update-rule-and-bias-correction) | [`optim.py`](../src/transformer/optim.py) | [`test_optim.py`](../tests/test_optim.py) |
| 7 | [LR schedules: warmup + cosine](#7-lr-schedules-warmup-and-cosine-decay) | [`schedule.py`](../src/transformer/schedule.py) | [`test_schedule.py`](../tests/test_schedule.py) |
| 8 | [Inverted dropout](#8-inverted-dropout-expectation-and-gradient) | [`dropout.py`](../src/transformer/dropout.py) | [`test_dropout.py`](../tests/test_dropout.py) |
| 9 | [Weight tying](#9-weight-tying-combining-two-gradient-contributions) | [`model.py`](../src/transformer/model.py) | [`test_model.py`](../tests/test_model.py) |

## 1. Linear and ReLU: the building blocks

**Implementation:** [`src/transformer/linear.py`](../src/transformer/linear.py) (`Linear`, `ReLU`).
**Numerical check:** [`tests/test_linear.py`](../tests/test_linear.py).

Every parameterized layer reduces to these two primitives. The FFN is literally
Linear → ReLU → Linear, and the attention and output projections are Linears, so
deriving them once covers the rest by composition.

### Linear

For input $X \in \mathbb{R}^{N \times d_{\text{in}}}$, weight
$W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ and bias
$b \in \mathbb{R}^{d_{\text{out}}}$, the forward pass is $Y = XW + b$, i.e.
$Y_{nj} = \sum_{i} X_{ni} W_{ij} + b_j$. Given $\mathrm{d}Y = \partial L / \partial Y$,
read off each gradient with the chain rule:

$$
\mathrm{d}W_{ij} = \sum_{n} \frac{\partial L}{\partial Y_{nj}} \frac{\partial Y_{nj}}{\partial W_{ij}}
= \sum_{n} X_{ni} \, \mathrm{d}Y_{nj}
\;\Longrightarrow\; \mathrm{d}W = X^{\top} \mathrm{d}Y,
$$

$$
\mathrm{d}b_j = \sum_{n} \mathrm{d}Y_{nj}
\;\Longrightarrow\; \mathrm{d}b = \sum_{n} \mathrm{d}Y_{n,:},
\qquad
\mathrm{d}X = \mathrm{d}Y \, W^{\top}.
$$

This is `Linear.backward` (`linear.py`); the leading `reshape(-1, ...)`
just folds batch and sequence axes together so the same 2D identity applies.

### ReLU

$\mathrm{ReLU}(x) = \max(0, x)$ is elementwise, with derivative
$\mathbb{1}[x > 0]$. Caching the boolean mask $M = \mathbb{1}[x > 0]$ from the
forward pass, the backward is

$$
\mathrm{d}x = M \odot \mathrm{d}y.
$$

The boundary point $x = 0$ is non-differentiable; we follow the standard
convention of using a subgradient of $0$ there ($M_i = 0$ when $x_i = 0$).

### FFN composition

The feed-forward block is $\mathrm{FFN}(x) = \mathrm{ReLU}(x W_1 + b_1) W_2 + b_2$.
Its backward is just the two Linear backwards above with the ReLU mask in
between; no new calculus. The test checks $\mathrm{d}W_1, \mathrm{d}b_1,
\mathrm{d}W_2, \mathrm{d}b_2, \mathrm{d}x$ against central differences.

## 2. Softmax and cross-entropy: the combined gradient

**Implementation:** [`src/transformer/linear.py`](../src/transformer/linear.py) (`softmax`, `cross_entropy_loss`).
**Numerical check:** [`tests/test_linear.py::test_softmax_cross_entropy_gradient_matches_numerical`](../tests/test_linear.py).

### Setup

Let $z \in \mathbb{R}^{C}$ be the logits for one example and $t \in \{0, \dots, C-1\}$
its target class. The softmax produces a probability vector

$$
p_i = \mathrm{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{C} \exp(z_j)},
$$

and the per-example cross-entropy is

$$
L = -\log p_t.
$$

The two operations are almost always composed, so we derive the combined
gradient $\partial L / \partial z$ directly. This avoids constructing the
$C \times C$ softmax Jacobian explicitly and is what `cross_entropy_loss`
implements.

### Softmax Jacobian

We start from the Jacobian of softmax. For $i, j \in \{1, \dots, C\}$,

$$
\frac{\partial p_i}{\partial z_j}
= \frac{\partial}{\partial z_j}
  \left( \frac{\exp(z_i)}{S} \right),
\qquad
S = \sum_{k=1}^{C} \exp(z_k).
$$

Apply the quotient rule. The numerator derivative is $\delta_{ij} \exp(z_i)$
(only the diagonal term survives), the denominator derivative is $\exp(z_j)$:

$$
\frac{\partial p_i}{\partial z_j}
= \frac{\delta_{ij} \exp(z_i) \, S - \exp(z_i) \exp(z_j)}{S^2}
= \delta_{ij} p_i - p_i p_j
= p_i (\delta_{ij} - p_j).
$$

So the Jacobian has diagonal entries $p_i(1 - p_i)$ and off-diagonal entries
$-p_i p_j$.

### Cross-entropy gradient

The multivariate chain rule gives

$$
\frac{\partial L}{\partial z_j}
= \sum_{i=1}^{C}
  \frac{\partial L}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_j}.
$$

Because $L = -\log p_t$ is a function of $p_t$ alone, only the $i = t$ term is
non-zero ($\partial L / \partial p_i = -\delta_{it} / p_t$), so the sum
collapses:

$$
\frac{\partial L}{\partial z_j}
= -\frac{1}{p_t} \cdot \frac{\partial p_t}{\partial z_j}
= -\frac{1}{p_t} \cdot p_t (\delta_{tj} - p_j)
= p_j - \delta_{tj}.
$$

Vectorized: $\nabla_z L = p - e_t$, where $e_t$ is the one-hot vector at
position $t$. This is the famous "subtract one from the correct class"
identity. It collapses what would be a chain through a $C \times C$ Jacobian
into a single subtraction.

### Batch averaging

For a batch of $N$ examples we average the per-example losses
$L = \frac{1}{N} \sum_{n=1}^{N} L^{(n)}$, so each per-example gradient is
divided by $N$:

$$
\frac{\partial L}{\partial z^{(n)}_j} = \frac{1}{N} \left( p^{(n)}_j - \delta_{t^{(n)} j} \right).
$$

This matches `cross_entropy_loss` (which works in log space, so `p = exp(log_probs)`):

```python
dlogits = np.exp(log_probs)
dlogits[np.arange(N), targets] -= 1
dlogits /= N
```

### Numerical check (sketch)

Given random `logits` of shape `(N, C)` and integer `targets`:

```python
loss, dlogits = cross_entropy_loss(logits, targets)

def L_at(z):
    loss, _ = cross_entropy_loss(z, targets)
    return loss

num_dlogits = numerical_gradient(L_at, logits.copy())  # central differences
assert relative_error(dlogits, num_dlogits) < 1e-5
```

The test runs this with `N=5, C=7` and passes with relative error well under
the threshold.

## 3. LayerNorm backward

**Implementation:** [`src/transformer/layernorm.py`](../src/transformer/layernorm.py) (`LayerNorm`).
**Numerical check:** [`tests/test_layernorm.py::test_layernorm_dx_matches_numerical`](../tests/test_layernorm.py)
and `::test_layernorm_dgamma_dbeta_match_numerical`.

This is the most error-prone gradient in the model: $\mu$ and $\sigma$ both
depend on every $x_k$, so $\hat{x}$ is not a coordinate-wise function of $x$
and the Jacobian is dense. We derive it from first principles.

### Setup

For one token, let $x \in \mathbb{R}^{D}$ and define

$$
\mu = \frac{1}{D} \sum_{i=1}^{D} x_i,
\qquad
\sigma^{2} = \frac{1}{D} \sum_{i=1}^{D} (x_i - \mu)^{2},
\qquad
\sigma = \sqrt{\sigma^{2} + \varepsilon}.
$$

Then

$$
\hat{x}_i = \frac{x_i - \mu}{\sigma},
\qquad
y_i = \gamma_i \hat{x}_i + \beta_i.
$$

Given $\mathrm{d}y_i = \partial L / \partial y_i$, we want $\mathrm{d}\gamma$,
$\mathrm{d}\beta$, and $\mathrm{d}x$. The code stores `std_inv = 1/sigma` and
`x_hat`, so the final formula should be expressible in those two cached
tensors.

### The easy gradients

From $y_i = \gamma_i \hat{x}_i + \beta_i$:

$$
\frac{\partial y_i}{\partial \beta_j} = \delta_{ij},
\qquad
\frac{\partial y_i}{\partial \gamma_j} = \delta_{ij} \hat{x}_i,
\qquad
\frac{\partial y_i}{\partial \hat{x}_j} = \delta_{ij} \gamma_i.
$$

So

$$
\mathrm{d}\beta_i = \mathrm{d}y_i,
\qquad
\mathrm{d}\gamma_i = \mathrm{d}y_i \, \hat{x}_i,
\qquad
\mathrm{d}\hat{x}_i = \mathrm{d}y_i \, \gamma_i.
$$

For a batch of tokens, $\mathrm{d}\gamma$ and $\mathrm{d}\beta$ sum over the
batch axis (`layernorm.py`). $\gamma$ and $\beta$ are shared across all
positions, so every token contributes to their gradient.

### The Jacobian of $\hat{x}$ with respect to $x$

The hard part is $\mathrm{d}x$, because $\hat{x}_i$ depends on every $x_k$
through both $\mu$ and $\sigma$. We compute the full Jacobian
$\partial \hat{x}_i / \partial x_k$ and then contract it with $\mathrm{d}\hat{x}$.

Pieces we need:

$$
\frac{\partial \mu}{\partial x_k} = \frac{1}{D}.
$$

$$
\frac{\partial (x_i - \mu)}{\partial x_k} = \delta_{ik} - \frac{1}{D}.
$$

$$
\frac{\partial \sigma^{2}}{\partial x_k}
= \frac{1}{D} \sum_{i=1}^{D} 2(x_i - \mu) \left( \delta_{ik} - \frac{1}{D} \right)
= \frac{2}{D}(x_k - \mu) - \frac{2}{D^{2}} \underbrace{\sum_{i=1}^{D} (x_i - \mu)}_{= \, 0}
= \frac{2}{D}(x_k - \mu).
$$

The middle sum vanishes because deviations from the mean sum to zero. Now

$$
\frac{\partial \sigma}{\partial x_k}
= \frac{1}{2\sigma} \cdot \frac{\partial \sigma^{2}}{\partial x_k}
= \frac{x_k - \mu}{D \sigma}.
$$

Apply the quotient rule to $\hat{x}_i = (x_i - \mu) / \sigma$:

$$
\frac{\partial \hat{x}_i}{\partial x_k}
= \frac{1}{\sigma}\left(\delta_{ik} - \frac{1}{D}\right)
  - \frac{x_i - \mu}{\sigma^{2}} \cdot \frac{x_k - \mu}{D \sigma}.
$$

Using $\hat{x}_i = (x_i - \mu)/\sigma$ in the second term:

$$
\frac{\partial \hat{x}_i}{\partial x_k}
= \frac{1}{\sigma}\left(\delta_{ik} - \frac{1}{D}\right) - \frac{\hat{x}_i \hat{x}_k}{D \sigma}
= \frac{1}{D \sigma} \left( D \delta_{ik} - 1 - \hat{x}_i \hat{x}_k \right).
$$

This is a dense $D \times D$ matrix and the source of the "every input
position affects every output position" coupling.

### Contracting against $\mathrm{d}\hat{x}$

$$
\mathrm{d}x_k
= \sum_{i=1}^{D} \mathrm{d}\hat{x}_i \cdot \frac{\partial \hat{x}_i}{\partial x_k}
= \frac{1}{D \sigma} \sum_{i=1}^{D} \mathrm{d}\hat{x}_i \left( D \delta_{ik} - 1 - \hat{x}_i \hat{x}_k \right).
$$

Split the sum into three pieces:

$$
\mathrm{d}x_k = \frac{1}{D \sigma}
\left[
  D \cdot \mathrm{d}\hat{x}_k
  - \sum_{i} \mathrm{d}\hat{x}_i
  - \hat{x}_k \sum_{i} \mathrm{d}\hat{x}_i \hat{x}_i
\right].
$$

Divide through by $D$ to rewrite the bracketed sums as means $\bar{\cdot}$ over
the feature axis:

$$
\boxed{
\mathrm{d}x_k = \frac{1}{\sigma}
\left[
  \mathrm{d}\hat{x}_k - \overline{\mathrm{d}\hat{x}} - \hat{x}_k \cdot \overline{\mathrm{d}\hat{x} \, \hat{x}}
\right].
}
$$

This matches `LayerNorm.backward` line for line:

```python
return self.std_inv * (                                # 1/sigma
    dx_hat                                             # d x_hat_k
    - np.mean(dx_hat, axis=-1, keepdims=True)          # mean(d x_hat)
    - self.x_hat * np.mean(dx_hat * self.x_hat, axis=-1, keepdims=True)
)
```

### Numerical check (sketch)

For $D = 7$ and random $x, \mathrm{d}y, \gamma, \beta$:

```python
ln = LayerNorm(D)
ln.gamma = gamma; ln.beta = beta
ln.forward(x); dx = ln.backward(dy)

def L_at(x_in):
    return float(np.sum(ln.forward(x_in) * dy))

num_dx = numerical_gradient(L_at, x.copy())
assert relative_error(dx, num_dx) < 1e-5
```

Direct verification (during the derivation of this section): the analytical
$\mathrm{d}x$ matches central differences with relative error $\approx 4 \times
10^{-10}$, and the dense Jacobian $\partial \hat{x}_i / \partial x_k$ matches
its numerical counterpart with absolute error $\approx 3.5 \times 10^{-10}$.

## 4. Scaled dot-product attention: gradients in matrix form

**Implementation:** [`src/transformer/attention.py`](../src/transformer/attention.py) (`SingleHeadAttention`).
**Numerical check:** [`tests/test_attention.py::test_attention_projection_gradients_match_numerical`](../tests/test_attention.py)
and `::test_attention_dx_matches_numerical`.

The attention block is a stack of five operations, all linear or row-wise.
Each one is simple in isolation; the value of writing this section is to keep
the matrix shapes straight and to confirm that the row-wise softmax gradient
combines correctly with the scale and masking.

### Forward

Drop the batch axis for clarity. Let $X \in \mathbb{R}^{T \times d_{\text{model}}}$
be the input. The forward pass is

$$
Q = X W_Q,
\qquad
K = X W_K,
\qquad
V = X W_V
\qquad
(\text{all } T \times d_k),
$$

$$
S' = Q K^{\top} \in \mathbb{R}^{T \times T},
\qquad
S = \frac{S'}{\sqrt{d_k}},
\qquad
\tilde{S} = \text{mask}(S),
$$

$$
A = \mathrm{softmax}(\tilde{S}) \text{ row-wise},
\qquad
O = A V \in \mathbb{R}^{T \times d_k}.
$$

The mask replaces upper-triangular entries with $-\infty$ before softmax, so
$A_{ij} = 0$ at masked positions and each row of $A$ still sums to $1$. We
will see that this makes the mask's backward trivial.

### Backward through $O = AV$

This is two matrix multiplies. Given $\mathrm{d}O$, the standard identities
for $C = AB$ (with shapes that match) give

$$
\mathrm{d}V = A^{\top} \, \mathrm{d}O,
\qquad
\mathrm{d}A = \mathrm{d}O \, V^{\top}.
$$

Shapes: $\mathrm{d}V$ is $T \times d_k$, $\mathrm{d}A$ is $T \times T$.

### Backward through row-wise softmax

Each row $A_{i:}$ is the softmax of $\tilde S_{i:}$, independent of other rows.
From section&nbsp;2, the single-row softmax Jacobian is

$$
\frac{\partial A_{ij}}{\partial \tilde S_{ik}} = A_{ij} (\delta_{jk} - A_{ik}).
$$

Chaining $\mathrm{d}\tilde S_{ik} = \sum_j \mathrm{d}A_{ij} \cdot \partial A_{ij} / \partial \tilde S_{ik}$:

$$
\mathrm{d}\tilde S_{ik}
= \sum_{j=1}^{T} \mathrm{d}A_{ij} A_{ij} (\delta_{jk} - A_{ik})
= A_{ik} \, \mathrm{d}A_{ik} - A_{ik} \sum_{j=1}^{T} \mathrm{d}A_{ij} A_{ij}
= A_{ik} \Bigl( \mathrm{d}A_{ik} - \sum_{j} \mathrm{d}A_{ij} A_{ij} \Bigr).
$$

Vectorize over the row and stack rows back into a matrix. Let
$\overline{r}_i = \sum_j \mathrm{d}A_{ij} A_{ij}$ be the row-wise inner
product, broadcast across columns. Then

$$
\boxed{
\mathrm{d}\tilde S = A \odot \bigl( \mathrm{d}A - \overline{r} \mathbf{1}^{\top} \bigr).
}
$$

This matches `SingleHeadAttention.backward`:

```python
sum_term = np.sum(d_attn * self.attn_weights, axis=-1, keepdims=True)
d_scores = self.attn_weights * (d_attn - sum_term)
```

### Backward through the mask

The mask is $\tilde S_{ij} = S_{ij}$ at unmasked positions and $-\infty$
otherwise. The $-\infty$ entries are constants with respect to $S$, so
$\partial \tilde S_{ij} / \partial S_{ij} = 1$ at unmasked positions and the
mask backward would be: zero $\mathrm{d}\tilde S$ at masked positions, copy
elsewhere.

But from the previous step, $\mathrm{d}\tilde S = A \odot (\ldots)$, and
$A_{ij} = 0$ at masked positions, so $\mathrm{d}\tilde S$ is already zero
there. No explicit mask handling is needed in backward.

### Backward through the scale

$S = S' / \sqrt{d_k}$ is a scalar division, so

$$
\mathrm{d}S' = \frac{\mathrm{d}S}{\sqrt{d_k}} = \frac{\mathrm{d}\tilde S}{\sqrt{d_k}}.
$$

This is `attention.py` (`d_scores /= np.sqrt(self.d_k)`).

### Backward through $S' = Q K^{\top}$

Index form: $S'_{ij} = \sum_l Q_{il} K_{jl}$.

$$
\frac{\partial S'_{ij}}{\partial Q_{ab}} = \delta_{ia} K_{jb},
\qquad
\frac{\partial S'_{ij}}{\partial K_{ab}} = \delta_{ja} Q_{ib}.
$$

Contract with $\mathrm{d}S'$:

$$
\mathrm{d}Q_{ab}
= \sum_{i, j} \mathrm{d}S'_{ij} \delta_{ia} K_{jb}
= \sum_j \mathrm{d}S'_{aj} K_{jb}
\;\Longrightarrow\;
\mathrm{d}Q = \mathrm{d}S' \, K.
$$

$$
\mathrm{d}K_{ab}
= \sum_{i, j} \mathrm{d}S'_{ij} \delta_{ja} Q_{ib}
= \sum_i \mathrm{d}S'_{ia} Q_{ib}
\;\Longrightarrow\;
\mathrm{d}K = (\mathrm{d}S')^{\top} \, Q.
$$

Both lines match `attention.py`.

### Backward through the projections

Each of $Q = X W_Q$, $K = X W_K$, $V = X W_V$ is a plain matrix product, with
the same backward pattern as Linear:

$$
\mathrm{d}W_Q = X^{\top} \mathrm{d}Q,
\qquad
\mathrm{d}W_K = X^{\top} \mathrm{d}K,
\qquad
\mathrm{d}W_V = X^{\top} \mathrm{d}V,
$$

and the three contributions to $\mathrm{d}X$ add up:

$$
\mathrm{d}X = \mathrm{d}Q \, W_Q^{\top} + \mathrm{d}K \, W_K^{\top} + \mathrm{d}V \, W_V^{\top}.
$$

This is in `SingleHeadAttention.backward`. The reshape `x_flat = x.reshape(-1, d_model)`
just folds the batch and sequence axes together so the same identity applies.

### Numerical check (sketch)

The test instantiates a `SingleHeadAttention(d_model=8, d_k=4)` and verifies
each of $\mathrm{d}W_Q$, $\mathrm{d}W_K$, $\mathrm{d}W_V$, $\mathrm{d}x$
against central differences. During the derivation here we also confirmed
$\mathrm{d}Q$, $\mathrm{d}K$, $\mathrm{d}V$ directly at the pre-projection
level on $T=4, d_k=3$ random tensors: all three matched numerically with
relative error below $1.3 \times 10^{-9}$.

## 5. Multi-head reshape in tensor index notation

**Implementation:** [`src/transformer/mha.py`](../src/transformer/mha.py) (`MultiHeadAttention`).
**Numerical check:** [`tests/test_mha.py::test_mha_projection_gradients_match_numerical`](../tests/test_mha.py),
`::test_mha_dx_matches_numerical`,
`::test_mha_attention_weights_respect_causal_mask`.

Multi-head attention does not introduce any new calculus. It is the
single-head attention of section&nbsp;4 applied to $H$ disjoint $d_k$-wide
slices of the embedding axis, plus an output projection. The "split heads"
and "merge heads" reshapes look like rearrangements of memory and nothing
more; what they actually do is pick out and re-assemble those slices. Writing
this out in index notation makes the equivalence explicit, so the backward
needs no new derivation.

### The reshape, in indices

Let $d_{\text{model}} = H \cdot d_k$. The split reshape acts on a tensor of
shape $(B, T, d_{\text{model}})$ and returns one of shape $(B, H, T, d_k)$:

$$
(\texttt{split}(X))_{b, h, t, k} = X_{b, t, h \cdot d_k + k},
\qquad h \in \{0, \dots, H-1\}, \; k \in \{0, \dots, d_k - 1\}.
$$

This is a bijection between the index sets $\{(b, t, d)\}$ (with $d \in [0,
d_{\text{model}})$) and $\{(b, h, t, k)\}$. Pythonically the call performs a
`reshape((B, T, H, d_k))` followed by `transpose((0, 2, 1, 3))`. The merge
reshape is the literal inverse:

$$
(\texttt{merge}(Y))_{b, t, h \cdot d_k + k} = Y_{b, h, t, k}.
$$

Because both maps are bijections of indices that carry tensor values
unchanged, they are isometries: forward backward of either reshape is the
other. Concretely, $\partial \texttt{split} / \partial X$ contracted with
any cotangent $\mathrm{d}Y$ is just $\texttt{merge}(\mathrm{d}Y)$, and vice
versa. The backward in `mha.py` exploits this by calling `_split_heads` on
the gradient flowing back through `merge` (line&nbsp;89) and `_merge_heads`
on the gradients flowing back through `split` (lines&nbsp;101-103).

### MHA equals $H$ parallel single-head attentions

Define per-head projection slices

$$
W_{Q, h} := W_Q[\,:,\, h \cdot d_k : (h+1) \cdot d_k] \in \mathbb{R}^{d_{\text{model}} \times d_k},
$$

and similarly $W_{K, h}, W_{V, h}$. Then column $h \cdot d_k + k$ of $XW_Q$ is

$$
(X W_Q)_{b, t, h \cdot d_k + k}
= \sum_{d=0}^{d_{\text{model}}-1} X_{b, t, d} \cdot (W_Q)_{d, h \cdot d_k + k}
= (X W_{Q, h})_{b, t, k}.
$$

So the split-then-project pipeline is identical to project-then-split:

$$
\texttt{split}(XW_Q)_{b, h, t, k} = (X W_{Q, h})_{b, t, k}.
$$

Now feed each head through the section&nbsp;4 scaled dot-product attention:

$$
O_h = \mathrm{attn}(X W_{Q, h}, X W_{K, h}, X W_{V, h}),
\qquad O_h \in \mathbb{R}^{B \times T \times d_k}.
$$

The merge reshape concatenates these along the feature axis:

$$
\texttt{merge}((O_h)_{h=0}^{H-1})_{b, t, h \cdot d_k + k} = (O_h)_{b, t, k}.
$$

The final output is one Linear:

$$
Y = \mathrm{merge}(O) \, W_O \in \mathbb{R}^{B \times T \times d_{\text{model}}}.
$$

This factorization was confirmed numerically: instantiating an MHA with
$d_{\text{model}}=12, H=3, T=5$ and rebuilding the output with three
independent `SingleHeadAttention` modules sharing sliced parameters produced
the same tensor (max absolute difference $0$).

### Backward, head by head

Given $\mathrm{d}Y$:

1. **$W_O$ backward**: $Y = \mathrm{merge}(O) W_O$ is a Linear, so

   $$
   \mathrm{d}W_O = \mathrm{merge}(O)^{\top} \mathrm{d}Y,
   \qquad
   \mathrm{d}\mathrm{merge}(O) = \mathrm{d}Y \, W_O^{\top}.
   $$

   (`mha.py`.)

2. **Reshape**: apply `split` to $\mathrm{d}\mathrm{merge}(O)$ to get
   $\mathrm{d}O_h$ for each head. This is the `_split_heads` call.

3. **Per-head section&nbsp;4 backward**: for each $h$, run the single-head
   backward of section&nbsp;4 with $V_h$, $A_h$, $K_h$, $Q_h$ in place of
   $V, A, K, Q$. This yields $\mathrm{d}Q_h, \mathrm{d}K_h, \mathrm{d}V_h$ of
   shape $(B, T, d_k)$ each. The code does this in parallel across the head
   axis using 4D matmul (`mha.py`).

4. **Reassemble**: merge the per-head gradients back into the original
   layout, $\mathrm{d}(X W_Q) = \mathrm{merge}((\mathrm{d}Q_h)_h)$, then
   apply the Linear backward of $Q = X W_Q$:

   $$
   \mathrm{d}W_Q = X^{\top} \, \mathrm{d}(X W_Q),
   $$

   and likewise for $W_K, W_V$. The contributions to $\mathrm{d}X$ from the
   three projections sum:

   $$
   \mathrm{d}X = \mathrm{d}(X W_Q) W_Q^{\top} + \mathrm{d}(X W_K) W_K^{\top} + \mathrm{d}(X W_V) W_V^{\top}.
   $$

   (`mha.py`.)

### Numerical check (sketch)

`test_mha_projection_gradients_match_numerical` checks $\mathrm{d}W_Q,
\mathrm{d}W_K, \mathrm{d}W_V, \mathrm{d}W_O$ for an MHA with $d_{\text{model}}
= 8, H = 2$ against central differences. `test_mha_dx_matches_numerical`
checks $\mathrm{d}X$ similarly. Both pass with relative error well below
$10^{-5}$.

Because the factorization above is exact, the matrix-form code does not loop
over heads: each step ($A V$, the softmax jacobian, $QK^{\top}$, etc.) is
done with a 4D matmul that contracts over the appropriate axis, and the
result is identical to running $H$ independent single-head backwards.

## 6. Adam update rule and bias correction

**Implementation:** [`src/transformer/optim.py`](../src/transformer/optim.py) (`Adam`, with `step` at `optim.py`).
**Numerical check:** [`tests/test_optim.py::test_adam_converges_on_quadratic`](../tests/test_optim.py),
`::test_adam_bias_correction_first_step_matches_grad`.

Adam (Kingma & Ba, 2014) maintains two running averages per parameter, $m_t$
for the gradient and $v_t$ for the squared gradient. Initialising both to
zero introduces a transient bias, and the "bias correction" step is the fix.
This section derives where the bias comes from and why dividing by
$1 - \beta^{t}$ removes it exactly.

### The update rule

For each step $t \geq 1$ with gradient $g_t$:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t,
\qquad
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^{2},
$$

with $m_0 = 0$ and $v_0 = 0$. Then bias-correct,

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^{t}},
\qquad
\hat{v}_t = \frac{v_t}{1 - \beta_2^{t}},
$$

and update,

$$
\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}.
$$

In `optim.py` (`step` method) this is one loop over parameters, applying
exactly these five lines per tensor.

### Why bias correction is needed

Unroll the recursion for $m_t$ given $m_0 = 0$:

$$
m_t = (1 - \beta_1) \sum_{k=1}^{t} \beta_1^{t-k} g_k.
$$

Take expectations under the assumption that gradients have a stationary mean
$\mathbb{E}[g_k] = \mu$ (this is the "constant gradient" regime that approximates
early training). Pull the expectation through the linear combination:

$$
\mathbb{E}[m_t]
= (1 - \beta_1) \mu \sum_{k=1}^{t} \beta_1^{t-k}
= (1 - \beta_1) \mu \cdot \frac{1 - \beta_1^{t}}{1 - \beta_1}
= \mu \cdot (1 - \beta_1^{t}).
$$

So $m_t$ underestimates $\mu$ by exactly the factor $(1 - \beta_1^{t})$, and
the underestimate is worst at small $t$ (when $\beta_1^{t}$ is close to $1$).
Dividing by $1 - \beta_1^{t}$ gives an unbiased estimator,

$$
\mathbb{E}[\hat{m}_t] = \mu.
$$

The same argument applies to $v_t$ as an estimator of $\mathbb{E}[g^{2}]$:
unbiased estimator is $\hat{v}_t = v_t / (1 - \beta_2^{t})$.

Direct verification (during the derivation of this section): with $\beta_1 =
0.9, \mu = 0.7$ and $m_0 = 0$, the recursive value of $m_t$ matches
$(1 - \beta_1^{t}) \mu$ exactly for $t = 1, \ldots, 7$, and the
bias-corrected $\hat{m}_t$ is the constant $0.7$ at every step.

### The first-step "sign of the gradient" property

Plugging $t = 1$ into the formulas with $m_0 = v_0 = 0$:

$$
m_1 = (1 - \beta_1) g_1,
\qquad
v_1 = (1 - \beta_2) g_1^{2},
$$

$$
\hat{m}_1 = g_1,
\qquad
\hat{v}_1 = g_1^{2},
$$

$$
\theta_1 - \theta_0 = -\alpha \cdot \frac{g_1}{|g_1| + \varepsilon}
\approx -\alpha \cdot \mathrm{sign}(g_1) \quad \text{when } \varepsilon \ll |g_1|.
$$

So the first Adam step has magnitude approximately $\alpha$ in every
coordinate, regardless of the gradient's magnitude. This is the property
that lets Adam tolerate poorly scaled gradients near initialisation. The
test `test_adam_bias_correction_first_step_matches_grad` confirms this
exactly with $\varepsilon = 0$.

### Decoupled weight decay (AdamW)

Adding L2 regularisation $\tfrac{\lambda}{2} \lVert \theta \rVert^{2}$ to the
loss would put $\lambda \theta$ inside the gradient, where Adam's EMA and
$\sqrt{\hat v_t}$ normalisation would distort it: weights with small
historical $\sqrt{\hat v_t}$ would be decayed disproportionately. Loshchilov
& Hutter (2019) decouple the decay from the gradient:

$$
\theta_t = \theta_{t-1} - \alpha \left[
  \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} + \lambda \, \theta_{t-1}
\right].
$$

The decay term uses $\theta_{t-1}$ directly, never enters $m_t$ or $v_t$, and
is applied alongside (not through) the Adam step. With $\lambda = 0$ this
reduces to plain Adam. Following the GPT-style convention, `Adam.step`
applies decay only to 2D parameters (weight matrices) and skips biases and
LayerNorm $\gamma, \beta$.

### Epsilon placement

The implementation in `optim.py` writes

$$
\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
$$

(i.e., $\varepsilon$ is added outside the square root). This is the PyTorch
convention from the original paper. An alternative,
$\hat{m}_t / \sqrt{\hat{v}_t + \varepsilon}$ (TensorFlow's older convention),
is numerically very similar but technically different and not used here.

### Gradient clipping

`Adam.step` also applies *global-norm* gradient clipping before forming
$m_t, v_t$. Let $g = (g_1, g_2, \ldots, g_P)$ be the concatenation of every
parameter's gradient. Compute its total L2 norm

$$
\lVert g \rVert_2 = \sqrt{\sum_{i=1}^{P} g_i^{2}}.
$$

If $\lVert g \rVert_2 > \tau$ (where $\tau$ is `max_norm`), every gradient is
rescaled by the same factor $\tau / \lVert g \rVert_2$, otherwise the
gradients pass through unchanged. This is *direction-preserving*: it shrinks
the step magnitude in pathological regimes without rotating it.

Per-element clipping (e.g. $\lvert g_i \rvert \leq c$) would silently change
the gradient *direction* whenever any single coordinate exceeds the cap,
which can mis-align Adam's EMA estimates with the actual loss surface. We
use global-norm clipping for that reason, matching the standard practice
since GPT-2. Set `max_norm=None` to disable (and the optimiser convergence
test does, since its quadratic-loss gradients are well-behaved).

### Numerical check (sketch)

`test_adam_converges_on_quadratic` minimises $f(w) = \tfrac{1}{2} \lVert w -
w^\star \rVert^{2}$ from $w_0 = 0$ using Adam with $\alpha = 0.1$,
$\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, no clipping. After
500 steps the loss is below $10^{-6}$ and $w$ matches $w^\star$ to within
$10^{-3}$.

## 7. LR schedules: warmup and cosine decay

**Implementation:** [`src/transformer/schedule.py`](../src/transformer/schedule.py) (`cosine_warmup_lr`).
**Numerical check:** [`tests/test_schedule.py`](../tests/test_schedule.py).

The training loop uses a two-phase learning-rate schedule: linear warmup
followed by cosine decay. Both phases are individually trivial; what makes
them worth a section is *why* this specific shape interacts well with
Adam, and the boundary smoothness property of the cosine half-period.

### The schedule

For step $t = 1, 2, \ldots$ with base rate $\alpha$, warmup length $T_w$,
total length $T$, and floor $\alpha_{\min}$:

$$
\alpha_t =
\begin{cases}
\alpha \cdot \dfrac{t}{T_w}                                    & 1 \leq t \leq T_w, \\[6pt]
\alpha_{\min} + \tfrac{1}{2}(\alpha - \alpha_{\min})
  \bigl(1 + \cos(\pi \, p_t)\bigr)                              & T_w < t < T, \\[6pt]
\alpha_{\min}                                                  & t \geq T,
\end{cases}
\qquad
p_t = \frac{t - T_w}{T - T_w}.
$$

At $t = T_w$ we have $p_t = 0$ and $\cos 0 = 1$, so $\alpha_{T_w} = \alpha$:
the warmup hands off to the cosine phase at the peak value, continuously.
At $t = T$ we have $p_t = 1$ and $\cos \pi = -1$, so $\alpha_T = \alpha_{\min}$.

### Why warmup, specifically with Adam

From section&nbsp;6, Adam's first step has the property

$$
\theta_1 - \theta_0
= -\alpha \cdot \frac{g_1}{\lvert g_1 \rvert + \varepsilon}
\approx -\alpha \cdot \mathrm{sign}(g_1).
$$

The magnitude of this update is approximately $\alpha$ in every coordinate,
*independent* of the actual gradient scale. That is desirable in steady
state (it makes Adam robust to poorly scaled gradients) but dangerous at
initialisation: $m_t$ and $v_t$ are zero-biased early on, and the
direction encoded in $\mathrm{sign}(g_1)$ is informed only by a
single mini-batch's gradient. Taking $T_w$ updates at reduced rates lets
the EMA estimates accumulate signal before steps reach full size. For SGD,
where the update magnitude scales with $\lvert g \rvert$, this matters less.

### Cosine boundary smoothness

Differentiate the cosine branch with respect to $t$:

$$
\frac{\mathrm{d}\alpha_t}{\mathrm{d}t}
= -\frac{\pi (\alpha - \alpha_{\min})}{2(T - T_w)}
  \sin(\pi \, p_t).
$$

Because $\sin 0 = \sin \pi = 0$, this derivative vanishes at *both*
endpoints: the schedule is $C^1$-continuous when joining warmup at $t = T_w$
(if we set the warmup derivative to match) and lands at $\alpha_{\min}$
with zero slope at $t = T$. Linear or step schedules have a discontinuity
in $\alpha_t$ itself or its derivative; cosine has neither, which is the
empirical motivation for using it.

### Numerical check (sketch)

The schedule is a pure scalar function, so the test file pins five
checkpoints rather than a gradient: the first warmup step gives
$\alpha / T_w$, the warmup boundary $t = T_w$ recovers $\alpha$, the
midpoint $p_t = 1/2$ gives $\alpha_{\min} + \tfrac{1}{2}(\alpha - \alpha_{\min})$
(since $\cos(\pi/2) = 0$), $t = T$ gives $\alpha_{\min}$, and $t > T$ is
clamped to $\alpha_{\min}$.

## 8. Inverted dropout: expectation and gradient

**Implementation:** [`src/transformer/dropout.py`](../src/transformer/dropout.py) (`Dropout`).
**Numerical check:** [`tests/test_dropout.py`](../tests/test_dropout.py).

Dropout is a single elementwise multiplication; the value of writing it out
is to see *why* the $1/(1-p)$ scale is chosen and why backward needs no
extra cache beyond the forward mask.

### Forward

In train mode, draw a Bernoulli mask $b \in \{0, 1\}^{D}$ with
$\Pr(b_i = 1) = 1 - p$, then form the scaled mask

$$
m_i = \frac{b_i}{1 - p},
\qquad
y_i = m_i \, x_i.
$$

In eval mode, $y = x$ identically: no mask is sampled, nothing is scaled.

### Why the $1/(1-p)$ scale (expectation preservation)

Take the expectation over the Bernoulli draw, treating $x_i$ as fixed:

$$
\mathbb{E}[m_i] = \frac{\mathbb{E}[b_i]}{1 - p} = \frac{1 - p}{1 - p} = 1,
\qquad
\mathbb{E}[y_i] = \mathbb{E}[m_i] \, x_i = x_i.
$$

So the *expected* train-time activation equals the eval-time activation.
That is what justifies running inference with the unmodified network: no
test-time rescaling needed, because at train time we already absorbed the
correction. This is the "inverted dropout" convention.

The alternative (the original Hinton 2012 form) uses $m_i = b_i$ at train
time, then scales activations by $1-p$ at eval. The two conventions are
mathematically equivalent on expectation; inverted dropout is preferred
because it leaves inference untouched.

### Backward

Since $y_i = m_i x_i$ with $m_i$ a constant during the backward pass (the
same draw used in forward), $\partial y_i / \partial x_i = m_i$, so

$$
\mathrm{d}x_i = m_i \, \mathrm{d}y_i.
$$

This is `dropout.py`'s one-line backward. No additional cache is needed
because forward already stored $m$. In eval mode $m$ is absent, and
backward is identity.

### Numerical check (sketch)

`test_dropout_preserves_expectation` fills a $500 \times 500$ tensor with
$x = 7$ and applies dropout with $p = 0.3$. Empirical mean of the output
matches $7$ to within $0.05$, confirming $\mathbb{E}[m \odot x] = x$
across the realised mask.

## 9. Weight tying: combining two gradient contributions

**Implementation:** [`src/transformer/model.py`](../src/transformer/model.py) (`Transformer`, `tie_weights=True`).
**Numerical check:** [`tests/test_model.py::test_tie_weights_combined_gradient_matches_numerical`](../tests/test_model.py).

Weight tying shares storage between the embedding matrix and the output
projection: $W_{\text{out}} = W_{\text{emb}}^{\top}$, pointing at the same
buffer. The loss $L$ depends on $W$ through *two* paths, so the gradient
$\nabla_W L$ is the sum of the two contributions. This section spells the
sum out and confirms it numerically.

### Setup

Let $W \in \mathbb{R}^{V \times D}$ be the shared parameter ($V$ vocab
size, $D$ model width). The model uses it twice:

$$
\text{(1) embedding lookup: } \quad E_{b, t, :} = W_{\text{idx}(b, t), :},
$$

$$
\text{(2) output projection: } \quad
\text{logits}_{b, t, :} = X^{\text{final}}_{b, t, :} \, W^{\top} + c,
$$

where $X^{\text{final}}$ is the hidden state after the final LayerNorm and
$c$ is the output bias (a separate parameter, unaffected by tying).

### Gradient from each path

**Embedding path.** Only one row of $W$ is touched per token, so the
gradient is a sparse accumulation:

$$
(\mathrm{d}W_{\text{emb}})_{i, :} = \sum_{(b, t) : \text{idx}(b, t) = i} (\mathrm{d}E)_{b, t, :}.
$$

In code this is `np.add.at(dW, indices, dE)`. (Buffered `dW[indices] += dE`
would double-count when the same index appears more than once in a
batch.)

**Output projection path.** $\text{logits} = X W^{\top} + c$ is a plain
Linear with input $X$ and weight $W^{\top}$. Using the Linear backward
identity from section&nbsp;1 (transposing for our orientation):

$$
\mathrm{d}W_{\text{out}} = (\mathrm{d}\,\text{logits})^{\top} \, X^{\text{final}}
\in \mathbb{R}^{V \times D}.
$$

This is the gradient w.r.t. $W^{\top}$ reshaped as a $V \times D$ matrix,
matching $W$'s orientation directly. (In `Linear.backward` the stored
$\mathrm{d}W$ is in the $D \times V$ orientation, so we transpose.)

### Combining

Total derivative of $L$ with respect to the shared $W$ is the sum:

$$
\boxed{
\nabla_W L = \mathrm{d}W_{\text{emb}} + (\mathrm{d}W_{\text{out}})^{\top}.
}
$$

This is just the chain rule applied twice: $L$ depends on $W$ through both
paths, and the partial derivatives along each path add (multivariate
chain rule, independent paths).

In `model.py`'s `backward`, after the normal pass populates
`embedding.dW` and `output_proj.dW`, one line folds them:

```python
if self.tie_weights:
    self.embedding.dW += self.output_proj.dW.T
```

The Adam list `params()` then exposes only `(embedding, "W")`, so the
optimiser updates the shared buffer once with the combined gradient.

### Numerical check (sketch)

`test_tie_weights_combined_gradient_matches_numerical` runs forward,
backward on a tied model, reads the combined `embedding.dW`, and matches
it against the central-difference derivative of the loss w.r.t. the
shared buffer ($V = 5, D = 4$). Relative error stays below $10^{-5}$.
