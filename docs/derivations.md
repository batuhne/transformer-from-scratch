# Derivations

Hand-derived gradients for every parameterized operation in `src/transformer/`,
each paired with a numerical check (central differences, relative error
`< 1e-5`) that lives in `tests/`.

The convention throughout: scalars in lowercase, vectors in bold lowercase,
matrices in uppercase. Gradients of a scalar loss `L` with respect to a tensor
`X` are written `dX` (the same shape as `X`). All sums are over indices made
explicit by the bounds.

## Table of contents

- [1. Softmax and cross-entropy: the combined gradient](#1-softmax-and-cross-entropy-the-combined-gradient)
- [2. LayerNorm backward](#2-layernorm-backward)
- [3. Scaled dot-product attention: gradients in matrix form](#3-scaled-dot-product-attention-gradients-in-matrix-form)

## 1. Softmax and cross-entropy: the combined gradient

**Implementation:** `src/transformer/linear.py` (`softmax`, `cross_entropy_loss`).
**Numerical check:** `tests/test_linear.py::test_softmax_cross_entropy_gradient_matches_numerical`.

### Setup

Let $z \in \mathbb{R}^{C}$ be the logits for one example and $t \in \{0, \dots, C-1\}$
its target class. The softmax produces a probability vector

$$
p_i = \operatorname{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{C} \exp(z_j)},
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

This matches `linear.py:31-33`:

```python
dlogits = probs.copy()
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

## 2. LayerNorm backward

**Implementation:** `src/transformer/layernorm.py`.
**Numerical check:** `tests/test_layernorm.py::test_layernorm_dx_matches_numerical`
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
batch axis (`layernorm.py:39-40`). $\gamma$ and $\beta$ are shared across all
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

This matches `layernorm.py:44-48` line for line:

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

## 3. Scaled dot-product attention: gradients in matrix form

**Implementation:** `src/transformer/attention.py`.
**Numerical check:** `tests/test_attention.py::test_attention_projection_gradients_match_numerical`
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
A = \operatorname{softmax}(\tilde{S}) \text{ row-wise},
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
From section&nbsp;1, the single-row softmax Jacobian is

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

This matches `attention.py:75-77`:

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

This is `attention.py:78` (`d_scores /= np.sqrt(self.d_k)`).

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

Both lines match `attention.py:80-81`.

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

This is `attention.py:83-88`. The reshape `x_flat = x.reshape(-1, d_model)`
just folds the batch and sequence axes together so the same identity applies.

### Numerical check (sketch)

The test instantiates a `SingleHeadAttention(d_model=8, d_k=4)` and verifies
each of $\mathrm{d}W_Q$, $\mathrm{d}W_K$, $\mathrm{d}W_V$, $\mathrm{d}x$
against central differences. During the derivation here we also confirmed
$\mathrm{d}Q$, $\mathrm{d}K$, $\mathrm{d}V$ directly at the pre-projection
level on $T=4, d_k=3$ random tensors: all three matched numerically with
relative error below $1.3 \times 10^{-9}$.
