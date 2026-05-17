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
