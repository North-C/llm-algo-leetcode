# Chapter 0 Practice and Explanations

This document is the running study notebook for Chapter 0. It records exercise sets, prerequisite concepts, common mistakes, corrections, and reference implementations.

Status labels:

- `Pending`: the exercise has been assigned but not reviewed yet.
- `Reviewed`: the user's answer has been checked and discussed.
- `Reference added`: a reference solution or canonical explanation has been added.

---

## Exercise 001: Python Essentials for LLM Code

Date: 2026-05-29

Status: Reference added

### Learning Goal

Build fluency with Python patterns that appear frequently in LLM codebases: list comprehensions, dictionary config access, shape-based parameter counting, decorators, and config-driven model construction.

### Prerequisite Concepts

- List comprehension syntax: `[expr for item in iterable if condition]`
- Dictionary safe reads with `dict.get(key, default)`
- Tensor/model parameter shapes and element counts
- Decorators as higher-order functions
- Difference between model parameters and hyperparameters

### Questions

1. Use a list comprehension to generate the squares of all even numbers from `0` through `19`.
2. Given `config = {"num_layers": 12, "num_heads": 12}`, safely read `hidden_size`; use `768` if it is missing.
3. Implement `count_parameters(params)` for a dictionary whose values are shape tuples.
4. Implement a `timer` decorator that prints elapsed runtime while preserving the wrapped function's return value.
5. Explain why LLM code often stores hyperparameters in a config dictionary instead of hardcoding every value.

### Key Concepts and Pitfalls

- In Python, the expression comes before the `for` clause in a list comprehension.
- `range(20)` covers `0` through `19`.
- `config["hidden_size"]` raises `KeyError` if the key is missing; `config.get("hidden_size", 768)` is safer for defaults.
- Model weights are trainable parameters. Values such as `hidden_size`, `num_layers`, and `num_heads` are usually hyperparameters fixed before training.
- A decorator receives a function and returns a replacement function, usually called `wrapper`.

### Reference Implementation

```python
import math
import time
from functools import wraps

even_squares = [i * i for i in range(20) if i % 2 == 0]

config = {"num_layers": 12, "num_heads": 12}
hidden_size = config.get("hidden_size", 768)

params = {
    "embed.weight": (32000, 768),
    "layer1.q_proj.weight": (768, 768),
    "layer1.k_proj.weight": (768, 768),
}

def count_parameters(params):
    total = 0
    for shape in params.values():
        total += math.prod(shape)
    return total

def timer(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        end = time.perf_counter()
        print(f"{fn.__name__} took {end - start:.6f}s")
        return result
    return wrapper

print(count_parameters(params))  # 25780224
```

### Explanation

Config dictionaries make model code reproducible and flexible. They allow the same model class to instantiate different model sizes, save and reload architecture choices, and run experiments without editing model source code. These config values should not be confused with trainable weights, which are updated by backpropagation.

---

## Exercise 002: NumPy Broadcasting and Einsum

Date: 2026-05-29

Status: Pending

### Learning Goal

Understand NumPy broadcasting and `einsum` well enough to express the core tensor operations used by attention, especially `QK^T`.

### Prerequisite Concepts

- NumPy arrays have explicit shapes such as `(batch, sequence, hidden)`.
- Broadcasting aligns shapes from the trailing dimensions.
- `keepdims=True` preserves reduced dimensions, which makes later broadcasting predictable.
- `softmax` should subtract the row maximum before exponentiation for numerical stability.
- `einsum` names tensor dimensions with letters and defines how dimensions are matched, reduced, and ordered.

### Questions

```python
import numpy as np
```

1. Broadcasting bias addition:

```python
x = np.array([
    [1, 2, 3],
    [4, 5, 6],
])

bias = np.array([10, 20, 30])

# TODO: produce
# [[11, 22, 33],
#  [14, 25, 36]]
```

2. Row-wise normalization:

```python
x = np.array([
    [1.0, 2.0, 3.0],
    [10.0, 20.0, 30.0],
])

# TODO: subtract each row's own mean while keeping result shape (2, 3)
```

3. Numerically stable softmax along the last dimension:

```python
x = np.array([
    [1.0, 2.0, 3.0],
    [1.0, 1.0, 1.0],
])

def softmax(x):
    # TODO
    pass
```

4. Matrix multiplication with `einsum`, without `@` or `np.matmul`:

```python
A = np.random.randn(2, 3)
B = np.random.randn(3, 4)

# TODO: C shape should be (2, 4)
```

5. Attention scores with `einsum`:

```python
B = 2       # batch size
Q = 4       # query length
K = 5       # key length
D = 8       # head dimension

q = np.random.randn(B, Q, D)
k = np.random.randn(B, K, D)

# TODO: scores = QK^T / sqrt(D)
# Expected shape: (B, Q, K)
```

### Key Concepts and Pitfalls

- Bias shape `(3,)` broadcasts across the first dimension of `x` with shape `(2, 3)`.
- For row-wise mean, use `axis=1` and `keepdims=True`; otherwise the mean has shape `(2,)`, which does not broadcast as intended against `(2, 3)`.
- Stable softmax uses `x - x.max(axis=-1, keepdims=True)`.
- Matrix multiplication `A[i, j] * B[j, k] -> C[i, k]` is written as `np.einsum("ij,jk->ik", A, B)`.
- Attention scores use `q[b, q, d] * k[b, k, d] -> scores[b, q, k]`, written as `np.einsum("bqd,bkd->bqk", q, k)`.

### Reference Implementation

To be added after the user's answer is reviewed.

