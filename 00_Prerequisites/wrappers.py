import time
from functools import wraps

def timer(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        end = time.perf_counter()
        print(f"{fn.__name__} took {end - start:.6f}s")
        return result
    return wrapper

@timer
def foo(x):
	return x * x

params = {
    "embed.weight": (32000, 768),
    "layer1.q_proj.weight": (768, 768),
    "layer1.k_proj.weight": (768,768),
}

@timer
def count_parameters(params):
    total = 0
    for shape in params.values():
        n = 1
        for dim in shape:
            n *= dim
        total += n
    return total

print(count_parameters(params))

[ i * i for i in range(20) if i % 2 == 0]

hidden_size = config.get("hidden_size", 768)

