import importlib
import inspect

import numpy as np


def _call_with_best_guess(fn):
    x = np.array([1.0, 1.0, 1.0], dtype=float)
    params = {}
    dt = 0.01
    sig = inspect.signature(fn)
    n = len(sig.parameters)
    for args in ((x, params, dt), (x, dt), (x,)):
        try:
            if len(args) == n:
                return fn(*args)
        except Exception:
            continue
    return None


def test_step_functions_return_shape():
    mod = importlib.import_module(
        "neural_chaos_lab_max.systems".replace(
            "neural_chaos_lab_max", "neural_chaos_lab"
        )
    )
    funcs = [
        getattr(mod, n)
        for n in dir(mod)
        if n.endswith("_step") and callable(getattr(mod, n))
    ]
    assert funcs
    for f in funcs:
        out = _call_with_best_guess(f)
        assert out is not None
        try:
            arr = np.array(out, dtype=float)
            assert arr.shape[0] >= 1
        except Exception:
            assert hasattr(out, "__len__")
