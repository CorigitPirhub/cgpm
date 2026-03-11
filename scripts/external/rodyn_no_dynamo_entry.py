from __future__ import annotations

import functools
import os
import runpy
import sys
from pathlib import Path

import torch
import torch._compile as torch_compile


def _noop_disable(fn=None, recursive=True):
    if fn is not None:
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            return fn(*args, **kwargs)
        return inner
    return lambda wrapped: wrapped


def main() -> None:
    torch_compile._disable_dynamo = _noop_disable
    torch._disable_dynamo = _noop_disable
    import torch.optim as torch_optim
    from torch.optim.optimizer import Optimizer
    for _name in dir(Optimizer):
        _obj = getattr(Optimizer, _name, None)
        if hasattr(_obj, '__wrapped__'):
            setattr(Optimizer, _name, _obj.__wrapped__)
    for _name in dir(torch_optim):
        _cls = getattr(torch_optim, _name, None)
        if isinstance(_cls, type) and issubclass(_cls, Optimizer):
            _step = getattr(_cls, 'step', None)
            if hasattr(_step, '__wrapped__'):
                _orig = _step.__wrapped__
                def _make_step(orig_fn):
                    def _patched_step(self, *args, **kwargs):
                        with torch.no_grad():
                            return orig_fn(self, *args, **kwargs)
                    return _patched_step
                _cls.step = _make_step(_orig)
    if len(sys.argv) < 2:
        raise SystemExit('usage: rodyn_no_dynamo_entry.py path/to/rodynslam.py [args...]')
    target = Path(sys.argv[1]).resolve()
    os.chdir(str(target.parent))
    if str(target.parent) not in sys.path:
        sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
