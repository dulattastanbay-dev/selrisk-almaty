"""Explicit legacy demonstration isolation. Never used for real-data training."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_ACTIVE=ContextVar("selrisk_demo",default=False)


def require_demo():
    if not _ACTIVE.get():
        raise RuntimeError("Synthetic/legacy workflow is disabled. Use the explicit --demo command.")


def demo_only(fn):
    @wraps(fn)
    def guarded(*args, **kwargs):
        require_demo()
        return fn(*args, **kwargs)
    return guarded


@contextmanager
def demo_mode():
    from .config import load_config
    cfg=load_config()
    old={k:cfg["paths"][k] for k in ("sample","processed","outputs","models","raw")}
    token=_ACTIVE.set(True)
    try:
        for k,path in old.items():
            cfg["paths"][k]=path.rstrip("/")+"/demo"
        yield
    finally:
        cfg["paths"].update(old)
        _ACTIVE.reset(token)
