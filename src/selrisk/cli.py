"""Compatibility entry points route to research unless --demo is explicit."""
import argparse
import json
import sys
from pathlib import Path
from .realdata import ROOT, ensure_dataset
from .research import OUTPUT, main as research_main, lookup_date, load_regional


def entry(name, legacy):
    if "--demo" in sys.argv:
        from .demo import demo_mode
        sys.argv.remove("--demo")
        print("DEMO ONLY: synthetic labels and unvalidated legacy models.")
        with demo_mode():
            legacy()
        return
    if name in ("run_all","train","experiments"):
        research_main()
        if name=="run_all":
            from .web_export import export_web
            export_web()
    elif name=="01_fetch_data":
        print(f"Verified frozen dataset: {ensure_dataset()}; no network or synthetic fallback.")
    elif name in ("export_web","make_map"):
        from .web_export import export_web
        export_web()
    elif name=="evaluate":
        for kind in ("power","stations"):
            p=OUTPUT/f"report_{kind}.json"
            if not p.exists():
                raise SystemExit("Run python scripts/train.py first; evaluation will not silently refit.")
            print(p.read_text(encoding="utf-8"))
    elif name=="predict":
        p=argparse.ArgumentParser(description="Historical regional score, not a probability")
        p.add_argument("--date",required=True)
        p.add_argument("--kind",choices=["power","stations"],default="power")
        a=p.parse_args()
        model=ROOT/f"web/data/model_{a.kind}.json"
        if not model.exists():
            raise SystemExit("Run python scripts/run_all.py first")
        frame,_,_=load_regional(a.kind)
        print(json.dumps(lookup_date(frame,json.loads(model.read_text()),a.date),ensure_ascii=False))
    else:
        raise SystemExit("Legacy generator is disabled by default; use --demo intentionally.")
