"""Real-data validation by default; legacy synthetic pipeline only with --demo."""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    args, rest = parser.parse_known_args(argv)
    if not args.demo:
        from selrisk.scientific import main as validate
        return validate(rest)
    if rest:
        parser.error("--demo cannot be combined with real-data arguments")
    from selrisk.config import load_config
    cfg = load_config()
    for name in ("models", "outputs"):
        cfg["paths"][name] = str(pathlib.Path(cfg["paths"][name]) / "demo")
    import make_sample_data
    import train
    import evaluate
    import experiments
    import make_map
    print("DEMO ONLY: synthetic/mixed inputs; metrics are not scientific validation.")
    for step in (make_sample_data.main, lambda: train.main(demo=True), evaluate.main,
                 experiments.main, make_map.main):
        step()
    print("Demo finished; model/output paths are isolated under demo/. No real-data validation claimed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
