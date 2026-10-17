"""Command line: essay-scorer generate | train | score | report | serve."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .service import Scorer

log = logging.getLogger("essay_scorer")
DEFAULT_MODEL = Path("artifacts/model.joblib")


def cmd_generate(args: argparse.Namespace) -> None:
    from .synthetic import GeneratorConfig, generate_essays

    df = generate_essays(GeneratorConfig(n_essays=args.essays, seed=args.seed))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log.info("wrote %d synthetic essays to %s", len(df), args.out)


def cmd_train(args: argparse.Namespace) -> None:
    from .evaluate import Dataset, agreement_table, cross_validated_bands, fit_model
    from .synthetic import GeneratorConfig, generate_essays

    ds = Dataset.from_frame(generate_essays(GeneratorConfig(n_essays=args.essays, seed=args.seed)))
    table = agreement_table(cross_validated_bands(ds, args.kind, by_prompt=True), ds.df)
    print(table.round(3).to_string())
    fit_model(ds, args.kind).save(args.model)
    log.info("cross-validated by prompt (above); saved a model fit on all essays to %s", args.model)


def _scorer(model_path: Path) -> Scorer:
    from .models import EssayModel
    from .providers import provider_from_env

    if not model_path.exists():
        raise SystemExit(f"{model_path} not found: run `essay-scorer train` first")
    return Scorer(EssayModel.load(model_path), provider_from_env())


def cmd_score(args: argparse.Namespace) -> None:
    result = _scorer(args.model).score(args.essay.read_text(), args.prompt)
    print(json.dumps(result.to_dict(), indent=2))


def cmd_report(args: argparse.Namespace) -> None:
    from .report import write

    study = write(args.out)
    print(study.by_prompt["ridge"].round(3).to_string())
    log.info("wrote %s and charts in %s", args.out / "results.md", args.out / "img")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(_scorer(args.model)), host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="essay-scorer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write synthetic essays as CSV")
    gen.add_argument("--out", type=Path, default=Path("data/synthetic_essays.csv"))
    train = sub.add_parser("train", help="cross-validate by prompt, then fit and save a model")
    train.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    train.add_argument("--kind", choices=["ridge", "hgb", "hgb_content"], default="ridge")
    for p in (gen, train):
        p.add_argument("--essays", type=int, default=2000)
        p.add_argument("--seed", type=int, default=7)

    score = sub.add_parser("score", help="score one essay from a text file")
    score.add_argument("essay", type=Path, help="a .txt file with the essay")
    score.add_argument("--prompt", required=True, help="the Task 2 question")
    score.add_argument("--model", type=Path, default=DEFAULT_MODEL)

    report = sub.add_parser("report", help="write docs/results.md and docs/img/*.png")
    report.add_argument("--out", type=Path, default=Path("docs"))

    serve = sub.add_parser("serve", help="run the API")
    serve.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    gen.set_defaults(func=cmd_generate)
    train.set_defaults(func=cmd_train)
    score.set_defaults(func=cmd_score)
    report.set_defaults(func=cmd_report)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
