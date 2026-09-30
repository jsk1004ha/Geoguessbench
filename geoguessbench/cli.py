"""Command-line entrypoints. Real benchmark commands refuse synthetic fixtures."""
import argparse
import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from .agent import ChatAgent, run_agent
from .dataset import Dataset, digest_json
from .metrics import compare, summarize, telemetry_summary
from .schema import Protocol
from .sources import fetch_mapillary, fetch_osv, import_csv


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def export_run(db, run_id, private=False):
    with closing(sqlite3.connect(f"file:{Path(db).resolve().as_posix()}?mode=ro", uri=True)) as connection:
        record = connection.execute("SELECT snapshot FROM runs WHERE id=?", (run_id,)).fetchone()
    if not record:
        raise ValueError("run not found")
    run = json.loads(record[0])
    if run["cursor"] != len(run["order"]):
        raise ValueError("run is incomplete")
    public = {k: v for k, v in run.items() if k not in {"order", "cursor", "active", "rows", "traces"}}
    public["metrics"] = summarize(run["rows"], samples=run["protocol"]["bootstrap_samples"],
                                   seed=run["protocol"]["seed"])
    public["usage"] = telemetry_summary(run["traces"])
    if private:
        public.update(rows=run["rows"], traces=run["traces"])
    return public


def leaderboard(paths):
    groups, seen = {}, set()
    for path in paths:
        report = read_json(path)
        if report.get("fixture"):
            raise ValueError("synthetic fixture results are not eligible for a leaderboard")
        if report["id"] in seen:
            raise ValueError("duplicate run in leaderboard")
        seen.add(report["id"])
        settings = {k: v for k, v in report.get("settings", {}).items() if k not in {"model", "base_url"}}
        group_id = digest_json([report["dataset_sha256"], report["protocol_sha256"], settings])
        group = groups.setdefault(group_id, {"comparison_group": group_id,
                      "dataset_sha256": report["dataset_sha256"], "protocol_sha256": report["protocol_sha256"],
                      "inference_settings": settings, "entries": []})
        group["entries"].append({"run_id": report["id"], "model": report["model"],
                                  "metrics": report["metrics"]})
    for group in groups.values():
        group["entries"].sort(key=lambda item: item["metrics"]["mean_score"], reverse=True)
    return {"verification": "local exported reports; not a cryptographically certified public leaderboard",
            "groups": list(groups.values())}


def parser():
    root = argparse.ArgumentParser(description="Geoguessbench: real-image, server-scored geolocation evaluation")
    commands = root.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-osv", help="download a seeded real OSV-5M test subset using HTTP ranges")
    fetch.add_argument("--out", required=True)
    fetch.add_argument("--limit", type=int, default=100)
    fetch.add_argument("--seed", type=int, default=42)
    fetch.add_argument("--revision", default="main")
    mapillary = commands.add_parser("fetch-mapillary", help="build actual panoramic sequence routes")
    mapillary.add_argument("--out", required=True)
    mapillary.add_argument("--sequence", action="append", required=True)
    mapillary.add_argument("--max-frames", type=int, default=40)
    mapillary.add_argument("--max-gap-m", type=float, default=75)
    imp = commands.add_parser("import-csv", help="import your own licensed, geotagged real imagery")
    imp.add_argument("--csv", required=True)
    imp.add_argument("--images", required=True)
    imp.add_argument("--out", required=True)
    imp.add_argument("--name", required=True)
    imp.add_argument("--license", required=True)
    validate = commands.add_parser("validate", help="check hashes, images, graph, coordinates and split leakage")
    validate.add_argument("--data", required=True)
    validate.add_argument("--track", choices=["nmpz", "nm", "moving"], default="nmpz")
    validate.add_argument("--split", choices=["train", "dev", "test"], default="test")
    serve = commands.add_parser("serve", help="start the private evaluator and local browser interface")
    serve.add_argument("--data", required=True)
    serve.add_argument("--config", required=True)
    serve.add_argument("--db", default="private/runs.sqlite")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    evaluate = commands.add_parser("evaluate", help="run a real vision model against the separate evaluator")
    evaluate.add_argument("--server", default="http://127.0.0.1:8000")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--base-url", default="https://api.openai.com/v1")
    evaluate.add_argument("--key-env", default="MODEL_API_KEY")
    evaluate.add_argument("--max-tokens", type=int, default=1024)
    evaluate.add_argument("--token-parameter", choices=["max_tokens", "max_completion_tokens"], default="max_completion_tokens")
    evaluate.add_argument("--temperature", type=float)
    evaluate.add_argument("--image-history", type=int, default=4)
    evaluate.add_argument("--timeout", type=float, default=90)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--resume", help="run ID; requires the original output checkpoint")
    export = commands.add_parser("export", help="export a finished run from evaluator-owned storage")
    export.add_argument("--db", default="private/runs.sqlite")
    export.add_argument("--run", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--private", action="store_true", help="include ground truth/traces; do not publish hidden-test exports")
    comp = commands.add_parser("compare", help="paired cluster-bootstrap comparison of private exports")
    comp.add_argument("--left", required=True)
    comp.add_argument("--right", required=True)
    comp.add_argument("--out", required=True)
    leader = commands.add_parser("leaderboard", help="group compatible completed reports; never combine tracks")
    leader.add_argument("reports", nargs="+")
    leader.add_argument("--out", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "fetch-osv":
            print(fetch_osv(args.out, limit=args.limit, seed=args.seed, revision=args.revision))
        elif args.command == "fetch-mapillary":
            print(fetch_mapillary(args.out, args.sequence, os.environ.get("MAPILLARY_ACCESS_TOKEN", ""),
                                  max_frames=args.max_frames, max_gap_m=args.max_gap_m))
        elif args.command == "import-csv":
            print(import_csv(args.csv, args.images, args.out, name=args.name, license_name=args.license))
        elif args.command == "validate":
            data = Dataset(args.data)
            data.validate_track(args.track, args.split)
            print(json.dumps({"valid": True, "nodes": len(data.nodes), "rounds": len(data.rounds),
                              "dataset_sha256": data.fingerprint}, indent=2))
        elif args.command == "serve":
            import uvicorn
            from .server import create_app
            token = os.environ.get("BENCH_TOKEN", "")
            if args.host not in {"127.0.0.1", "localhost", "::1"} and not token:
                raise ValueError("BENCH_TOKEN is required for non-loopback binding")
            app = create_app(Dataset(args.data), Protocol.model_validate(read_json(args.config)), args.db, token)
            uvicorn.run(app, host=args.host, port=args.port, workers=1)
        elif args.command == "evaluate":
            agent = ChatAgent(args.model, args.base_url, os.environ.get(args.key_env, ""),
                              max_tokens=args.max_tokens, token_parameter=args.token_parameter,
                              temperature=args.temperature, image_history=args.image_history, timeout=args.timeout)
            report = run_agent(args.server, agent, args.out, token=os.environ.get("BENCH_TOKEN", ""), resume=args.resume)
            print(json.dumps(report["metrics"], indent=2))
        elif args.command == "export":
            write_json(args.out, export_run(args.db, args.run, args.private))
            if args.private:
                print("PRIVATE export contains ground truth: keep it off public repositories and model machines.")
        elif args.command == "compare":
            write_json(args.out, compare(read_json(args.left), read_json(args.right)))
        elif args.command == "leaderboard":
            write_json(args.out, leaderboard(args.reports))
        return 0
    except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # External API errors can contain credential-bearing URLs. Avoid printing response bodies.
        print(f"External operation failed ({type(exc).__name__}); no fallback data/results were generated.", file=sys.stderr)
        return 2
