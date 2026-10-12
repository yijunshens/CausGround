"""Offline RoboCasa CausGround extractor.

This entry point reads the bundled CausGround mechanism graph and runs the
same text-goal, contract-instantiation, and dependency-backward-expansion
flow used by the online interface. It never calls HTTP.
"""

from __future__ import annotations

from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
sys.path.insert(0, str(RUNTIME))

from cg_contracts import enrich_query_contracts
from cg_goals import extract_goals
from cg_instantiate import template_instances
from cg_render import (
    build_causground_record_from_response,
    render_causground_text,
)
from cg_routes import build_response as build_graph_response


GRAPH_PATH = ROOT / "causground.json"
TASKS_PATH = ROOT / "tasks.json"
EMBEDDING_MODEL = os.environ.get(
    "CAUSGROUND_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"
)


def load_graph():
    return json.loads(GRAPH_PATH.read_text(encoding="utf-8"))


def load_tasks():
    return json.loads(TASKS_PATH.read_text(encoding="utf-8"))


def build_pool(graph, description):
    pool = []
    seen = set()
    for mechanism in graph["specific_mechanisms"] + graph.get("role_templates", []):
        if mechanism["id"] not in seen:
            pool.append(deepcopy(mechanism))
            seen.add(mechanism["id"])
    instances, _ = template_instances(graph, description)
    for mechanism in instances:
        if mechanism["id"] not in seen:
            pool.append(mechanism)
            seen.add(mechanism["id"])
    pool, _ = enrich_query_contracts(pool, description, graph)
    return pool


def extract_direct(description):
    """Build a graph from CausGround contracts, without task lookup/model."""
    graph = load_graph()
    pool = build_pool(graph, description)
    result = build_graph_response(
        description,
        pool,
        {},
        [],
        [],
        mode="goal",
        max_nodes=32,
    )
    result["causground_source"] = "bundled causground.json"
    return result


def _mechanism_text(mechanism):
    bindings = " ".join(str(value) for value in mechanism.get("bindings", {}).values())
    return " ".join(
        [
            str(mechanism.get("action", "")),
            str(mechanism.get("api_call", "")),
            str(mechanism.get("description", "")),
            bindings,
        ]
    )


def _embedding_candidates(description, pool, model_name):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "embedding fallback requires sentence-transformers; install it with "
            "pip install sentence-transformers, or pass a local model path via "
            "CAUSGROUND_EMBEDDING_MODEL"
        ) from exc
    model = SentenceTransformer(model_name)
    texts = [_mechanism_text(mechanism) for mechanism in pool]
    vectors = model.encode([description, *texts], normalize_embeddings=True)
    scores = vectors[1:] @ vectors[0]
    score_map = {mechanism["id"]: float(score) for mechanism, score in zip(pool, scores)}
    selected = [
        mechanism["id"]
        for mechanism, _ in sorted(
            zip(pool, scores), key=lambda item: float(item[1]), reverse=True
        )[:18]
    ]
    return score_map, selected


def extract_with_embedding(description, model_name=EMBEDDING_MODEL):
    """Use the selected embedding model to supplement CausGround selection."""
    graph = load_graph()
    pool = build_pool(graph, description)
    scores, selected = _embedding_candidates(description, pool, model_name)
    result = build_graph_response(
        description,
        pool,
        scores,
        selected,
        [],
        mode="hybrid",
        max_nodes=32,
    )
    result["causground_source"] = "bundled causground.json + embedding mechanism retrieval"
    result["embedding_model"] = model_name
    if not result["graph"].get("nodes"):
        # If the text has no parseable final predicate, expose the semantic
        # CausGround candidates explicitly rather than returning an empty graph.
        semantic = result.get("graphs", {}).get("semantic")
        if semantic and semantic.get("nodes"):
            result["graphs"]["embedding"] = semantic
            result["graph"] = semantic
            result["mode"] = "embedding"
    return result


def _normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def choose_task_id(task_id):
    tasks = load_tasks()
    wanted = task_id.strip().lower()
    for item in tasks:
        if item["id"].lower() == wanted:
            return item
    if wanted.isdigit():
        index = int(wanted) - 1
        if 0 <= index < len(tasks):
            return tasks[index]
    raise ValueError(f"unknown task id: {task_id}")


def choose_description(value):
    """Accept an exact preset id/description, otherwise keep it as new text."""
    for item in load_tasks():
        if _normalize(value) == _normalize(item["id"]):
            return item["description"], item["id"]
        if _normalize(value) == _normalize(item["description"]):
            return item["description"], item["id"]
    return value, None


def render_result(description, task_id=None, use_embedding=False, model_name=EMBEDDING_MODEL):
    response = (
        extract_with_embedding(description, model_name)
        if use_embedding
        else extract_direct(description)
    )
    route = response.get("mode", "goal")
    if route == "goal":
        graph_route = "goal"
        public_route = "causground"
    elif route == "embedding":
        graph_route = "embedding"
        public_route = "causground+embedding"
    else:
        graph_route = "hybrid"
        public_route = "causground+embedding"
    record = build_causground_record_from_response(response, route=graph_route)
    record["route"] = public_route
    record["task_id"] = task_id
    if not record["actions"]:
        raise RuntimeError(
            "CausGround did not extract an action graph for this description. "
            "Use --embedding to enable embedding-based mechanism retrieval."
        )
    text = render_causground_text(record)
    if task_id:
        text = f"task_id: {task_id}\n" + text
    return text, record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run RoboCasa CausGround offline")
    parser.add_argument("task", nargs="?", help="a new task description, or a preset task id")
    parser.add_argument("--task-id", help="run one bundled task by name or 1-based number")
    parser.add_argument("--list", action="store_true", help="list the 32 bundled task ids")
    parser.add_argument("--embedding", action="store_true", help="enable embedding mechanism retrieval")
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="local path or model name for --embedding")
    parser.add_argument("--json", action="store_true", help="output the structured record instead of text")
    args = parser.parse_args(argv)

    if args.list:
        tasks = load_tasks()
        print(f"CausGround task catalog: {len(tasks)} tasks")
        for index, item in enumerate(tasks, 1):
            print(f"{index:02d}. {item['id']} - {item['short']}")
        return 0

    task_id = None
    if args.task_id:
        item = choose_task_id(args.task_id)
        description, task_id = item["description"], item["id"]
    elif args.task:
        description, task_id = choose_description(args.task)
    else:
        parser.error("provide a task description, --task-id, or --list")

    try:
        text, record = render_result(
            description,
            task_id=task_id,
            use_embedding=args.embedding,
            model_name=args.model,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"CausGround error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(record, ensure_ascii=False, indent=2))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
