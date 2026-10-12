"""Stable text/record adapter for the RoboCasa CausGround dependency interface.

The graph service is intentionally kept separate from this module.  A caller
can therefore render a saved ``/v1/graph`` response without loading the
embedding model, while a live caller can use :func:`task_to_causground_text` with an
injected ``CausGroundGraphService`` instance.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


SCHEMA_VERSION = "robocasa-causground-text-v1"


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _predicate_key(predicate: Mapping[str, Any]) -> tuple[Any, ...]:
    kind = predicate.get("type")
    if kind == "edge":
        return (
            "edge",
            predicate.get("source"),
            predicate.get("relation"),
            predicate.get("target"),
            bool(predicate.get("present", True)),
        )
    if kind in {"node_attribute", "attribute"}:
        return (
            "attribute",
            predicate.get("node", predicate.get("object")),
            predicate.get("attribute"),
            _json_key(predicate.get("value")),
        )
    return (kind, _json_key(predicate))


def predicate_text(predicate: Mapping[str, Any] | None) -> str:
    """Return a compact, human-readable state predicate.

    This format is deliberately close to the graph vocabulary used by the
    action contracts, so a downstream VLM can read it without another schema
    translation step.
    """

    if not isinstance(predicate, Mapping):
        return "<unknown predicate>"
    kind = predicate.get("type")
    if kind == "edge":
        value = "{} {} {}".format(
            predicate.get("source", "?"),
            predicate.get("relation", "?"),
            predicate.get("target", "?"),
        )
        return value if predicate.get("present", True) else f"NOT {value}"
    if kind in {"node_attribute", "attribute"}:
        subject = predicate.get("node", predicate.get("object", "?"))
        return f"{subject}.{predicate.get('attribute', '?')} = {_scalar_text(predicate.get('value'))}"
    if kind == "requested_order":
        return "phase {} -> phase {}".format(
            predicate.get("before_phase", "?"), predicate.get("after_phase", "?")
        )
    return _json_key(predicate)


def _scalar_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return _json_key(value)


def _edge_text(edge: Mapping[str, Any]) -> str:
    predicate = edge.get("predicate")
    if isinstance(predicate, Mapping):
        return predicate_text(predicate)
    return "<missing predicate>"


def _copy_list(value: Any) -> list[Any]:
    return deepcopy(value) if isinstance(value, list) else []


def _node_sort_key(node: Mapping[str, Any]) -> tuple[Any, str, str]:
    phase = node.get("phase")
    numeric_phase = phase if isinstance(phase, (int, float)) else 10**9
    return numeric_phase, str(node.get("api_call", "")), str(node.get("id", ""))


def _select_graph(response: Mapping[str, Any], route: str) -> Mapping[str, Any]:
    if not isinstance(response, Mapping):
        raise TypeError("response must be a mapping")
    if not isinstance(route, str) or not route.strip():
        raise ValueError("route must be a nonempty string")
    graphs = response.get("graphs")
    if isinstance(graphs, Mapping) and isinstance(graphs.get(route), Mapping):
        return graphs[route]
    graph = response.get("graph")
    if isinstance(graph, Mapping):
        graph_route = graph.get("route")
        if graph_route in {None, route} or not graphs:
            return graph
    raise ValueError(f"route {route!r} is not present in the graph response")


def _action_record(node: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the action contract fields while avoiding graph-only noise."""

    result: dict[str, Any] = {
        "id": node.get("id"),
        "action": node.get("action"),
        "api_call": node.get("api_call"),
        "phase": node.get("phase"),
        "bindings": deepcopy(node.get("bindings", {})),
        "conditions": _copy_list(node.get("conditions")),
        "effects": _copy_list(node.get("effects")),
        "sources": _copy_list(node.get("sources")),
        "evidence": deepcopy(node.get("evidence", {})),
        "execution_verified": bool(node.get("execution_verified", False)),
    }
    for key in (
        "mechanism_ref",
        "role_types",
        "role_type_constraints",
        "parent_template_id",
        "parent_generic_id",
        "instantiation_basis",
        "query_context_requirements",
        "new_type_outcome_observed",
        "selection_basis",
    ):
        if key in node:
            result[key] = deepcopy(node[key])
    return result


def _merge_dependency_edges(
    graph: Mapping[str, Any], nodes_by_id: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify and merge graph edges without losing their provenance."""

    causal: MutableMapping[tuple[Any, ...], dict[str, Any]] = {}
    textual: MutableMapping[tuple[Any, ...], dict[str, Any]] = {}
    for edge in graph.get("edges", []) if isinstance(graph.get("edges"), list) else []:
        if not isinstance(edge, Mapping):
            continue
        source_id, target_id = edge.get("source"), edge.get("target")
        source = nodes_by_id.get(source_id, {})
        target = nodes_by_id.get(target_id, {})
        source_action = source.get("api_call", source_id)
        target_action = target.get("api_call", target_id)
        predicate = deepcopy(edge.get("predicate", {}))
        base = str(edge.get("basis", "unknown"))
        if base == "declared_effect_condition":
            key = (source_id, target_id, _predicate_key(predicate))
            record = causal.get(key)
            if record is None:
                record = {
                    "producer_id": source_id,
                    "producer_action": source_action,
                    "consumer_id": target_id,
                    "consumer_action": target_action,
                    "predicate": predicate,
                    "produced_state": predicate_text(predicate),
                    "condition_bases": [],
                    "edge_bases": [],
                    "edge_count": 0,
                    "depends_on_initial_state": False,
                    "causal_necessity_verified": False,
                }
                causal[key] = record
            record["edge_count"] += 1
            if base not in record["edge_bases"]:
                record["edge_bases"].append(base)
            condition_basis = edge.get("condition_basis")
            if condition_basis and condition_basis not in record["condition_bases"]:
                record["condition_bases"].append(condition_basis)
            record["depends_on_initial_state"] = bool(
                record["depends_on_initial_state"] or edge.get("depends_on_initial_state", False)
            )
            record["causal_necessity_verified"] = bool(
                record["causal_necessity_verified"] or edge.get("causal_necessity_verified", False)
            )
        else:
            key = (source_id, target_id, _predicate_key(predicate), base)
            record = textual.get(key)
            if record is None:
                record = {
                    "before_id": source_id,
                    "before_action": source_action,
                    "after_id": target_id,
                    "after_action": target_action,
                    "predicate": predicate,
                    "constraint": _edge_text(edge),
                    "bases": [],
                    "edge_count": 0,
                    "causal_necessity_verified": False,
                }
                textual[key] = record
            record["edge_count"] += 1
            if base not in record["bases"]:
                record["bases"].append(base)
    return (
        sorted(causal.values(), key=lambda x: (str(x["producer_action"]), str(x["consumer_action"]), x["produced_state"])),
        sorted(textual.values(), key=lambda x: (str(x["before_action"]), str(x["after_action"]), x["constraint"])),
    )


def _initial_requirements(
    graph: Mapping[str, Any], producer_dependencies: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    requirements: MutableMapping[tuple[Any, ...], dict[str, Any]] = {}

    def add(predicate: Any, status: str, basis: str, **extra: Any) -> None:
        if not isinstance(predicate, Mapping):
            return
        key = (_predicate_key(predicate), status)
        if key not in requirements:
            requirements[key] = {
                "predicate": deepcopy(predicate),
                "state": predicate_text(predicate),
                "status": status,
                "basis": basis,
            }
        for name, value in extra.items():
            if value is not None and name not in requirements[key]:
                requirements[key][name] = deepcopy(value)

    for item in graph.get("initial_satisfied", []) if isinstance(graph.get("initial_satisfied"), list) else []:
        if isinstance(item, Mapping):
            add(item.get("predicate"), "provided", str(item.get("basis", "provided_initial_state")), consumer=item.get("consumer"))
    for item in graph.get("unresolved_conditions", []) if isinstance(graph.get("unresolved_conditions"), list) else []:
        if isinstance(item, Mapping):
            reason = str(item.get("reason", ""))
            if "initial" in reason or "unspecified" in reason or "state" in reason:
                add(item.get("predicate"), "unknown", reason or "unresolved_condition", consumer=item.get("consumer"))
    return sorted(requirements.values(), key=lambda x: (x["status"], x["state"], str(x.get("consumer", ""))))


def build_causground_record_from_response(response: Mapping[str, Any], route: str = "hybrid") -> dict[str, Any]:
    """Convert a graph-service response into the stable text-interface record."""

    graph = _select_graph(response, route)
    nodes = [node for node in graph.get("nodes", []) if isinstance(node, Mapping)]
    nodes = sorted(nodes, key=_node_sort_key)
    nodes_by_id = {str(node.get("id")): node for node in nodes}
    actions = [_action_record(node) for node in nodes]
    producer_dependencies, text_order_constraints = _merge_dependency_edges(graph, nodes_by_id)
    initial_requirements = _initial_requirements(graph, producer_dependencies)

    uncertainty_keys = (
        "unresolved_conditions",
        "unresolved_goals",
        "alternatives",
        "conditional_bindings",
        "state_threats",
        "resource_conflicts",
    )
    uncertainties = {
        key: _copy_list(graph.get(key))
        for key in uncertainty_keys
    }
    uncertainties["order_cycles"] = _copy_list((graph.get("order") or {}).get("cycles"))
    uncertainties["support_status"] = graph.get("support_status", "unknown")
    uncertainties["execution_verified"] = False

    goals = _copy_list(graph.get("goal_states"))
    final_goals = _copy_list(graph.get("final_goal_states"))
    return {
        "schema_version": SCHEMA_VERSION,
        "input": response.get("input", ""),
        "route": route,
        "support_status": graph.get("support_status", "unknown"),
        "execution_verified": False,
        "ordering_policy": response.get("ordering_policy", ""),
        "actions": actions,
        "goal_states": goals,
        "final_goal_states": final_goals,
        "producer_backed_dependencies": producer_dependencies,
        "text_order_constraints": text_order_constraints,
        "initial_state_requirements": initial_requirements,
        "uncertainties": uncertainties,
        "metadata": {
            "source_schema_version": response.get("schema_version"),
            "graph_route": graph.get("route"),
            "graph_sha256": (response.get("metadata") or {}).get("graph_sha256"),
        },
    }


def _format_bindings(bindings: Mapping[str, Any]) -> str:
    if not bindings:
        return "{}"
    return ", ".join(f"{key}={_scalar_text(bindings[key])}" for key in sorted(bindings))


def _condition_text(item: Any) -> str:
    if not isinstance(item, Mapping):
        return _json_key(item)
    text = predicate_text(item.get("predicate"))
    basis = item.get("basis")
    return f"{text} [basis={basis}]" if basis else text


def _effect_text(item: Any) -> str:
    if not isinstance(item, Mapping):
        return _json_key(item)
    operation = item.get("operation", "effect")
    if operation in {"add_edge", "delete_edge"}:
        predicate = deepcopy(item.get("edge", {}))
        predicate["type"] = "edge"
        predicate["present"] = operation == "add_edge"
        text = predicate_text(predicate)
    elif operation == "set_attribute":
        text = predicate_text({
            "type": "node_attribute",
            "node": item.get("node", "?"),
            "attribute": item.get("attribute", "?"),
            "value": item.get("after"),
        })
    else:
        text = _json_key(item)
    basis = item.get("basis")
    return f"{text} [operation={operation}{', basis=' + str(basis) if basis else ''}]"


def _section_lines(items: Iterable[str]) -> list[str]:
    values = [item for item in items if item]
    return [f"- {item}" for item in values] if values else ["- none"]


def render_causground_text(record: Mapping[str, Any]) -> str:
    """Render a record into deterministic plain text suitable for a VLM prompt."""

    if not isinstance(record, Mapping) or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"record must use schema_version={SCHEMA_VERSION!r}")
    lines = [
        "RoboCasa CausGround",
        f"task: {record.get('input', '')}",
        f"route: {record.get('route', 'unknown')}",
        f"support_status: {record.get('support_status', 'unknown')}",
        "execution_verified: false",
        f"ordering_policy: {record.get('ordering_policy') or 'unspecified'}",
        "",
        "goal_states:",
    ]
    goals = record.get("final_goal_states") or record.get("goal_states") or []
    lines.extend(_section_lines(predicate_text(item.get("predicate")) if isinstance(item, Mapping) else _json_key(item) for item in goals))

    lines.extend(["", "actions:"])
    actions = record.get("actions") if isinstance(record.get("actions"), list) else []
    for action in actions:
        phase = action.get("phase", "?")
        lines.append(f"- [phase {phase}] {action.get('api_call', '<unknown action>')}")
        lines.append(f"  id: {action.get('id', '<unknown>')}")
        lines.append(f"  bindings: {_format_bindings(action.get('bindings') or {})}")
        lines.append("  requires:")
        lines.extend(f"    {line}" for line in _section_lines(_condition_text(item) for item in action.get("conditions", [])))
        lines.append("  effects:")
        lines.extend(f"    {line}" for line in _section_lines(_effect_text(item) for item in action.get("effects", [])))
        if action.get("role_type_constraints"):
            lines.append(f"  conditional_bindings: {_json_key(action['role_type_constraints'])}")

    lines.extend(["", "producer_backed_dependencies:"])
    dependencies = record.get("producer_backed_dependencies") if isinstance(record.get("producer_backed_dependencies"), list) else []
    if dependencies:
        for dep in dependencies:
            suffix = []
            if dep.get("condition_bases"):
                suffix.append("basis=" + ",".join(map(str, dep["condition_bases"])))
            lines.append(
                "- {} --produces--> {} --required_by--> {}{}".format(
                    dep.get("producer_action", "<unknown producer>"),
                    dep.get("produced_state", "<unknown state>"),
                    dep.get("consumer_action", "<unknown consumer>"),
                    f" [{'; '.join(suffix)}]" if suffix else "",
                )
            )
    else:
        lines.append("- none")

    lines.extend(["", "text_order_constraints:"])
    constraints = record.get("text_order_constraints") if isinstance(record.get("text_order_constraints"), list) else []
    if constraints:
        for item in constraints:
            bases = ",".join(map(str, item.get("bases", []))) or "unknown"
            lines.append(f"- {item.get('before_action', '<unknown>')} -> {item.get('after_action', '<unknown>')} [basis={bases}; constraint={item.get('constraint', '<unknown>')}]")
    else:
        lines.append("- none")

    lines.extend(["", "initial_state_requirements:"])
    initial = record.get("initial_state_requirements") if isinstance(record.get("initial_state_requirements"), list) else []
    lines.extend(_section_lines(
        f"{item.get('state', '<unknown>')} [status={item.get('status', 'unknown')}; basis={item.get('basis', 'unknown')}]"
        for item in initial
    ))

    lines.extend(["", "uncertainties:"])
    uncertainties = record.get("uncertainties") if isinstance(record.get("uncertainties"), Mapping) else {}
    uncertainty_order = (
        "support_status",
        "unresolved_conditions",
        "unresolved_goals",
        "alternatives",
        "conditional_bindings",
        "state_threats",
        "resource_conflicts",
        "order_cycles",
    )
    any_uncertainty = False
    for key in uncertainty_order:
        value = uncertainties.get(key)
        if key == "support_status":
            lines.append(f"- support_status={value or 'unknown'}")
            continue
        if not value:
            continue
        any_uncertainty = True
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping) and "predicate" in item:
                    rendered = dict(item)
                    rendered["predicate"] = predicate_text(item["predicate"])
                    lines.append(f"- {key}: {_json_key(rendered)}")
                else:
                    lines.append(f"- {key}: {_json_key(item)}")
        else:
            lines.append(f"- {key}: {_json_key(value)}")
    if not any_uncertainty:
        lines.append("- none beyond unverified execution")
    return "\n".join(lines)


def response_to_causground_text(
    response: Mapping[str, Any], route: str = "hybrid", *, return_record: bool = False
) -> str | tuple[str, dict[str, Any]]:
    """Render one existing graph response, optionally returning its record."""

    record = build_causground_record_from_response(response, route=route)
    text = render_causground_text(record)
    return (text, record) if return_record else text


def task_to_causground_text(
    task_description: str,
    service: Any,
    *,
    route: str = "hybrid",
    initial_state: Sequence[Mapping[str, Any]] | None = None,
    max_nodes: int = 32,
    return_record: bool = False,
) -> str | tuple[str, dict[str, Any]]:
    """Query an injected graph service and return the text CausGround.

    ``service`` is injected on purpose: importing this module never loads the
    embedding model, and tests can use a saved graph response or a small fake
    service.  The graph service's public mode names are the same as route
    names, with ``hybrid`` as the default used for downstream planning.
    """

    if not isinstance(task_description, str) or not task_description.strip():
        raise ValueError("task_description must be a nonempty string")
    if service is None or not hasattr(service, "query"):
        raise TypeError("service must provide query(input, mode, initial_state, max_nodes)")
    response = service.query(task_description, mode=route, initial_state=initial_state, max_nodes=max_nodes)
    return response_to_causground_text(response, route=route, return_record=return_record)


__all__ = [
    "SCHEMA_VERSION",
    "build_causground_record_from_response",
    "predicate_text",
    "render_causground_text",
    "response_to_causground_text",
    "task_to_causground_text",
]
