# CausGround-Robocasa Offline Minimal Package

When running `causground.py`, it directly reads the bundled `causground.json` and re-parses the input description using the CausGround mechanism library, generating actions, conditions, effects, producer-backed dependencies, ordering, and uncertainty.

The graph is derived from 32 long-horizon tasks in RoboCasa. RoboCasa contains more than 50 tasks in total, of which 32 are multi-step tasks consisting of multiple actions, such as picking, moving, and placing. The remaining tasks are atomic-level tasks, where a single action, such as picking or placing, constitutes an entire task. During extraction, I temporarily excluded these standalone atomic tasks because their coverage is relatively narrow, and they are already covered by the long-horizon (multi-step) tasks.

The input can be one of the 32 original RoboCasa tasks (relevant examples are provided in `tasks.json`) or any other task (e.g., `Pick up the bread and put it into the plate.`). The program still generates `PICK(bread) -> PLACE(bread,plate,IN)` according to the mechanism contracts and lists conditions such as `agent NEAR bread`, `agent.gripper_empty = true`, and `agent HOLDS bread`.

The supported scope is determined by the action and object-type contracts in the bundled CausGround mechanism library. Elements outside this scope will retain an `unresolved` marker.

## Minimal Run (Recommended)

```bash
python causground.py "Open the cabinet, pick up the bread and put it in the basket."
```

The output is a textual CausGround representation that can be directly passed to a VLM/LLM.

To output structured JSON:

```bash
python causground.py --json "Open the cabinet, pick up the bread and put it in the basket."
```

To view a short list of the 32 RoboCasa tasks:

```bash
python causground.py --list
```

You can also run a bundled task by its ID or name:

```bash
python causground.py --task-id ArrangeBreadBasket
python causground.py --task-id 1
```

`tasks.json` contains the full English descriptions and brief annotations of the 32 RoboCasa tasks. It serves only as a task catalog and a set of regression examples. When a new task is provided, the program does not match the input to an existing task. Instead, it re-extracts the corresponding information from the mechanism library in `causground.json`.

## Semantic Supplementation Path

If a new description does not contain a sufficiently clear and parseable target (e.g., RoboCasa tasks generally use explicit actions such as `pick up xxx` and `put xx`), then for tasks where the actions are not clearly specified, Goal Interpretation is needed to infer a goal state as much as possible, and then search backward from that goal state.

Alternatively, mechanism-level embedding retrieval can be explicitly enabled. CausGround retrieves subgraphs that are semantically relevant to the current task and uses them for further processing. The currently selected model is `Qwen3-Embedding-0.6B`.

```bash
pip install sentence-transformers
python causground.py --embedding \
  --model /share/project/zhiwei/syj/models/Qwen3-Embedding-0.6B \
  "Prepare an unfamiliar kitchen task."
```

The default model name is `Qwen/Qwen3-Embedding-0.6B`. You can also point `--model` to a local model directory or set `CAUSGROUND_EMBEDDING_MODEL`. This pathway retrieves candidates from the bundled CausGround mechanism library, and the output route will be marked as `causground+embedding`.

## Package Contents

- `causground.py`: The only entry point that needs to be executed directly.
- `causground.json`: The complete CausGround mechanism library.
- `tasks.json`: Full descriptions and brief annotations of the 32 tasks.
- `runtime/`: Minimal parsing, instantiation, dependency construction, and text rendering code used by the entry point.

## TL;DR

Run directly: `python causground.py "Your task description"`. The program generates a causal subgraph directly from the bundled CausGround mechanism library. For semantic supplementation, use `--embedding` and provide an external Qwen3 model directory.