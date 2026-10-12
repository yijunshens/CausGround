[English](README.md)

# CausGround-Robocasa 离线最小包

运行 `causground.py` 时，它直接读取包内的 `causground.json`，从 CausGround 机制库重新解析输入描述，生成动作、条件、效果、producer-backed dependencies、顺序和不确定性。

这里的图来源于Robocasa的32个长序列tasks，他一共有50+，其中有32个是多个actions的那种多步任务，比如抓，移动，放；剩下的那些是原子级别任务，比如只有抓，只有放构成一个任务，我提取的时候暂时排除了单独的原子任务，因为覆盖范围比较窄，而且其实已经被长序列（多步）任务都覆盖了。

输入可以是RoboCasa 的 32 个原生 task （已给了相关的示例，在tasks.json中）
 或已有任意的 task （e.g., `Pick up the bread and put it into the plate.`）；程序仍会按机制合同生成 `PICK(bread) -> PLACE(bread,plate,IN)`，并列出 `agent NEAR bread`、`agent.gripper_empty = true`、`agent HOLDS bread` 等条件。
支持范围由包内 CausGround 的动作和对象类型合同决定，超出范围时会保留 unresolved 标记。

## 最小运行(推荐)

```bash
python causground.py "Open the cabinet, pick up the bread and put it in the basket."
```

输出是可以直接传给 VLM/LLM 的文本 CausGround。

输出结构化 JSON：

```bash
python causground.py --json "Open the cabinet, pick up the bread and put it in the basket."
```

查看 32 个 RoboCasa task 的短列表：

```bash
python causground.py --list
```

也可以按编号或名称运行包内任务：

```bash
python causground.py --task-id ArrangeBreadBasket
python causground.py --task-id 1
```

`tasks.json` 保存了 Robocasa的 32 个 task 的完整英文描述和短注释。它只是任务目录与回归样例；输入新的 task 时，程序不会把输入匹配成某个旧 task，而是从 `causground.json` 的机制库重新抽取。


## 语义补充路径

如果新描述没有足够明确的可解析目标（e.g., robocasa的task一般会是pick up xxx, put xx，动作比较明确），但要是针对一些动作不是非常明确的，需要进行Goal interpreation，尽可能推理出一个goal state，然后再回过来搜索。
当然也可以显式开启机制级 embedding 检索。CausGround去检索出与当前任务比较相关的subgraph，然后拿出来用，当前选择 `Qwen3-Embedding-0.6B`。

```bash
pip install sentence-transformers
python causground.py --embedding \
  --model /share/project/zhiwei/syj/models/Qwen3-Embedding-0.6B \
  "Prepare an unfamiliar kitchen task."
```

默认模型名是 `Qwen/Qwen3-Embedding-0.6B`。也可以把 `--model` 指向本地模型目录，或设置 `CAUSGROUND_EMBEDDING_MODEL`。这条路径从包内 CausGround 机制库取候选，输出 route 会标记为 `causground+embedding`。

## 包内容

- `causground.py`：唯一需要直接运行的入口。
- `causground.json`：完整 CausGround 机制库。
- `tasks.json`：32 个 task 的完整描述和短注释。
- `runtime/`：入口使用的最小解析、实例化、依赖和文本渲染代码。

## 省流版

直接运行：`python causground.py "你的 task 描述"`。直接从包内 CausGround 机制库生成causal subgraph；需要语义补充时用 `--embedding` 并提供外部 Qwen3 模型目录。
