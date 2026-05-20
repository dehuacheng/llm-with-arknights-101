# llm-with-arknights-101

**[English](README.md) | 简体中文**

> 一个端到端的教学项目：以《明日方舟》的世界观文本为语料，完整演示如何对一个
> 参数量低于 10 亿的语言模型做继续预训练（continued-pretraining）与后训练
> （post-training）。

本仓库通过实例讲解小型 LLM 的完整训练流程。"学生"模型为 **Qwen3-0.6B**；领域
语料取自游戏《明日方舟》的世界观设定 —— 之所以选它，是因为其文本丰富、天然多语言
（中/英/日），且有明确的"标准答案"可供评测。

这是一个学习项目 —— 每个阶段都设计为可阅读、可运行、可修改。

> [!IMPORTANT]
> **与鹰角网络（Hypergryph）无任何关联。**《明日方舟》及其全部文本、角色均归
> 鹰角网络所有。本项目为非商业、仅供教学用途，**不**分发任何游戏原始文本 ——
> 详见[数据与知识产权](#数据与知识产权)。

## 路线图

项目拆分为若干相互独立的阶段，每个阶段产出下一个阶段所需的 checkpoint 或产物。

| 阶段 | 状态 |
|---|---|
| 环境搭建与选型 —— Qwen3-0.6B base，`transformers + peft + trl` 技术栈 | 已决定 |
| **原始数据生成** —— 构建世界观语料 | ✅ 本次提交 |
| 继续预训练（CPT）—— 全参微调 vs LoRA，含回放混合语料 | 计划中 |
| SFT 蒸馏 —— 从教师 LLM 蒸馏出闭卷问答 | 计划中 |
| DPO —— 用"看似合理的幻觉"构造偏好对 | 计划中 |
| 拒答训练 —— 对设定外问题回答"我不知道" | 计划中 |
| 思维链蒸馏 —— 可选的 `<think>` 推理轨迹 | 计划中 |
| 评测与发布 —— 约 200 题的人工评分测试集 | 计划中 |

评测集在训练开始**之前**就编写完成，以便每个阶段都能对同一套题重新评分。各项
决策的详细理由见 `plan.md`（规划工作区，未提交至本仓库）。

本次提交仅包含**原始数据生成**阶段。后续各阶段将分别落到各自的顶层目录
（`tokenizer/`、`pretrain/` 等）。

## 仓库结构

```
llm-with-arknights-101/
├── README.md             # 英文说明
├── README.zh-CN.md       # 简体中文版（本文件）
├── AGENTS.md             # 给 AI 编码 agent 的长期决策记录
├── CLAUDE.md             # 为 Claude Code 导入 AGENTS.md
├── requirements.txt      # Python 依赖（目前极简）
├── tools/
│   └── build_raw_data.py # 重新生成语料
└── data/                 # 生成产物；已 git-ignore；永不提交
    └── raw/
        ├── cn/stories/   # 中文活动剧情脚本（标准答案）
        ├── cn/operators/ # 中文干员档案（标准答案）
        ├── wiki/         # LLM 总结的 wiki（仅供评测/测试）
        └── MANIFEST.md   # 溯源信息：来源仓库 + 提交哈希
```

## 外部数据依赖

本仓库存放的是*代码*，不是数据。语料由三个上游仓库重新构建，这三个仓库必须与本
仓库克隆在**同一级目录下**（同一父目录）：

| 仓库 | 角色 | 在 `build_raw_data.py` 中的用途 |
|------|------|----|
| [`Kengxxiao/ArknightsGameData`](https://github.com/Kengxxiao/ArknightsGameData) | 中文游戏数据原始转储 —— **标准答案** | 输入 → `data/raw/cn/` |
| [`littlepangding/arknights_lore_wiki`](https://github.com/littlepangding/arknights_lore_wiki) | LLM 总结的世界观 wiki —— *非*标准答案 | 输入 → `data/raw/wiki/` |
| [`littlepangding/arknights_lore_wiki_lib`](https://github.com/littlepangding/arknights_lore_wiki_lib) | 游戏数据转储的确定性解析器 | 仅作构建期解析器 |

> [!NOTE]
> **依赖边界。** `arknights_lore_wiki_lib` **仅**由 `tools/build_raw_data.py`
> 使用，且纯粹作为构建期的解析器。项目中其他任何代码都不导入它，也不会将其
> vendor 进来或通过 pip 安装。项目长期依赖的是两个*数据*仓库。

### 克隆依赖仓库

```bash
# 在包含本仓库的父目录下执行
git clone --depth 1 https://github.com/Kengxxiao/ArknightsGameData.git
git clone https://github.com/littlepangding/arknights_lore_wiki.git
git clone https://github.com/littlepangding/arknights_lore_wiki_lib.git
```

最终目录结构：

```
projects/
├── llm-with-arknights-101/   # 本仓库
├── ArknightsGameData/
├── arknights_lore_wiki/
└── arknights_lore_wiki_lib/
```

### 锁定的提交（用于复现逐字节一致的输出）

`build_raw_data.py` 在*来源仓库处于固定提交*的前提下是确定性的。本仓库当前所对应
的语料是基于以下提交构建的：

| 仓库 | 提交 |
|------|--------|
| ArknightsGameData | `de85be497656612992f25092d3df2045476d6dc1` |
| arknights_lore_wiki | `9c2f8cb64f2453802989808890bb031333b17cca` |
| arknights_lore_wiki_lib | `fbf68f3eaf14ee7032b8b9941280c9f4f9b6a777` |

若要精确复现，请对每个仓库 `git checkout` 到对应提交（需要完整克隆 —— `--depth 1`
浅克隆只会拉取当前 `HEAD`）。每次运行还会把实际使用的提交写入
`data/raw/MANIFEST.md`。

## 环境搭建

**原始数据生成几乎不需要任何依赖。** `tools/build_raw_data.py` 只使用 Python
标准库。它唯一的要求是解析器仓库的虚拟环境，该环境提供解析所需的依赖
（`pypinyin` 等）。

```bash
# 1. Python 3.10+
python3 --version

# 2. 创建解析器 venv，一次即可（在本仓库根目录下执行）
python3 -m venv ../arknights_lore_wiki_lib/.venv
../arknights_lore_wiki_lib/.venv/bin/pip install -r ../arknights_lore_wiki_lib/requirements.txt
```

`build_raw_data.py` 会自动在该 venv 下重新执行（re-exec）自身 —— 你可以用任意
Python 3 启动它。

本仓库的 `requirements.txt` 目前刻意保持极简；后续阶段（CPT、SFT 等）会在各自落地
时锁定自己的训练依赖（`torch`、`transformers`、`peft`、`trl` 等）。

## 重新生成训练数据

在仓库根目录下执行：

```bash
python3 tools/build_raw_data.py
```

它会写入以下内容（全部位于已 git-ignore 的 `data/` 下）：

| 产物 | 内容 | 数量 |
|--------|----------|-------|
| `data/raw/cn/stories/` | 中文活动剧情脚本（对白 + 旁白） | 461 |
| `data/raw/cn/operators/` | 中文干员档案 | 444 |
| `data/raw/wiki/` | LLM 总结的 wiki 页面（仅供评测/测试） | 3000 个文件 |
| `data/raw/MANIFEST.md` | 溯源信息 —— 来源仓库 + 提交哈希 | — |

`cn/` 是用于训练的**标准答案**语料；`wiki/` 是*模型生成*的总结，只应用于评测/测试，
绝不可作为正式的训练目标。

## 数据与知识产权

- 整个 `data/` 目录**已 git-ignore 且永不提交** —— 它体积庞大，且衍生自鹰角网络
  的游戏知识产权。
- 本仓库随附**重新生成脚本 + 来源提交哈希**，使语料在不被分发的前提下仍可复现。
- 《明日方舟》及其文本与角色版权归鹰角网络（Hypergryph）所有。本项目与鹰角网络
  **无任何关联**，为**非商业**项目，**仅供教学用途**。

## 许可证

- 代码：**Apache-2.0**（计划中）。
- 后续落地的人工评分评测集：**CC-BY-4.0**。
- 衍生自游戏的文本**不**由本项目授权，且绝不在此分发。
