# WXdown-RAG

<div align="center">

**基于微信公众号知识库语料的 RAG（Retrieval-Augmented Generation）检索问答系统**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)](LICENSE)

</div>

---

## 📋 目录

- [功能特性](#-功能特性)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [配置说明](#-配置说明)
- [使用指南](#-使用指南)
- [技术架构](#-技术架构)
- [开发与测试](#-开发与测试)
- [许可证](#-许可证)

---

## ✨ 功能特性

### 核心功能

- **📚 本地知识库构建**
  - 支持微信公众号导出的知识库文档加载、切分、向量化与索引构建（搭配WXdown-Web-server项目）
  - 支持多种文档切分策略：固定大小、句子、语义、层次化、滑动窗口
  - 增量更新机制，通过 `manifest.json` 记录变更，仅处理新增或修改文件

- **🔍 混合检索系统**
  - **向量检索**：基于 FAISS 的语义相似度检索
  - **BM25 检索**：基于关键词匹配的稀疏检索
  - **混合检索**：使用 RRF（Reciprocal Rank Fusion）融合两种检索结果
  - **重排序（Rerank）**：使用 Cross-Encoder 模型对检索结果进行精细重排序

- **💬 标准化问答接口**
  - 提供统一的 CLI 接口：`ingest` / `query` / `embedded-pdfs`
  - 支持 JSON 输出格式，便于集成到上层服务
  - 完整的文档溯源能力，输出命中 PDF、页码、相似度等信息

- **🛠️ 检索调试能力**
  - 支持输出检索中间状态，便于调参与排错
  - 可预览命中文档片段，帮助理解检索效果

### 技术亮点

- **多种切分策略**：支持 5 种文档切分策略，适应不同场景需求
- **语义切分**：基于 BGE 模型的智能语义切分，保持语义完整性
- **混合检索**：结合关键词匹配与语义检索的优势
- **重排序优化**：使用 Cross-Encoder 模型提升检索精度
- **增量更新**：智能追踪文档变更，节省向量化成本

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- 虚拟环境（推荐使用 conda 或 venv）

### 安装步骤

#### 1. 克隆仓库

```bash
git clone https://github.com/ElliotSky/WXDown-Web-RAG.git
cd WXDown-Web-RAG
```

#### 2. 安装依赖

```bash
pip install -r requirements.txt
```

或安装为可执行包：

```bash
pip install -e .
```

#### 3. 配置环境变量

复制环境变量模板：

```bash
# Windows
copy configs\env.example .env

# Linux / macOS
cp configs/env.example .env
```

编辑 `.env` 文件，配置必要的 API Key：

```env
# 嵌入模型 API（用于向量化）
EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# 对话模型 API（用于生成回答）
CHAT_API_KEY=your_chat_api_key
CHAT_BASE_URL=https://api.deepseek.com
```

#### 4. 准备数据

将微信公众号导出的 PDF 文件按以下结构组织：

```
data/WXhub/
├── <公众号名1>/
│   ├── pdf/
│   │   ├── 2024-01-01-文章标题1.pdf
│   │   └── 2024-01-02-文章标题2.pdf
│   └── db/
│       └── db.jsonl  # 可选：元数据文件
└── <公众号名2>/
    └── pdf/
        └── ...
```

#### 5. 构建向量库

```bash
python -m wxchatrag.cli ingest
```

#### 6. 开始问答

```bash
python -m wxchatrag.cli query --question "你的问题"
```

---

## 📁 项目结构

```
WXdown-RAG/
├── configs/
│   ├── env.example          # 环境变量配置模板
│   └── wxchatrag.json       # RAG 运行配置（数据路径、模型、切分参数等）
├── data/
│   └── WXhub/               # 微信公众号数据根目录（可由 WXHUB_ROOT 覆盖）
├── storage/
│   ├── vector_store/        # FAISS 向量库与 manifest.json
│   ├── bm25_index/          # BM25 索引存储
│   └── models/               # 本地模型缓存（Rerank 等）
├── src/
│   └── wxchatrag/
│       ├── cli.py            # 命令行入口
│       ├── services.py       # 入库服务 / 查询服务封装
│       ├── exceptions.py     # 统一异常类型定义
│       ├── ingest.py         # 加载 PDF、切分、向量化、落盘
│       ├── rag_query.py      # 检索、构造 Prompt、调用 LLM
│       ├── wxhub_loader.py   # 微信数据加载与元数据构建
│       ├── manifest.py       # PDF 变更追踪，支持增量更新
│       ├── settings.py       # 配置加载（.env + JSON）
│       ├── chunking/         # 文档切分策略模块
│       │   ├── chunk_strategy.py
│       │   ├── fixed_splitter.py
│       │   ├── sentence_splitter.py
│       │   ├── semantic_splitter.py
│       │   ├── hierarchical_splitter.py
│       │   └── sliding_window_splitter.py
│       ├── embeddings/       # 嵌入模型模块
│       │   └── bge_embeddings.py
│       ├── retrieval/        # 检索模块
│       │   ├── bm25_store.py
│       │   ├── hybrid_retriever.py
│       │   └── rrf_fusion.py
│       ├── rerank/           # 重排序模块
│       │   ├── reranker.py
│       │   └── cross_encoder_rerank.py
│       └── filtering/       # 元数据过滤模块（预留）
├── tests/                    # 测试目录
├── requirements.txt          # Python 依赖
├── pyproject.toml            # 项目配置
└── README.md                 # 项目文档
```

> **注意**：`.gitignore` 中已忽略 `storage/`、`data/` 与 `WXhub/`，避免将大文件上传到仓库。

---

## ⚙️ 配置说明

### 配置优先级

```
环境变量（.env） > configs/wxchatrag.json > 代码默认值
```

### 环境变量配置（`.env`）

#### 必填配置

```env
# 嵌入模型 API（用于向量化）
EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# 对话模型 API（用于生成回答）
CHAT_API_KEY=your_chat_api_key
CHAT_BASE_URL=https://api.deepseek.com
```

#### 可选配置

```env
# 数据路径
WXHUB_ROOT=data/WXhub
VECTOR_STORE_DIR=storage/vector_store
BM25_INDEX_DIR=storage/bm25_index
MODELS_CACHE_DIR=storage/models

# 模型配置
EMBEDDING_MODEL_NAME=embedding-3
CHAT_MODEL_NAME=deepseek-chat

# 文档切分
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
CHUNK_STRATEGY=fixed  # fixed | sentence | semantic | hierarchical | sliding_window
SEMANTIC_THRESHOLD=0.5
SEMANTIC_EMBEDDING_MODE=local  # local | api
SEMANTIC_EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5

# 检索配置
RETRIEVAL_STRATEGY=hybrid  # vector | bm25 | hybrid
RETRIEVER_K=5
HYBRID_ALPHA=0.7
BM25_K=20
VECTOR_K=20

# 重排序配置
ENABLE_RERANK=true
RERANK_MODEL_NAME=BAAI/bge-reranker-base
RERANK_TOP_N=20
RERANK_TOP_K=5

# 生成配置
TEMPERATURE=0.2
```

### 配置文件（`configs/wxchatrag.json`）

```jsonc
{
  // 数据路径配置
  "wxhub_root": "data/WXhub",
  "pdf_subdir_name": "pdf",
  "pdf_glob_pattern": "",
  "vector_store_dir": "storage/vector_store",
  "bm25_index_dir": "storage/bm25_index",
  "models_cache_dir": "storage/models",
  
  // 模型配置
  "embedding_model_name": "embedding-3",
  "chat_model_name": "deepseek-chat",
  
  // 文档切分配置
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "chunk_strategy": "fixed",
  "semantic_threshold": 0.5,
  "semantic_embedding_mode": "local",
  "semantic_embedding_model_name": "BAAI/bge-small-zh-v1.5",
  
  // 检索配置
  "retrieval_strategy": "hybrid",
  "retriever_k": 5,
  "hybrid_alpha": 0.7,
  "bm25_k": 20,
  "vector_k": 20,
  
  // 重排序配置
  "enable_rerank": true,
  "rerank_model_name": "BAAI/bge-reranker-base",
  "rerank_top_n": 20,
  "rerank_top_k": 5,
  
  // 生成配置
  "temperature": 0.2
}
```

### 文档切分策略说明

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `fixed` | 固定大小切分 | 通用场景，快速切分 |
| `sentence` | 基于句子的切分 | 需要保持句子完整性 |
| `semantic` | 语义切分（基于 BGE 模型） | 需要保持语义完整性，质量优先 |
| `hierarchical` | 层次化切分（先段落后句子） | 结构化文档（如技术文档） |
| `sliding_window` | 滑动窗口切分 | 需要重叠上下文 |

### 检索策略说明

| 策略 | 说明 | 特点 |
|------|------|------|
| `vector` | 纯向量检索 | 语义理解强，适合概念性查询 |
| `bm25` | 纯 BM25 检索 | 关键词匹配精确，适合精确术语查询 |
| `hybrid` | 混合检索（RRF 融合） | 结合两者优势，推荐使用 |

---

## 📖 使用指南

### CLI 命令

所有命令均在项目根目录执行，确保虚拟环境已激活。

#### 1. 构建向量库（`ingest`）

**增量更新（默认推荐）**

```bash
python -m wxchatrag.cli ingest
```

**全量重建**

```bash
python -m wxchatrag.cli ingest --mode rebuild
```

**限制处理数量（调试用）**

```bash
python -m wxchatrag.cli ingest --mode update --limit 50
```

> **说明**：增量模式通过 `storage/vector_store/manifest.json` 记录已向量化的 PDF，只处理新增或变更的文件。

#### 2. 在线问答（`query`）

**交互式问答**

```bash
python -m wxchatrag.cli query
```

**一次性问答 + 命中文档来源**

```bash
python -m wxchatrag.cli query --question "DeepSeek 的架构特点是什么？" --with-sources
```

**JSON 输出（便于集成）**

```bash
python -m wxchatrag.cli query --question "DeepSeek 近期有哪些重要更新？" --json
```

**检索调试模式**

```bash
python -m wxchatrag.cli query \
  --question "肠道菌群的代谢物如何影响人体心理健康？" \
  --with-sources \
  --debug-retrieval \
  --preview-chars 200
```

**参数说明**

- `--with-sources`：输出命中文档来源（频道、日期、标题、页码、URL、本地路径等）
- `--json`：以 JSON 格式输出 `answer` 与 `sources`
- `--debug-retrieval`：输出检索中间状态，便于调试
- `--preview-chars`：控制每个命中文档的正文预览长度（字符数），`0` 表示不打印预览

#### 3. 向量库检查（`embedded-pdfs`）

列出当前向量库中已向量化的 PDF 清单：

```bash
python -m wxchatrag.cli embedded-pdfs
```

该命令直接从 FAISS 向量库读取文档元数据，真实反映当前已入库的 PDF。

---

## 🏗️ 技术架构

### RAG 流程概览

#### 1. 知识入库（Ingest Pipeline）

```
PDF 文档
  ↓
加载与元数据补全（wxhub_loader.py）
  ↓
文档切分（chunking/）
  ├── fixed_splitter.py
  ├── sentence_splitter.py
  ├── semantic_splitter.py（基于 BGE）
  ├── hierarchical_splitter.py
  └── sliding_window_splitter.py
  ↓
向量化（OpenAIEmbeddings / BGEEmbeddings）
  ↓
构建索引
  ├── FAISS 向量库（vector_store/）
  └── BM25 索引（bm25_index/）
  ↓
持久化存储（manifest.json 记录变更）
```

#### 2. 在线问答（Query Pipeline）

```
用户问题
  ↓
检索阶段
  ├── 向量检索（FAISS）
  ├── BM25 检索
  └── 混合检索（RRF 融合）
  ↓
重排序（可选，Cross-Encoder）
  ↓
构造上下文与 Prompt
  ↓
调用 LLM 生成回答
  ↓
返回答案与来源列表
```

### 核心模块

- **`chunking/`**：文档切分策略模块，支持 5 种切分策略
- **`embeddings/`**：嵌入模型封装，支持 OpenAI API 和本地 BGE 模型
- **`retrieval/`**：检索模块，实现 BM25、向量检索和混合检索
- **`rerank/`**：重排序模块，使用 Cross-Encoder 模型优化检索结果
- **`settings.py`**：统一配置管理，支持环境变量和 JSON 配置

---

## 📄 许可证

如仓库未显式提供 `LICENSE` 文件，则默认视为**保留所有权利**，仅供学习与个人使用。若需在生产或商业环境中使用，请先与作者沟通确认授权方式。

---

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - LLM 应用开发框架
- [FAISS](https://github.com/facebookresearch/faiss) - 高效的相似度搜索和密集向量聚类库
- [BGE](https://github.com/FlagOpen/FlagEmbedding) - 北京智源研究院的通用嵌入模型
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) - BM25 检索实现

---

<div align="center">

**⭐ 如果这个项目对你有帮助，欢迎 Star！**

</div>
