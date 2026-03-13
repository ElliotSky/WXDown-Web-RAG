## WXdown-RAG

基于微信公众号 PDF 语料的本地 RAG（Retrieval-Augmented Generation）检索问答项目，支持增量向量化、命中文档溯源与检索调试。

- **`data/WXhub/`**：推荐的微信公众号数据根目录（PDF/HTML/图片/JSONL 等）。
- **`wxchatrag`**：围绕 `data/WXhub/` 中的 PDF 构建向量索引，并提供问答 CLI 接口。

---

### 功能特性

- **本地知识库构建**
  - 针对微信公众平台导出的 PDF 进行加载、切分、向量化与索引构建。
- **增量更新机制**
  - 通过 `manifest.json` 记录 PDF `size/mtime`，支持只处理新增或变更文件，节省向量化成本。
- **标准化问答接口**
  - 提供统一的 `ingest` / `query` / `embedded-pdfs` 子命令与 JSON 输出格式。
- **检索调试能力**
  - 支持输出命中 PDF、页码、相似度与正文片段，方便调参与排错。

---

### 目录结构

```text
WXdown-RAG/
├─ configs/
│  ├─ env.example          # 环境变量示例（API Key、DeepSeek / OpenAI 兼容配置等）
│  └─ wxchatrag.json       # RAG 运行配置（数据路径、模型、切分参数等）
├─ data/
│  └─ WXhub/               # 推荐的微信数据根目录（可由 WXHUB_ROOT 覆盖）
├─ storage/
│  └─ vector_store/        # FAISS 向量库与 manifest.json（自动创建，已在 .gitignore 中忽略）
├─ src/
│  └─ wxchatrag/
│     ├─ cli.py            # 命令行入口：ingest / query / embedded-pdfs 子命令
│     ├─ services.py       # 入库服务 / 查询服务封装
│     ├─ exceptions.py     # 统一异常类型定义
│     ├─ ingest.py         # 加载 PDF、切分、向量化、落盘
│     ├─ rag_query.py      # 检索、构造 Prompt、调用 LLM
│     ├─ wxhub_loader.py   # 微信数据加载与元数据构建（频道、日期、URL 等）
│     ├─ manifest.py       # PDF 变更追踪，支持增量更新
│     └─ settings.py       # 配置加载（.env + JSON），统一管理路径和模型参数
├─ tests/                  # 单元测试目录（占位）
└─ requirements.txt / pyproject.toml
```

`.gitignore` 中已忽略 `storage/`、`data/` 与 `WXhub/`，避免将大文件上传到仓库。

---

### 环境准备与安装

#### 1. 安装依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

如需安装为可执行包（支持在任意目录通过 `wxchatrag` 命令调用）：

```bash
pip install -e .
# 或 pip install -e . --no-deps 
```

> 建议使用虚拟环境（如 conda / venv），并在安装前先激活环境。

#### 2. 准备 `.env`

```bash
copy configs\env.example .env   # Windows PowerShell / CMD
# 或
cp configs/env.example .env     # Linux / macOS
```

在 `.env` 中配置以下变量（全部为大写）：

- **OPENAI_API_KEY**（必填）  
  - DeepSeek / OpenAI 兼容 Key，底层统一通过 `openai` / `langchain-openai` 调用。
- **OPENAI_BASE_URL**（推荐）  
  - 使用 DeepSeek 时通常为 `https://api.deepseek.com`，使用其它兼容厂商时填对应 Base URL。
- **DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL**（可选）  
  - 若只填写这两个，代码会自动回填到 `OPENAI_API_KEY` / `OPENAI_BASE_URL`。
- **WXHUB_ROOT**（可选）  
  - 微信数据根目录，优先级高于 `wxchatrag.json` 中的 `wxhub_root`，默认推荐 `data/WXhub/`。
- **VECTOR_STORE_DIR**（可选）  
  - 向量库持久化目录，默认 `storage/vector_store`。
- **EMBEDDING_MODEL_NAME**（可选）  
  - 覆盖 JSON 中的 `embedding_model_name`，如 `text-embedding-3-large`。
- **CHAT_MODEL_NAME**（可选）  
  - 覆盖 JSON 中的 `chat_model_name`，如 `deepseek-chat`。
- **CHUNK_SIZE / CHUNK_OVERLAP**（可选）  
  - 文本切分长度与重叠长度，整数，用于 `RecursiveCharacterTextSplitter`。
- **RETRIEVER_K**（可选）  
  - 检索时返回的 top-k 文档数量。
- **TEMPERATURE**（可选）  
  - 生成温度，0.0–1.0，越低越保守。
- **WXCHATRAG_CONFIG_PATH**（可选）  
  - 自定义配置文件路径，默认 `configs/wxchatrag.json`。

> 配置优先级：**环境变量 > `configs/wxchatrag.json` > 代码默认值**。

---

### `configs/wxchatrag.json` 参数说明

示例：

```jsonc
{
  "wxhub_root": "data/WXhub",
  "pdf_subdir_name": "pdf",
  "pdf_glob_pattern": "",
  "vector_store_dir": "storage/vector_store",
  "embedding_model_name": "text-embedding-3-large",
  "chat_model_name": "deepseek-chat",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "retriever_k": 5,
  "temperature": 0.2
}
```

- **wxhub_root**  
  - 微信公众号数据根目录路径，推荐结构：`data/WXhub/<公众号名>/pdf/*.pdf`。  
  - 可被环境变量 `WXHUB_ROOT` 覆盖。
- **pdf_subdir_name**  
  - 每个公众号目录下存放 PDF 的子目录名，例如 `pdf`，即 `data/WXhub/<公众号名>/pdf/`。
- **pdf_glob_pattern**  
  - 自定义 PDF 匹配模式（glob），如 `"**/*.pdf"`；为空字符串时默认 `*/pdf/*.pdf`。
- **vector_store_dir**  
  - FAISS 向量库与 `manifest.json` 的持久化目录，可被 `VECTOR_STORE_DIR` 覆盖。
- **embedding_model_name**  
  - 向量化模型，对应 `OpenAIEmbeddings(model=...)`。
- **chat_model_name**  
  - 聊天模型，对应 `ChatOpenAI(model=...)`。
- **chunk_size**  
  - 单个文本块的最大字符长度。
- **chunk_overlap**  
  - 相邻文本块之间的重叠字符数，用于缓解句子被截断的问题。
- **retriever_k**  
  - 每次检索从向量库中召回的文本块数量。
- **temperature**  
  - 生成温度，控制回答多样性和稳定性。

---

### 微信数据目录要求

- **根目录**  
  - 由 `WXHUB_ROOT` 或 `wxhub_root` 指定，例如：`data/WXhub/`。
- **公众号目录结构约定**
  - `<WXHUB_ROOT>/<公众号名>/pdf/*.pdf`：存放该公众号导出的 PDF 文章。  
  - `<WXHUB_ROOT>/<公众号名>/db/db.jsonl`（可选）：存放抓取时的原始元数据，用于补全标题、发布时间、URL、消息链接等。

---

### 标准化 RAG 流程（概览）

#### 1. 知识入库（Ingest Pipeline）

主要涉及：`ingest.py`、`wxhub_loader.py`、`manifest.py`。

1. **数据准备**  
   - 将各公众号目录放入 `WXHUB_ROOT` 指向路径（推荐 `data/WXhub/`）。
2. **加载与补全元数据**  
   - `wxhub_loader.iter_pdf_paths` 按 `pdf_glob_pattern` 收集 PDF 路径。  
   - 通过 `db/db.jsonl` 补充频道名、发布时间、URL 等元信息（如存在）。
3. **文档切分**  
   - 使用 `RecursiveCharacterTextSplitter` 按 `chunk_size` / `chunk_overlap` 切分为语义块，并保留来源路径、频道、日期、页码等 `metadata`。
4. **向量化与构建向量库**  
   - 使用 `OpenAIEmbeddings(model=embedding_model_name)` 向量化。  
   - 通过 `FAISS.from_documents` 构建向量索引。
5. **增量 / 重建策略**  
   - 使用 `manifest.json` 记录每个 PDF 的 `size` 与 `mtime`。  
   - `mode="update"`：只处理有变更的 PDF。  
   - `mode="rebuild"`：清空旧索引后全量重建。

#### 2. 在线问答（Query Pipeline）

主要涉及：`rag_query.py`。

1. **加载向量库**  
   - 从 `vector_store_dir` 加载 FAISS 索引（如不存在会提示先执行 `ingest`）。
2. **检索相关文档**  
   - `vs.as_retriever(search_type="similarity", k=retriever_k)` 检索 top-k 文档。
3. **构造上下文与来源列表**  
   - `_format_docs` 生成用于 Prompt 的上下文字符串和标准化的来源列表（频道、日期、页码等）。
4. **调用 LLM 生成回答**  
   - `_build_prompt` 组装中文提示模板，将 `context` 与用户 `question` 注入。  
   - 通过 `ChatOpenAI(model=chat_model_name, temperature=temperature)` 生成答案。
5. **返回标准化结果**  
   - 返回 `RagResponse`：包含 `answer`（回答）与 `sources`（命中文档来源列表）。

---

### CLI 使用方式

所有命令均在项目根目录执行，确保虚拟环境已激活（下面示例以 Windows PowerShell 为例，`python` 可替换为你的 Python 解释器）。

#### 1. 构建 / 更新向量库（`ingest`）

- **增量更新（默认推荐）**

```bash
python -m wxchatrag.cli ingest
```

- **全量重建（清空旧索引后重建）**

```bash
python -m wxchatrag.cli ingest --mode rebuild
```

- **限制本次处理的 PDF 数量（调试用）**

```bash
python -m wxchatrag.cli ingest --mode update --limit 50
```

> 增量模式内部通过 `storage/vector_store/manifest.json` 记录已向量化的 PDF 清单，只对新增或变更的文件进行重向量化。

#### 2. 在线问答（`query`）

- **交互式问答**

```bash
python -m wxchatrag.cli query
```

- **一次性问答 + 命中文档来源**

```bash
python -m wxchatrag.cli query --question "DeepSeek 的架构特点是什么？" --with-sources
```

- **JSON 输出（便于集成到上层服务）**

```bash
python -m wxchatrag.cli query --question "DeepSeek 近期有哪些重要更新？" --json
```

- **检索调试模式**

```bash
python -m wxchatrag.cli query --question "肠道菌群的代谢物如何影响人体心理健康？" ^
  --with-sources ^
  --debug-retrieval ^
  --preview-chars 200
```

说明：

- **--with-sources**：在回答后输出命中文档来源（频道、日期、标题、页码、URL、本地路径等）。
- **--json**：以 JSON 格式输出 `answer` 与 `sources`，便于 HTTP / 前端集成。
- **--debug-retrieval**：在回答前将检索到的 top-k 文档信息输出到 stderr，便于调试检索效果。
- **--preview-chars**：控制每个命中文档的正文预览长度（字符数），`0` 表示不打印正文预览。

#### 3. 向量库检查（`embedded-pdfs`）

列出当前向量库中已经向量化的 PDF 清单：

```bash
python -m wxchatrag.cli embedded-pdfs
```

该命令直接从 FAISS 向量库中读取文档元数据，真实反映当前已入库的 PDF，适合在多次调试或迁移后快速确认覆盖范围。

---

### 开发与测试

- **运行单元测试**

```bash
pytest
```

- **扩展建议**
  - 可以在保留 `services.py` 接口的前提下，接入 HTTP API / Web UI / 机器人等上层应用。

---

### 检索效果示例

- 更完整的检索日志与效果示例，见 `docs/retrieval-examples.md`。

---

### 许可证

如仓库未显式提供 `LICENSE` 文件，则默认视为**保留所有权利**，仅供学习与个人使用。若需在生产或商业环境中使用，请先与作者沟通确认授权方式。


