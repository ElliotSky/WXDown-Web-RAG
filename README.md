## WXdown-RAG

基于微信公众号 PDF 语料的本地 RAG（Retrieval-Augmented Generation）检索问答项目。

- **`data/WXhub/`**：推荐的微信公众号数据根目录（PDF/HTML/图片/JSONL 等）。
- **`wxchatrag`**：围绕 `data/WXhub/` 中的 PDF 构建向量索引，并提供问答接口的 RAG 实现。

代码默认只认“新架构”的数据根路径（`data/WXhub/` 或显式配置的 `WXHUB_ROOT`），不再隐式回退到历史的 `WXhub/` 目录。

### 目录结构（规范化建议）

建议整体目录大致如下（省略无关文件，仅保留关键结构）：

```text
WXdown-RAG/
├─ configs/
│  ├─ env.example          # 环境变量示例（API Key、DeepSeek / OpenAI 兼容配置等）
│  └─ wxchatrag.json       # RAG 运行配置（数据路径、模型名、切分参数等）
├─ data/
│  └─ WXhub/               # 推荐的微信数据根目录（实际可用 WXHUB_ROOT 指向任意位置）
├─ storage/
│  └─ vector_store/        # FAISS 向量库与 manifest.json（自动创建，已在 .gitignore 中忽略）
├─ src/
│  └─ wxchatrag/
│     ├─ cli.py            # 命令行入口：ingest / query 子命令
│     ├─ services.py       # RAG 服务层封装（入库服务 / 查询服务）
│     ├─ exceptions.py     # 统一异常类型定义
│     ├─ ingest.py         # 入库流程底层工具：加载 PDF、切分、向量化、落盘
│     ├─ rag_query.py      # RAG 在线问答：检索、构造 Prompt、调用 LLM
│     ├─ wxhub_loader.py   # 微信数据加载与元数据构建（频道、日期、URL 等）
│     ├─ manifest.py       # PDF 变更追踪（大小/mtime），支持增量更新
│     └─ settings.py       # 配置加载（.env + JSON），路径与模型参数统一管理
├─ tests/                  # 单元测试目录（当前提供最小占位测试）
└─ requirements.txt / pyproject.toml
```

.gitignore 中已忽略 `storage/`、`data/` 与 `WXhub/`，避免将大文件上传到仓库。

### 安装依赖

最小依赖安装（直接用本地源码运行）：

```bash
pip install -r requirements.txt
```

如需安装为可执行包（支持 `wxchatrag` 命令）：

```bash
pip install -e .
```

> 如果你使用 conda / venv，建议先创建并激活虚拟环境再安装。

### 环境与配置

1. **复制环境变量模板**

```bash
copy configs\env.example .env   # Windows PowerShell / CMD
# 或
cp configs/env.example .env     # Linux / macOS
```

2. **在 `.env` 中填写关键变量**

- **OPENAI_API_KEY**：DeepSeek / OpenAI 兼容 Key（底层统一走 `openai`/`langchain-openai` 接口）。
- **OPENAI_BASE_URL**：如使用 DeepSeek，示例为 `https://api.deepseek.com`。
- （可选）**DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL**：如只填这两个，代码会自动回填到 `OPENAI_*`。
- （可选）**WXHUB_ROOT**：微信数据根目录，默认读取 `configs/wxchatrag.json` 中的 `wxhub_root`（推荐为 `data/WXhub/`）。
- （可选）**VECTOR_STORE_DIR**：向量库持久化目录，默认 `storage/vector_store`。
- （可选）**EMBEDDING_MODEL_NAME / CHAT_MODEL_NAME / CHUNK_SIZE / CHUNK_OVERLAP / RETRIEVER_K / TEMPERATURE**：
  覆盖 JSON 配置中的模型与参数。

3. **编辑 `configs/wxchatrag.json`（标准 RAG 配置）**

```jsonc
{
  "wxhub_root": "data/WXhub",          // 微信数据根目录（可被 WXHUB_ROOT 覆盖）
  "pdf_subdir_name": "pdf",            // 每个公众号下 PDF 子目录名
  "pdf_glob_pattern": "",              // 为空时自动使用 */pdf/*.pdf
  "vector_store_dir": "storage/vector_store",
  "embedding_model_name": "text-embedding-3-large",
  "chat_model_name": "deepseek-chat",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "retriever_k": 5,
  "temperature": 0.2
}
```

### 标准化 RAG 流程

#### 1. 知识入库（Ingest Pipeline）

对应代码：`ingest.py`、`wxhub_loader.py`、`manifest.py`。

1. **数据准备**
   - 将各公众号目录放入 `WXHUB_ROOT` 指向的路径（推荐 `data/WXhub/`，当前示例在仓库根的 `WXhub/`）。
   - 每个公众号目录下约定结构：`pdf/`（PDF 文件）、`db/db.jsonl`（原始抓取元数据，可选）。
2. **加载与补全元数据（load_pdf_documents）**
   - 使用 `wxhub_loader.iter_pdf_paths` 按 `pdf_glob_pattern` 收集所有 PDF 路径。
   - 通过 `db/db.jsonl` 中的标题索引补充频道名、发布时间、URL、消息链接等元信息。
3. **文档切分（split_docs）**
   - 使用 `RecursiveCharacterTextSplitter` 按 `chunk_size` / `chunk_overlap` 切分为语义块，
     并保留 `metadata`（包括来源 PDF 路径、频道、日期、页码等）。
4. **向量化与构建向量库（build_vector_store）**
   - 使用 `OpenAIEmbeddings(model=embedding_model_name)` 将每个 chunk 向量化。
   - 通过 `FAISS.from_documents` 构建向量索引。
5. **增量 / 重建策略**
   - 通过 `manifest.json` 记录每个 PDF 的 `size` 与 `mtime`。
   - `mode="update"`：只处理有变更的 PDF（`select_changed`）。
   - `mode="rebuild"`：清空旧的向量库目录后全量重建。
6. **持久化**
   - 使用 `FAISS.save_local` 将索引保存到 `vector_store_dir`，下次可直接加载增量更新。

#### 2. 在线问答（Query Pipeline）

对应代码：`rag_query.py`。

1. **加载向量库**
   - 通过 `load_vector_store` 从 `vector_store_dir` 加载 FAISS 索引（如不存在会抛错提示先入库）。
2. **检索相关文档**
   - 通过 `vs.as_retriever(search_type="similarity", k=retriever_k)` 按相似度检索 top-k 文档。
3. **构造上下文与来源列表**
   - ` _format_docs` 将每个文档格式化为：`[index] 标题 | 频道 | 日期 | 页码` + 正文片段。
   - 并构建 `RagSource` 列表，用于最终在终端或 JSON 中输出来源信息。
4. **构造 Prompt 并调用 LLM**
   - `_build_prompt` 使用统一的中文提示模板，将 `context` 与用户 `question` 填入。
   - 使用 `ChatOpenAI(model=chat_model_name, temperature=temperature)` 调用大模型生成答案。
5. **返回标准化结果**
   - `RagResponse` 包含 `answer`（最终回答）与 `sources`（来源列表）。
   - CLI 层可选择以文本或 JSON 格式输出，并可附加来源元数据。

### CLI 使用方式

所有命令均在项目根目录执行，确保虚拟环境已激活（下面示例以 Windows PowerShell 为例，`python` 可替换为你的环境路径）。

#### 1. 构建 / 更新向量库（ingest）

- **增量更新（默认推荐）**：仅处理**新增或变更**的 PDF，避免重复向量化浪费 Token  
  - 内部通过 `storage/vector_store/manifest.json` 记录「已完成向量化的 PDF 清单」，
    每次运行会对比 `size/mtime`，只对有变化的文件重新切分 + 向量化。
  - 测试GML embedding 6000 文本块花费1.3r

```bash
python -m wxchatrag.cli ingest          # 等价于 --mode update
```

- **全量重建**：**清空旧索引后，对所有 PDF 重新向量化**  
  - 仅在你调整了切分策略 / 向量模型，或怀疑索引损坏时使用。

```bash
python -m wxchatrag.cli ingest --mode rebuild
```

- **限制本次处理的 PDF 数量（调试用）**  
  - 只在本地调试时使用，例如你想先用少量 PDF 试跑一遍流程。  
  - 注意：`manifest.json` 只会标记**本次真正处理到的 PDF**，
    之后再次运行未带 `--limit` 的 `ingest` 时，其余未处理的 PDF 会被正常补充向量化。

```bash
python -m wxchatrag.cli ingest --mode update --limit 50
```

#### 2. 在线问答（query）

- **交互式问答**

```bash
python -m wxchatrag.cli query
```

终端会提示输入问题，回车后返回答案。

- **一次性带问题参数 + 命中文档来源**

```bash
python -m wxchatrag.cli query --question "DeepSeek 的架构特点是什么？" --with-sources
```

终端输出将包括：

- 第一部分：模型生成的最终回答。
- 第二部分：命中的文档来源列表（频道、日期、标题、页码、URL、本地路径等）。

- **JSON 输出（便于集成到上层服务）**

```bash
python -m wxchatrag.cli query --question "DeepSeek 近期有哪些重要更新？" --json
```

返回的 JSON 结构示例：

```jsonc
{
  "answer": "……",
  "sources": [
    {
      "index": 1,
      "title": "DeepSeek 今年的两个重大更新，一篇详细的总结来了！",
      "channel": "Datawhale",
      "date": "2026-01-28",
      "page": 3,
      "url": "https://mp.weixin.qq.com/...",
      "path": "F:/AIhub/WXdown-RAG/data/WXhub/Datawhale/pdf/2026-01-28-DeepSeek...pdf"
    }
  ]
}
```

- **调试模式：输出检索中间状态（命中 PDF / 页码 / 相似度 / 片段预览）**

```bash
python -m wxchatrag.cli query --question "肠道菌群的代谢物如何影响人体心理健康？" ^
  --with-sources ^
  --debug-retrieval ^
  --preview-chars 200
```

说明：

- `--debug-retrieval` 会在回答前，将检索到的 top-k 文档、相似度、页码、PDF 路径、片段预览等**输出到 stderr**，
  便于排查「为什么检索不到预期文档」之类的问题。  
- `--preview-chars` 控制每个命中文档打印的预览长度（字符数），设置为 `0` 表示不打印正文预览。

#### 3. 辅助命令：查看与排查向量库

- **列出已经向量化过的 PDF 清单**

```bash
python -m wxchatrag.cli embedded-pdfs
```

输出示例（节选）：

```text
=== 已向量化的 PDF 文件 ===
F:\AIhub\WXdown-RAG\data\WXhub\AI寒武纪\pdf\2026-02-19-人工智能时代真正重要的技能：你的品味.pdf
F:\AIhub\WXdown-RAG\data\WXhub\Datawhale\pdf\2026-01-28-DeepSeek今年的两个重大更新，一篇详细的总结来了！.pdf
...

总计 123 个 PDF 已向量化。
```

这个命令直接从 FAISS 向量库中读取文档元数据，**真实反映当前有哪些 PDF 已经进入向量库**，
适合在迁移或多次调试 `ingest` 之后，快速确认向量化覆盖范围。

### 检索效果示例

- 真实终端日志示例（包含「检索中间状态 + 命中文档 + 最终回答」的完整链路分析），见：
  - `docs/retrieval-examples.md`

### 备注

- 推荐将所有公众号数据统一迁移或挂载到 `data/WXhub/`，并通过 `WXHUB_ROOT` / `WXCHATRAG_CONFIG_PATH` 显式管理配置，便于多环境部署与备份。
