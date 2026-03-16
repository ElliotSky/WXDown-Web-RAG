# Chunk 策略说明文档

## 概述

本项目实现了 5 种主流的文档切分策略，可以根据不同场景选择最适合的策略。

## 支持的策略

### 1. Fixed Size Chunking（固定大小切分）

**策略名称**: `fixed`

**实现**: `FixedSizeSplitter`

**核心思想**：
- 以**字符长度**为主导，将长文本切成大小近似的固定块；  
- 通过一系列分隔符（段落、句子、单词、字符）递归尝试，以“尽量在自然边界切开”为辅助目标。

**工作原理（概念层面）**：
1. 定义一组按优先级排序的分隔符（如段落空行、换行、句号、空格等）；  
2. 以 `chunk_size` 为目标长度，从上到下尝试在合适分隔符附近断开；  
3. 若在当前层级找不到合适分隔点，则退化到更细粒度的分隔符，最终必要时直接按字符硬切；  
4. 相邻块之间根据 `chunk_overlap` 保留一定字符重叠，用于缓冲被截断的上下文。

**特点**:
- 实现简单、**速度快**，几乎不依赖额外模型；  
- 块大小稳定，可控性强，适合大规模批处理；  
- 语义边界并非显式建模，更多依赖分隔符启发式。

**适用场景**:
- 通用文档处理流水线；  
- 对预处理吞吐量要求较高的场景；  
- 文档结构不明确、但对“语义边界精细度”要求不高的应用。

**成本与权衡**:
- 计算成本**最低**，主要是字符串处理；  
- 可能在句子或语义单元中部截断，块内部主题可能稍显混杂；  
- 通过合理设置 `chunk_size` 和 `chunk_overlap` 可以在“块大小稳定性”和“语义完整性”之间取得折中。

**配置示例**（概念）:
```python
chunks = ChunkStrategy.split_documents(
    documents=docs,
    strategy="fixed",
    chunk_size=1000,
    chunk_overlap=200,
)
```

---

### 2. Sentence-based Chunking（基于句子的切分）

**策略名称**: `sentence`

**实现**: `SentenceSplitter`

**核心思想**：
- 先按**句子边界**将文本拆分为语义原子单元，再在句子序列上按长度阈值组合成 chunk；  
- 优先保证“句子完整不被截断”，在此基础上兼顾块大小和重叠。

**工作原理（概念层面）**：
1. 利用正则表达式识别中英文句号、问号、感叹号以及段落空行，将文本拆分为句子列表；  
2. 以 `chunk_size` 为上限，顺序累加句子组成当前块，直到预计再加入一个句子会超限；  
3. 在块边界处，从尾部回溯若干句子作为重叠部分（受 `chunk_overlap` 约束），并作为下一个块的开头；  
4. 对于极少数“单句长度远超 `chunk_size`”的情况，再退回到字符级切分做兜底处理。

**特点**:
- 保留句子完整性，块内部语义更连贯、表达更自然；  
- 对问答、摘要等任务更友好，避免“一半句子在上一个块、一半在下一个块”的情况；  
- 仍然保持较低计算开销，仅依赖字符串和正则操作。

**适用场景**:
- 问答系统、对话系统中作为检索上下文；  
- 高度依赖句子语义边界的场景（说明文、教程、政策解读等）；  
- 文档翻译或摘要等需要以句子为最小单位的任务。

**成本与权衡**:
- 相比固定大小切分，块大小的波动稍大（受句子长短影响）；  
- 在句子极长或标点稀少的文本中，可能退化为近似固定大小分块；  
- 是“语义友好性”和“实现复杂度/性能”之间的折中方案。

**配置示例**（概念）:
```python
chunks = ChunkStrategy.split_documents(
    documents=docs,
    strategy="sentence",
    chunk_size=1000,
    chunk_overlap=200,
)
```

---

### 3. Semantic Chunking（语义切分）

**策略名称**: `semantic`

**实现**: `SemanticSplitter`

**核心思想**：
- 将文档先划分为较小的语义单元（句子或短段落），再基于**向量表示的相似度曲线**自动确定“应该在何处断开”；
- 相邻单元之间语义相似度较低的位置，被视为“语义边界”，优先作为 chunk 的切分点；
- 块的大小由 **语义结构 + 最小块长度约束（`min_chunk_size`）** 共同决定，而不是简单的固定字符数。

**工作原理（概念层面）**：
1. **语义单元划分**  
   - 使用正则表达式将文本划分为句子或小段落；  
   - 本项目内置多种预设：  
     - `english`：适合英文文档；  
     - `chinese`：适合中文文档；  
     - `mixed`：适合中英混合场景；  
   - 也支持自定义分句正则，以适配特殊文本格式。
2. **向量化与相似度计算**  
   - 使用嵌入模型（如 OpenAI Embeddings 或本地向量模型）将每个语义单元映射到向量空间；  
   - 对**相邻语义单元**计算语义相似度（通常为余弦相似度），得到一条“相似度曲线”。
3. **语义断点识别**  
   - 对相似度序列应用阈值策略：  
     - `percentile`：根据百分位数找到“相对较低”的相似度点；  
     - `standard_deviation`：基于均值与标准差识别“异常低”的相似度点；  
   - 将这些低相似度点视为候选断点，优先在此处切分。
4. **块构建与长度约束**  
   - 在上述语义断点的基础上，结合 `min_chunk_size`（最小块大小）进行约束：  
     - 过短的块会与前后单元合并，避免产生碎片化的段落；  
     - 对于过长的连续高相似度区域，则会在合适位置进行二次切分，保证块大小和可用性。

**特点**:
- **语义感知**：切分边界与主题变化高度相关，比单纯按字符/句子长度更贴近“人类阅读逻辑”；  
- **自适应结构**：在结构复杂、主题切换频繁的文档中，能自动找到较为自然的分段位置；  
- **可调节的敏感度**：通过 `similarity_threshold` 与阈值类型（`breakpoint_threshold_type`）控制“多切/少切”；  
- **语言无关**：只要嵌入模型对目标语言有效，理论上可以适用于多种语言和混合文本。

**适用场景**:
- 关注“语义边界”的高价值内容：长报告、技术文档、研究论文、政策解读等；  
- 固定大小 / 句子切分效果一般，容易出现“一个 chunk 内主题过多或主题被强行拆散”的情况；  
- 对检索命中质量和上下文连贯性要求较高，可以接受一定的预处理开销。

**成本与权衡**:
- **计算成本较高**：需要对每个语义单元调用嵌入模型，并计算相邻相似度，适合**小到中等规模**文档集，或只对关键文档开启；  
- **对嵌入模型质量敏感**：嵌入模型越能捕捉语义关系，切分边界越“自然”；  
- **需要额外配置**：必须提供 `embeddings` 实例，并正确配置 API Key 或本地模型环境。

**配置示例**（概念）:
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
chunks = ChunkStrategy.split_documents(
    documents=docs,
    strategy="semantic",
    chunk_size=1000,
    chunk_overlap=200,
    embeddings=embeddings,
    similarity_threshold=0.5,
)
```

---

### 4. Hierarchical Chunking（层次化切分）

**策略名称**: `hierarchical`

**实现**: `HierarchicalSplitter`

**核心思想**：
- 将文档的**自然层次结构（段落 → 句子）**显式建模，在保留层级信息的前提下构建 chunk；  
- 通过分层切分，使得每个块既包含完整句子，又能追溯其所在段落与位置。

**工作原理（概念层面）**：
1. 首先依据段落分隔符（如连续空行）将文档拆分为段落序列；  
2. 在每个段落内部，再按句子边界进行细粒度拆分；  
3. 组合段落内/跨段落的句子，控制单块长度不超过 `chunk_size`，并按需要引入重叠；  
4. 在元数据中记录 `chunk_level`、`paragraph_index` 等信息，为后续检索结果解释和可视化提供结构线索。

**特点**:
- 显式保留文档的层次结构，有利于**结构化文档**（如论文、报告、技术文档）的理解与展示；  
- 结合段落与句子两级边界，比单纯的句子切分更贴合“自然段落语义”；  
- 在需要追踪“这个回答来自哪一段/哪一节”时，非常有用。

**适用场景**:
- 有清晰段落结构的文档：学术论文、技术白皮书、长篇报告、说明书等；  
- 需要在前端 UI 或评估工具中展示“段落级来源信息”的系统；  
- 希望在检索阶段就利用文档结构作为先验信息的场景。

**成本与权衡**:
- 相比 fixed/sentence，预处理逻辑更复杂，但仍然是规则驱动，计算成本中等；  
- 对原始文档排版质量依赖较强（段落分隔符不规范会影响效果）；  
- 在结构松散、段落界限模糊的文本中，优势会减弱。

**配置示例**（概念）:
```python
chunks = ChunkStrategy.split_documents(
    documents=docs,
    strategy="hierarchical",
    chunk_size=1000,
    chunk_overlap=200,
    paragraph_separator="\n\n",  # 可选，默认 "\n\n"
)
```

**元数据字段**:
- `chunk_level`: "paragraph" 或 "sentence"
- `paragraph_index`: 段落索引

---

### 5. Sliding Window Chunking（滑动窗口切分）

**策略名称**: `sliding_window`

**实现**: `SlidingWindowSplitter`

**核心思想**：
- 使用固定长度的“滑动窗口”在文本上按步长前进，每次截取一个窗口作为 chunk；  
- 通过窗口步长与窗口大小的差值来控制**重叠区域的大小**，确保关键内容被多次覆盖。

**工作原理（概念层面）**：
1. 设定窗口大小 `chunk_size` 和步长 `window_step`（若未指定，通常为 `chunk_size - chunk_overlap`）；  
2. 从起始位置开始，以 `window_step` 为单位滑动，每次截取长度为 `chunk_size` 的片段；  
3. 窗口之间自然形成固定、可控的重叠区；  
4. 末尾不足一个完整窗口的部分，可选择单独成块或并入前一个块。

**特点**:
- 所有文本都会被覆盖，且重要内容往往会出现在多个块中，提升召回概率；  
- 重叠大小可精确调节，便于在“召回率”和“冗余度”之间做量化权衡；  
- 实现简单、性能开销低，适合大规模场景。

**适用场景**:
- 对**召回率**要求较高，希望减少“刚好切在边界导致信息缺失”的风险；  
- 模型上下文窗口较大，希望通过多次覆盖关键信息来提高命中概率；  
- 滑动窗口检索、时间序列文本（如日志、对话记录）等。

**成本与权衡**:
- 会产生比 fixed 更**多的块数**，向量化与检索成本随之增加；  
- 重叠越大，信息冗余越高，但也越不容易漏掉重要线索；  
- 适合作为“召回优先”的策略，在后续通过重排序或 rerank 控制最终结果质量。

**配置示例**（概念）:
```python
chunks = ChunkStrategy.split_documents(
    documents=docs,
    strategy="sliding_window",
    chunk_size=1000,
    chunk_overlap=200,
    window_step=800,  # 可选，默认 chunk_size - chunk_overlap
)
```

**元数据字段**:
- `window_start`: 窗口起始位置
- `window_end`: 窗口结束位置
- `chunk_index`: 块索引

---

## 使用方式

### 方式一：通过 ChunkStrategy 工厂类

```python
from wxchatrag.chunking import ChunkStrategy
from langchain_core.documents import Document

documents = [Document(page_content="...", metadata={...})]

chunks = ChunkStrategy.split_documents(
    documents=documents,
    strategy="fixed",  # 选择策略
    chunk_size=1000,
    chunk_overlap=200,
)
```

### 方式二：直接使用切分器类

```python
from wxchatrag.chunking import FixedSizeSplitter

splitter = FixedSizeSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(documents)
```

### 方式三：通过配置文件

在 `configs/wxchatrag.json` 中配置：

```json
{
  "chunk_strategy": "fixed",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "semantic_threshold": 0.5
}
```

然后在 `ingest` 时会自动使用配置的策略。

---

## 策略对比

| 策略 | 速度 | 语义完整性 | 计算成本 | 适用场景 |
|------|------|-----------|---------|---------|
| fixed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 低 | 通用场景 |
| sentence | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 低 | 需要句子完整性 |
| semantic | ⭐⭐ | ⭐⭐⭐⭐⭐ | 高 | 需要语义边界 |
| hierarchical | ⭐⭐⭐ | ⭐⭐⭐⭐ | 中 | 结构化文档 |
| sliding_window | ⭐⭐⭐⭐ | ⭐⭐⭐ | 低 | 需要更多重叠 |

---

## 性能建议

1. **小规模数据（< 1000 文档）**: 可以使用 `semantic` 策略获得最佳效果
2. **中等规模数据（1000-10000 文档）**: 推荐使用 `fixed` 或 `sentence`
3. **大规模数据（> 10000 文档）**: 推荐使用 `fixed` 或 `sliding_window`

---

## 对比分析

使用 `notebooks/chunk_strategy_comparison.ipynb` 可以：
- 对比不同策略的切分效果
- 查看统计指标（块数量、平均长度等）
- 可视化块大小分布
- 查看具体切分结果

运行方式：
```bash
jupyter notebook notebooks/chunk_strategy_comparison.ipynb
```

---

## 最佳实践

1. **首次使用**: 先用 `fixed` 策略快速验证流程
2. **效果优化**: 根据检索效果尝试 `sentence` 或 `hierarchical`
3. **质量优先**: 如果对质量要求高，可以尝试 `semantic`
4. **召回优先**: 如果召回率不足，可以尝试 `sliding_window` 增加重叠

---

**最后更新**: 2026-03-12

