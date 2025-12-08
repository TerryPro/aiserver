# AI Server 会话管理方案设计文档 (v2.0)

## 1. 背景与目标
为了提供更精细化的 AI 辅助编程体验，我们需要对用户在 Jupyter Notebook 中每一个代码单元格（Cell）的开发过程进行全程跟踪。
新的会话管理机制将**以单元格（Cell）为核心**，记录用户针对该单元格的所有 AI 交互历史（包括需求描述、AI 生成的代码、中间修改过程等）。这不仅有助于支持多轮对话修正，还能够为用户提供“回溯”功能，查看某个单元格的代码是如何一步步演变而来的。

## 2. 核心设计原则

1.  **以 Cell 为粒度 (Cell-Granularity)**：
    会话的生命周期与单个 Cell 的 AI 开发过程绑定。不同的 Cell 拥有独立的会话，互不干扰。
2.  **文件持久化 (File Persistence)**：
    会话数据不再仅存储于内存，而是实时持久化为 JSON 文件。这确保了即使 Jupyter Server 重启，用户的开发历史也不会丢失。
3.  **结构化存储**：
    每个会话文件完整记录了交互的“全貌”：用户的请求（Request）、AI 的应答（Response）以及生成的代码快照。

## 3. 存储方案

### 3.1 命名规范
会话文件采用 JSON 格式存储，文件命名规则如下：
```
{notebook_id}_{cell_id}.json
```
- **notebook_id**: Notebook 的唯一标识（可以使用文件路径的哈希或前端生成的 UUID）。
- **cell_id**: Jupyter Notebook 单元格的唯一 ID (Notebook format 4.5+ 标准字段)。

### 3.2 存储路径
所有会话文件集中存储在项目根目录下的隐藏文件夹中：
```
<Project_Root>/.aiserver_sessions/
```

### 3.3 数据结构 (JSON Schema)

单个会话文件的内容结构设计如下：

```json
{
  "meta": {
    "version": "1.0",
    "notebook_id": "nb-uuid-123456",
    "cell_id": "cell-uuid-789012",
    "created_at": "2023-10-27T10:00:00.000Z",
    "last_updated": "2023-10-27T10:05:00.000Z"
  },
  "interactions": [
    {
      "turn_id": 1,
      "timestamp": "2023-10-27T10:00:05.000Z",
      "user_request": {
        "intent": "读取 data.csv 并显示前5行",
        "current_code": "",  // 请求时的代码状态
        "language": "python"
      },
      "ai_response": {
        "suggestion": "import pandas as pd\ndf = pd.read_csv('data.csv')\ndf.head()",
        "explanation": "使用了 pandas 的 read_csv 函数读取数据...",
        "status": "success",
        "error": null
      }
    },
    {
      "turn_id": 2,
      "timestamp": "2023-10-27T10:02:10.000Z",
      "user_request": {
        "intent": "修改为只读取 'price' 列",
        "current_code": "import pandas as pd\ndf = pd.read_csv('data.csv')\ndf.head()",
        "language": "python"
      },
      "ai_response": {
        "suggestion": "import pandas as pd\ndf = pd.read_csv('data.csv', usecols=['price'])\ndf.head()",
        "explanation": "添加了 usecols 参数...",
        "status": "success",
        "error": null
      }
    }
  ]
}
```

## 4. 架构设计

### 4.1 会话管理器 (SessionManager)

`SessionManager` 负责管理会话文件的 CRUD（创建、读取、更新、删除）。

*   **加载会话** (`load_session`): 根据 `notebook_id` 和 `cell_id` 查找并读取对应的 JSON 文件。如果不存在，则初始化一个新的会话对象。
*   **保存交互** (`save_interaction`): 将新的一轮对话（Request + Response）追加到 `interactions` 列表，并写入磁盘。
*   **获取历史** (`get_history`): 提供给 Prompt 构建器，用于组装多轮对话上下文。

### 4.2 交互流程

1.  **前端请求**：
    前端发送生成请求时，**必须**携带以下上下文信息：
    ```json
    {
      "notebookId": "...",
      "cellId": "...",
      "intent": "...",
      "source": "..."
    }
    ```

2.  **后端处理**：
    *   **Step 1**: Handler 接收请求，提取 `notebookId` 和 `cellId`。
    *   **Step 2**: `SessionManager` 读取对应的 `{notebookId}_{cellId}.json`。
    *   **Step 3**: 提取历史交互中的 `user_request.intent` 和 `ai_response.suggestion`，构建 LangChain 的 `History` 对象。
    *   **Step 4**: 调用 LLM 生成新代码。
    *   **Step 5**: `SessionManager` 将本次请求和结果追加到 JSON 文件中保存。

## 5. 优势
1.  **可追溯性**：用户可以随时查看某个单元格的 AI 辅助历史。
2.  **上下文隔离**：不同 Cell 的任务通常是独立的，这种设计避免了上下文污染。
3.  **轻量持久化**：JSON 文件易于读写和调试，不依赖复杂的数据库服务。

## 6. 待办事项
- [ ] 实现 `FileSessionManager` 类，替换原有的内存版 `SessionManager`。
- [ ] 更新 `GenerateHandler` 和 `AnalyzeDataFrameHandler` 以对接新的 Session 接口。
- [ ] 确定 `notebook_id` 的生成策略（建议由前端扩展负责生成并传递，例如使用 Notebook 的 path 或内部 ID）。
