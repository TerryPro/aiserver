# 会话管理开发任务清单

基于 `aiserver/docs/session_design.md` 和现有代码库状态，为了完善基于 Cell 的持久化会话管理功能，需要进行以下前后端开发工作。

## 1. 后端开发 (Backend - Python)

### 1.1 Prompt 模块升级
**目标**: 支持将历史对话上下文注入到 Prompt 中，使 AI 能够理解之前的交互。
- [ ] **修改 `aiserver/prompts/user.py`**:
    - 更新 `construct_user_prompt` 函数签名，增加 `history` 参数（类型建议为 `List[dict]` 或 `str`）。
    - 在 Prompt 构建逻辑中，解析 `history` 并将其格式化为清晰的对话文本（例如使用 `<HISTORY>` 标签包裹），添加到 Prompt 的合适位置（建议在“用户意图”之前）。

### 1.2 Handler 改造 (Code Generation)
**目标**: 在代码生成流程中集成会话管理（读取历史 -> 生成 -> 保存）。
- [ ] **修改 `aiserver/handlers/code_gen.py`**:
    - 在 `initialize` 方法中实例化 `SessionManager`。
    - 在 `post` 方法中：
        1. 从 request body 提取 `notebookId` 和 `cellId`。若前端未传，需做兼容处理（如生成临时 ID 或降级为无状态模式）。
        2. 调用 `session_manager.get_history(notebookId, cellId)` 获取历史记录。
        3. 将 `history` 传递给更新后的 `construct_user_prompt`。
        4. 获取 AI 响应后，调用 `session_manager.save_interaction(...)` 保存本次请求的意图、代码上下文以及 AI 的回复。

### 1.3 Handler 改造 (Data Analysis)
**目标**: 在数据分析流程中集成会话管理。
- [ ] **修改 `aiserver/handlers/data_analysis.py`**:
    - 逻辑同 `code_gen.py`。
    - 确保数据分析的特定参数（如 DataFrame metadata）也能在保存记录时被合理处理（或仅保存关键的 Intent 和 Suggestion）。

### 1.4 新增 History API
**目标**: 允许前端获取特定 Cell 的历史会话记录，用于“回溯”或展示历史。
- [ ] **创建 `aiserver/handlers/session.py`**:
    - 定义 `SessionHistoryHandler` 类。
    - 实现 `GET` 方法，接收 `notebook_id` 和 `cell_id` 参数。
    - 调用 `session_manager.load_session` 并返回完整的 session JSON 数据。
- [ ] **注册路由**:
    - 在 `aiserver/extension.py` (或路由注册入口) 添加映射：`(r"/api/aiserver/sessions", SessionHistoryHandler)`。

## 2. 前端开发 (Frontend - TypeScript/Jupyter Lab Extension)

### 2.1 请求参数升级
**目标**: 在发送 AI 生成请求时，携带当前上下文的唯一标识。
- [ ] **修改 API Client**:
    - 在构建请求 Payload 时，增加 `notebookId` 和 `cellId` 字段。
    - **Notebook ID**: 可以使用 Notebook 文件的相对路径（`context.path`），或者在前端为每个打开的 Notebook 生成一个 UUID 并缓存。建议使用路径以保证跨 Session 的持久性，但需注意文件重命名问题（设计文档建议使用路径哈希或 UUID）。
    - **Cell ID**: 直接获取 Jupyter Cell Model 的 ID (`cell.model.id`)。

### 2.2 (可选) 历史记录 UI
**目标**: 让用户能看到 AI 的思考历史。
- [ ] **开发历史记录面板**:
    - 在侧边栏或 Cell 工具栏添加“历史记录”按钮。
    - 点击时调用后端 `SessionHistoryHandler` 接口。
    - 渲染历史交互列表（User Intent -> AI Suggestion）。

## 3. 联调与测试计划

1.  **单轮对话测试**:
    - 发送请求 -> 检查 `.aiserver_sessions` 目录下是否生成了对应的 JSON 文件。
2.  **多轮对话测试**:
    - 针对同一 Cell 发送第二次请求（如“修改变量名为 x”）。
    - 检查后端日志，确认 Prompt 中是否包含了第一轮的对话历史。
    - 检查 JSON 文件是否追加了第二条 Interaction。
3.  **持久化测试**:
    - 重启 Jupyter Server。
    - 再次针对该 Cell 发送请求，验证后端能否读取到重启前的历史记录。
