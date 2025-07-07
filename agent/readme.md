LLM (大脑) + 工具 (外部能力) + 记忆 (知识库/数据库) + 行动器 (执行操作)
举个典型的 Agent 组件：

组成	做什么
LLM	生成计划、理解指令
Tool Calling	调用外部 API（数据库、爬虫、RAG、搜索）
Memory	长期记忆/上下文缓存
Planner	多步推理，分解任务
Executor	调用工具并执行
Controller	负责循环、迭代直到完成

how to add agent in your project
Agent 架构是在推理之上包一层：

输入指令

由 Agent 调用：

你的 LLM（做语言理解）

外部工具（数据库查结果、爬取最新文献）

根据上下文做循环

输出答案或执行结果