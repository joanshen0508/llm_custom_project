class ExampleTool:
    def can_handle(self,plan):
        return "search" in plan
    
    def run(self,plan):
        # for example link to api, database and 知识库
        return f"[工具调用]: 我假装搜索到 {plan}"