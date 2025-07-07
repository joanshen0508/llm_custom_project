class BaseAgent:
    def __init__(self, llm_pipeline, tools=[],memory=None,retriever=None):
        self.llm_pipeline =llm_pipeline
        self.tools =tools
        self.memory = memory
        self.retriever = retriever

    def run(self,user_input):
        # if model contains a RAG, use the RAG first
        context =""
        if self.retriever is not None:
            passages = self.retriever.retrieve(user_input)
            context ="\n".join(passages)

            #  concat prompt and feed into the LLM
            prompt =f"
            用户问题:{user_input}
            相关资料:{context}
            请结合资料回答:
            "
            # "use llm for analysing intention and calling the tools"
            # 1. ask llm to analyse the intention of the user
            plan = self.llm_pipeline(prompt)[0]["generate_text"]
        else:
            prompt =f"分解这句话的步骤:{user_input}"
            plan = self.llm_pipeline(prompt)[0]["generate_text"]
        # 2. based on the plan, and call the tool
        result = None
        for tool in self.tools:
            if tool.can_handle(plan):
                result = tool.run(plan)
                break
        # 3. combine the results and return
        final_response = self.llm_pipeline(f"the user asks: {user_input},tool_result:{result},请总结成回答")[0]["generate_text"]

        return final_response