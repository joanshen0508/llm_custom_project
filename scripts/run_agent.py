from transformers import pipeline
from agent.base_agent import BaseAgent
from agent.tools import ExampleTool
from agent.retriever import SimpleRAGRetriever

# 1. LOAD YOUR LLM
pipe = pipeline(
    "text-generation",
    model ="./checkpoints/checkpoints-xxx",
    tokenizer = "./checkpoints/checkpoint-xxx"
)

embed_RAG =True
tools = [ExampleTool()] 
if not embed_RAG:
    # agent without RAG
    """ user input->
        LLM procesing (reasoning +planning)->
        tool calling(optional, e.g. calculator, web search, database)->
        LLM final response generation->
        output to user
    """
    #  instantiate an agent
    agent = BaseAgent(pipe,tools=tools)
else:
    # agent with RAG
    # if you want to embed the RAG(retrival augmentation generation) into the agent, the workflow should be:
    """ user input->
        RAG retriever:searches vector databases or knowledge base for relevant context->
        combine: retrieved context +user input->
        LLM processing (reasoning + generation)->
        tool calling (optional)->
        final LLM output->
        output to user
    """
    corpus = [
        "IVD 是体外诊断，使用血液样本进行分析。",
        "ISO 13485 是医疗设备质量管理体系标准。",
        "深度学习在病理图像分析中可实现自动诊断。"]
    
    retriever = SimpleRAGRetriever(corpus=corpus)
    agent =BaseAgent(pipe,tools=tools,retriever=retriever)

# run an example
query = "help me to search the latest paper regarding IVD, and summarize the key point."
result = agent.run(query)

print("Agent Response:" ,result)

