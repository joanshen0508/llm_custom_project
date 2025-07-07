from llm_wrapper import LLMWrapper
from agent.base_agent import BaseAgent
from agent.tools import ExampleTool
from agent.retriever import SimpleRAGRetriever
import argparse

parser  =argparse.ArgumentParser()
parser.add_argument("--model_path",type=str,requried=True)
args = parser.parse_args()

# 1.load llm using wrapper
llm = LLMWrapper(model_path=args.model_path)

# adapter function for agent compatibility( use wrapper like pipeline)
class WrappedPipline:
    def __init__(self,llm) -> None:
        self.llm = llm
    
    def __call__(self,prompt,*args: argparse.Any, **kwds: argparse.Any) -> argparse.Any:
        output = self.llm.generate(prompt,max_tokens=args.max_tokens)
        return [{"generate_text":output}]
    
#  define a port like pipeline    
pipe = WrappedPipline(llm)

embed_RAG = True
tools = [ExampleTool()]

if not embed_RAG:
    agent =BaseAgent(pipe, tools=tools)
else:
    corpus =[
        "IVD 是体外诊断，使用血液样本进行分析。",
        "ISO 13485 是医疗设备质量管理体系标准。",
        "深度学习在病理图像分析中可实现自动诊断。"]
    retriever = SimpleRAGRetriever(corpus=corpus)
    agent =BaseAgent(pipe, tool=tools, retriever=retriever)

query = "help me to search the latest paper regarding IVD, and summarize the latest key points."
result = agent.run(query)
print("Agent Response:",result)
