import argparse
from llm_wrapper import LLMWrapper

parser  =argparse.ArgumentParser()
parser.add_argument("--checkpoint",type=str,requried=True)
args = parser.parse_args()

llm = LLMWrapper(model_path=args.model_path,
                 use_lora =False,
                 load_in_4bit=False)

query ="explain ISO 13485 standard in simple terms."

result = llm.generate(query,max_tokens=100)
print(result)

