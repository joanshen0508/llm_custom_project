import argparse
from transformers import (AutoTokenizer,
                         AutoModelForCausalLM,
                         TrainingArguments,
                         Trainer,
                         DataCollatorForLanguageModeling)
from datasets import load_dataset
from peft import (LoraConfig,
                  PromptTuningConfig,
                  PrefixTuningConfig,
                  AdapterConfig,
                  TaskType,
                  get_peft_model)
from custom_modules.custom_attention import CustomLlamaAttention

"""how to use train
    custom attention +lora _int8: --peft_method lora --use_custom_attention
    use prompt tuning --peft_method prompt_tuning
    use prefix tuning --peft_method prefix_tuning
    use adapter --peft_method adapter
    use bitgit  --peft_method bitfit
    only use sft --peft_method none
    inferece --checkpoint ./checkpoints/checkpoint-xxx
"""
# 1.analytic the parameters

parser = argparse.ArgumentParser()
parser.add_argument("--peft_method",type=str,default="none",choices=["none","lora","prompt_tuning","prefix_tuning","adapter","bitfit"],help="PEFT method to user")
parser.add_argument("--user_custom_attention",type=bool,default=True,help="use custom attention with lora/int8")
parser.add_argument("--use_int8",type=bool,default=False,help="use int8")
parser.add_argument("--use_lora",type=bool,default=True,help="use lora")
parser.add_argument("--lora_r",type=int,default=8,help="lora rank")
parser.add_argument("--lora_alpha",type=int,default=16,help="lora alpha")
parser.add_argument("--lora_dropout",type=float,default=0.05,help="lora dropout")
parser.add_argument("--lora_bias",type=str,default="none",choices=["none","all","lora_only"],help="lora bias")
args = parser.parse_args()
print(f"PEFT: {args.peft_method}, CustomAttention: {args.user_custom_attention}")

# 2.load the model

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto")


# 3.if chooese custom attention, replace the attention layer

if args.user_custom_attention:
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    for layer in model.model.layers:
        layer.self_attn = CustomLlamaAttention(layer.self_attn.config,
                                               use_lora=args.use_lora,
                                               use_int8=args.use_int8)
        
# 4. if choose PEFT, add PEFT layer
peft_config =None

if args.peft_method =="lora":
    peft_config =LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules = ["q_proj","v_proj"],
        lora_dropout = 0.05,
        bias="none",
        task_type = TaskType.CAUSAL_LM
    )
elif args.peft_method =="prompt_tuning":
    peft_config = PromptTuningConfig(
        task_type = TaskType.CAUSAL_LM,
        num_virtual_tokens = 10,
        prefix_projection = True
    )
elif args.peft_method == "prefix_tuning":
    peft_config = PrefixTuningConfig(
        task_type = TaskType.CAUSAL_LM,
        num_virtual_tokens =10,
        prefix_projection=True
    )
elif args.peft_method =="adapter":
    peft_config = AdapterConfig(
        task_type=TaskType.CAUSAL_LM,
        adapter_dropout=0.1)
    
elif args.peft_method == "bitfit":
    # bitfit" only train bias
    for name, param in model.named_parameters():
        param.requries_grad ="bias" in name

if peft_config is not None:
    from peft import get_peft_model
    # add peft layer
    mdoel = get_peft_model(model,peft_config)
else:
    # add none and run pure sft
    print("run pure sft")

# 5. data + trainer
dataset = load_dataset("json",data_files="data/train.jsonl")["train"]
collator = DataCollatorForLanguageModeling(tokenizer,mlm=False)

train_args = TrainingArguments(
    outout_dir ="./checkpoints",
    per_device_train_batch_size = 1,
    gradient_accumulation_steps =4,
    num_train_epochs=1,
    loggin_steps=10,
    save_steps=50,
    learning_rate =2e-4,
    fp16=True,
)

trainer = Trainer(
    model =model,
    args = train_args,
    train_dataset = dataset,
    data_collator = collator
)

trainer.train()