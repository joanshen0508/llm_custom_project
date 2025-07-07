from transformers import AutoModelForCausualLM, AutoTokenizer,BitsAndBytesConfig
import torch

"""if you are a newer for using llm, you can directely use transformers.pipeline for inference, and get a result.
    it is diffcult to embed lora, qlora, mixing precision, quantized inference, many instances, etc using transformers.pipline.
    so if you want to do aboved operation, you can edit your own llm.
"""

class LLMWrapper:
    def __init__(self, model_path, device="cuda",
                 use_lora=False,lora_path=None,
                 quantize=False,
                 load_in_4bit=False, load_in_8bit=False,
                 use_fp16=True,
                 ):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if device is None:
            if torch.cuda.is_available():
                self.device ="cuda"
            elif torch.backends.mps.is_available():
                self.device ="mps"
            else:
                self.device ="cpu"
        else:
            self.device = device
        print(f"[LLMWrapper] using device: {self.device}")

        dtype = torch.float16 if use_fp16 else torch.float32
        if quantize or load_in_4bit or load_in_8bit:
            bnb_config =  BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                llm_int8_has_fp16_weight = use_fp16,
                bnb_4bit_compute_dtype=torch.float16 if use_fp16 else torch.float32,
                bnb_4bit_use_double_quant =True,
                bnb_4bit_quant_type = "nf4",
            )
            model = AutoModelForCausualLM.from_pretrained(model_path,
                                                        quantization_config =bnb_config,
                                                        device_map="auto"
                                                        )
        else:
            model = AutoModelForCausualLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto"
            )

        if use_lora and lora_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model,lora_path)
        self.model = model.eval()
        self.model.to(self.device)
    
    def generate(self, prompt, max_tokens=128):
        inputs = self.tokenizer(prompt,return_tensors="pt").input_ids.to(self.deivce)
        with torch.no_grad():
            output = self.model.generate(**inputs,
                                         max_new_tokens =max_tokens,
                                         do_sample = True,
                                         temperature= 0.7,
                                         top_p=0.9)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)