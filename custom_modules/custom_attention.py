from transformers.models.llama.modeling_llama import LlamaAttention
from peft.tunner.lora import LoraLayer
import bitsandbytes as bnb
import torch

# inheriting from offical llamaattention class
# custom Lora
# only use lora on q_proj
# support mixed precision training

class CustomLlamaAttention(LlamaAttention):
    # def __init__(self,config, user_lora=True, use_int8=False, **kwargs):
    def __init__(self,config, use_lora=True, use_int8=False):
        super().__init__(config)
        self.config = config
        self.use_lora = use_lora
        self.use_int8 = use_int8
        #  original q_proj, k_proj, v_proj, are in father class
        if self.use_lora:
            #  add lora to q_proj
            self.q_proj = LoraLayer(config.hidden_size,self.config.hidden_size, r=8)
        if self.use_int8:
            #  add int8 to q_proj
            self.q_proj = bnb.nn.Linear8biLt(self.q_proj.in_features,self.q_proj.out_features,bias=True
                                             )
    
    def forward(self, *args, **kwargs):
        # you can add hook here and print hidden_states
        return super().forward(*args, **kwargs)
        

        