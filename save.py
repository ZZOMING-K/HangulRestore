import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL =  "beomi/gemma-ko-7b"
ADAPTER_MODEL = "lora_adapter"

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL,device_map="cuda" , torch_dtype = torch.float16)
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL, device_map="cuda" , torch_dtype = torch.float16)
merged_model = model.merge_and_unload() #모델 병합 및 로드 
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL)

merged_model.push_to_hub("zzoming/Gemma-Ko-7B-SFT-AUG5") #huggingface로 push
tokenizer.push_to_hub("zzoming/Gemma-Ko-7B-SFT-AUG5")