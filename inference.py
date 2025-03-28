import torch
import yaml
import pandas as pd
import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from transformers import T5TokenizerFast, T5ForConditionalGeneration, pipeline

class KoreanLLMInference:
    
    def __init__(self, config_path='./config/inference.yaml'):
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
        self.llm_model = self._load_llm_model()
        self.t5_model, self.t5_tokenizer = self._load_t5_model()
        self.tokenizer = self._load_eos_tokenizer()
        
        # 프롬프트 템플릿 정의
        self.prompt_input = (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n"
            "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
            "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
            "### Instruction(명령어):{instruction}\n\n### Input(입력):{input}\n\n### Response(응답):"
        )
    
    def _load_llm_model(self):

        model_id = self.config['model']['base_model']
        
        llm = LLM(
            model=model_id, 
            dtype=torch.bfloat16, 
            trust_remote_code=True, 
            quantization="bitsandbytes", 
            max_model_len=4096, 
            gpu_memory_utilization=0.8
        )
        return llm
    
    def _load_t5_model(self) : 
        model_id = self.config['model']['t5_model']
        model = T5ForConditionalGeneration.from_pretrained(model_id)
        tokenizer = T5TokenizerFast.from_pretrained(model_id)
        return model, tokenizer 
    
    def _load_eos_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config['model']['base_model'])
        return tokenizer
    
    def split_txt(self, input_text):
        # 텍스트를 마침표를 기준으로 분리
        text = input_text.split('. ')
        
        text_li = []
        for txt in text:
            txt = txt + '.'
            text_li.append(txt)
        
        return text_li
    
    def correct_spelling(self, text):
        
        spelling_pipe = pipeline(
            "text2text-generation",
            model=self.t5_model,
            tokenizer=self.t5_tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            framework="pt"
        )

        input_text = f"어색한 표현 및 맞춤법을 교정해주세요: {text}"
      
        # 파이프라인으로 교정
        output_text = spelling_pipe(input_text, 
                                    max_length=512,
                                    do_sample=False)[0]
        
        # 생성된 텍스트 반환
        corrected_text = output_text['generated_text']
    
    
        return corrected_text
    
    
    def generate_dict(self, df):
        
        with open('./data/instruction.txt', 'r') as f:
            instruction = f.read()
     
        instruction_list = [instruction for _ in range(len(df))]
        
        # 새로운 데이터프레임 생성
        dataset_dict = {
            'instruction': instruction_list, 
            'input': df['input']
        }
        
        return pd.DataFrame(dataset_dict)
    
    def inference(self, df):
        
        restore_reviews = []
        
        # 각 입력에 대해 처리
        for i, input_text in enumerate(df['input']):
            
            # 입력 텍스트 분리
            split_text = self.split_txt(input_text)
            
            # 생성된 문장 저장 리스트
            generated_sentences = []
            
            for txt in split_text:
                # 프롬프트 생성
                instruction = df['instruction'][i]
                prompt = self.prompt_input.replace('{instruction}', instruction).replace('{input}', txt)
                
                # 모델로 생성
                output = self.llm_model.generate(prompt, 
                                            sampling_params = SamplingParams(
                                                temperature=self.config['inference']['temperature'], 
                                                top_p=self.config['inference']['top_p'], 
                                                top_k=self.config['inference']['top_k'], 
                                                seed=self.config['inference']['seed'], 
                                                max_tokens=len(txt), 
                                                stop_token_ids=[self.tokenizer.eos_token_id]))
                
                generated_text = output[0].outputs[0].text.strip()
                corrected_text = self.correct_spelling(generated_text)
                
                generated_sentences.append(corrected_text)
                print(generated_text)
            
            # 생성된 문장들 결합
            restored_text = " ".join(generated_sentences)
            restore_reviews.append(restored_text)
            print(restored_text)
        
        return restore_reviews
    
    def save_dataset(self, df, restore_reviews, output_path):
        df['restore_review'] = restore_reviews
        df.to_csv(output_path, index=False)

def main():
    
    df = pd.read_csv('./data/test.csv')
    
    inference_model = KoreanLLMInference()
    
    processed_df = inference_model.generate_dict(df)
    
    restore_reviews = inference_model.inference(processed_df)
    
    inference_model.save_dataset(processed_df, restore_reviews, './data/test_inference.csv')

if __name__ == "__main__":
    main()