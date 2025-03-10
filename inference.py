import vllm
from vllm import LLM, SamplingParams
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import time

def create_datasets(df):

    def preprocess(samples):

        batch = []
        PROMPT_DICT = {

        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n"
            "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
            "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
            "### Instruction(명령어):{instruction}\n\n### Input(입력):{input}\n\n### Response(응답):"
        )
    }

        for instruction, input in zip(samples["instruction"], samples["input"]):
            user_input = input
            conversation = PROMPT_DICT['prompt_input'].replace('{instruction}', instruction[0]).replace('{input}', user_input)
            batch.append(conversation)

        return {"text": batch}

    def generate_dict(df) :

        instruction_list = [ [open('./data/instruction.txt').read()] for _ in range(len(df)) ]
        input_list = df['input']
        dataset_dict = {'instruction' : instruction_list , 'input' : input_list}
        dataset = Dataset.from_dict(dataset_dict)

        return dataset

    datasets = generate_dict(df)
    raw_datasets = DatasetDict()

    raw_datasets['test'] = datasets

    raw_datasets = raw_datasets.map(
        preprocess,
        batched = True,
        remove_columns=['instruction']
    )

    test_data = raw_datasets["test"]

    print(f"Size of the test set: {len(test_data)}")
    print(f"A sample of test dataset: {test_data[0]}")

    return test_data


model_id ="zzoming/Gemma-Ko-7B-SFT-AUG5"

llm = LLM(
    model = model_id ,
    trust_remote_code = True ,
    quantization="bitsandbytes",
    load_format="bitsandbytes"
)

df = pd.read_csv('./data/test.csv')
test_data = create_datasets(df)

tokenizer = AutoTokenizer.from_pretrained(model_id) # 토크나이저 로드
eos_token_id = tokenizer.eos_token_id # EOS 토큰 ID 가져오기

restore_reviews = []

llm.llm_engine.scheduler_config.max_num_seqs = 128 # 배치 사이즈 설정정

sampling_params = SamplingParams(temperature = 0.2 , top_p = 0.9, top_k = 20, seed = 42 , max_tokens = 2048 , stop_token_ids = [eos_token_id])
outputs = llm.generate(test_data['text'], sampling_params)

for output in outputs :
    generated_text = output.outputs[0].text
    print(generated_text)
    restore_reviews.append(generated_text)

df_submission = pd.read_csv('./data/sample_submission.csv') 
df_submission['output'] = restore_reviews
df_submission.to_csv('gemma-ko-SFT-5M.csv' , index = False)