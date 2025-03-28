from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd 

class T5Dataset(Dataset):
    
    def __init__(self, data, tokenizer, max_length=1500):
        
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        input_text = self.data['restore_review'][idx]
        target_text = self.data['output'][idx]

        inputs = self.tokenizer(input_text, max_length=self.max_length, truncation=True, padding=True, return_tensors='pt')
        labels = self.tokenizer(target_text, max_length=self.max_length, truncation=True, padding=True, return_tensors='pt')

        item = {key: inputs[key].squeeze(0) for key in inputs.keys()}
        item["labels"] = labels["input_ids"].squeeze(0)

        return item
    
def split_txt(input_text) : # .을 기준으로 분리 
    
    text = input_text.split('. ')
    
    text_li = []
    
    for txt in text :
        txt = txt + '.'
        text_li.append(txt)
    
    return text_li

def create_T5_dataset(data_path, tokenizer, test_size=0.005, random_state=42):
    
    train = pd.read_csv(data_path)
    
    train['restore_review'] = train['restore_review'].str.strip().apply(split_txt) 
    train['output'] = train['output'].str.strip().apply(split_txt)

    train['restore_len'] = train['restore_review'].apply(len)
    train['output_len'] = train['output'].apply(len)
    train = train[train['restore_len'] == train['output_len']]

    train = train[['restore_review' , 'output']].explode(column=['restore_review', 'output']).reset_index(drop = True)
    train  = train[train['restore_review'] != train['output']].reset_index(drop = True)

    # restore 과 정답(output)이 일치할 경우에는 제거 
    train = train[train['restore_review'] != train['output']].reset_index(drop = True)

    # instruct 추가 
    train['restore_review'] = "어색한 표현 및 맞춤법을 교정해주세요: " + train['restore_review']
    
    train_data, valid_data = train_test_split(train, 
                                              test_size=test_size, 
                                              random_state=random_state )
    
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)    
    
    train_dataset = T5Dataset(train_data, tokenizer)
    eval_dataset = T5Dataset(valid_data, tokenizer)
    
    return train_dataset, eval_dataset
