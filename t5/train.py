from transformers import T5TokenizerFast, T5ForConditionalGeneration
from dotenv import load_dotenv
import os
import wandb
from huggingface_hub import login
import yaml
from transformers import TrainingArguments, Trainer
from utils import create_T5_dataset

def initialize_env() :
    
    load_dotenv()
    
    wandb_api_key = os.getenv("WANDB_API_KEY")
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    wandb.login(key=wandb_api_key) # wandb login
    
    login(hf_api_key) # huggingface login

class KoreanT5Trainer : 
    
    def __init__(self, config_path : str = '../config/t5_train.yaml') : 
        
        with open(config_path , 'r') as file :
            self.config = yaml.safe_load(file)
            
        self.train_dataset , self.eval_dataset = self._load_dataset()
        self.model , self.tokenizer = self._load_model()
        
        # 학습 인자 설정
        self.training_args = self._configure_training_args()
    
    def _load_dataset(self) :
        data_path = self.config['data']['path'] 
        train_dataset, eval_dataset = create_T5_dataset(data_path, self.tokenizer)
        return train_dataset , eval_dataset
    
    def _load_model(self) :
        model_name = self.config['training']['base_model']
        tokenizer = T5TokenizerFast.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    
        return model, tokenizer
    
    def _configure_training_args(self) :
        
        train_config = self.config['training'] 
        
        return TrainingArguments(
            output_dir = train_config['output_dir'] ,
            eval_strategy  = train_config['eval_strategy'] , 
            eval_steps = train_config['eval_steps'],
            per_device_train_batch_size=train_config['batch_size'],
            per_device_eval_batch_size=train_config['batch_size'],
            gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
            num_train_epochs=train_config['epochs'],
            lr_scheduler_type=train_config['lr_scheduler_type'],
            learning_rate= float(train_config['learning_rate']),
            warmup_ratio=train_config['warmup_ratio'],
            logging_strategy=train_config['logging_strategy'],
            logging_steps=train_config['logging_steps'],
            save_strategy=train_config['save_strategy'],
            seed = train_config['seed'],
            run_name = f"ko-bart-{train_config['epochs']}-FT",
            optim = train_config['optimizer'],
            report_to = "wandb",
            push_to_hub = True
        )
        
    def train(self) : 
        
        trainer = Trainer(
            model = self.model , 
            args = self.training_args,
            train_dataset = self.train_dataset,
            eval_dataset = self.eval_dataset,
            tokenizer = self.tokenizer
        )
        
        # train
        trainer.train()
        
        # save model 
        t5_dir = self.config['training']['t5_dir']
        trainer.model.save_pretrained(t5_dir)
        self.tokenizer.save_pretrained(t5_dir)
        
def main() : 
    initialize_env()
    trainer = KoreanT5Trainer()
    trainer.train()
    
if __name__ == "__main__" :
    main()
    
