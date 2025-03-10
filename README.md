# HangulRestore 

- 흔히 **에어비앤비체** 로 불리는 난독화된 한글 리뷰를 원본으로 복원하는 모델 개발 
- 🙌 [Hugging Face 모델 공개](https://huggingface.co/zzoming/Gemma-Ko-7B-SFT-AUG5)
- 🫠 [난독화된 한글 리뷰 복원 모델 개발 회고](https://until.blog/@zzoming/-dacon--%EB%82%9C%EB%8F%85%ED%99%94%EB%90%9C-%ED%95%9C%EA%B8%80-%EB%A6%AC%EB%B7%B0-%EB%B3%B5%EC%9B%90-ai-%EB%AA%A8%EB%8D%B8-%EA%B0%9C%EB%B0%9C%EA%B8%B0)
<br>

# 진행기간 및 성과

- 2025.01 ~ 2025.02 ( 약 1달간 진행 )
- 개인참가
- **상위 10% 내 등수 기록** (22등 / 291팀)
<br>

# 개발 과정 

## Data Augmentation
- 한글 난독화 패턴을 반영하여 데이터 증강
- Train 데이터 1만 개 → 5만 개 확장
- **데이터 5만 개 (1 epoch) 학습 시 좋은 성능** 기록
<br>

## Supervised Fine-Tuning (SFT)
- **Instruction Tuning**: 모델이 명령을 잘 따르도록 학습
- **LoRA 적용**: 메모리 절약 및 효율적 Fine-Tuning

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
```
<br>

## Inference
- **vLLM 활용**하여 Transformer 대비 속도 향상
- **Sampling 기법**을을 활용하여 추론

```python
sampling_params = SamplingParams(
    temperature=0.2, 
    top_p=0.9, 
    top_k=20, 
    seed=42, 
    max_tokens=2048, 
    stop_token_ids=[eos_token_id]
)
```
<br>

---

