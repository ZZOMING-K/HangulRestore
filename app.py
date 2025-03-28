import gradio as gr
import pandas as pd
from inference import KoreanLLMInference  # LLM 복원 모델 사용
from sft.data_aug import obfuscate_korean  # 난독화 모듈

# 난독화 함수
def obfuscate_review(review, transform_hangul, add_random_jongseong, apply_liaison, cho_to_jong):
    if not review:
        return "리뷰를 입력해주세요."
    
    settings = {
        "transform_hangul": transform_hangul,
        "add_random_jongseong": add_random_jongseong,
        "apply_liaison": apply_liaison,
        "cho_to_jong": cho_to_jong
    }  

    # 난독화 수행
    obfuscated_text = obfuscate_korean(review, settings)
    
    return obfuscated_text

# 복원 함수 (LLM + T5 직접 실행)
def restore_review(obfuscated_review):
    if not obfuscated_review:
        return "난독화된 리뷰를 입력해주세요."
    
    # 모델 로드
    inference_model = KoreanLLMInference()
    
    # LLM 실행
    generated_text = inference_model.inference(pd.DataFrame({"input": [obfuscated_review]}))[0]
    
    # T5 맞춤법 및 자연스러운 문장 수정
    restored_review = inference_model.correct_spelling(generated_text)
    
    return restored_review

# Gradio UI 생성
with gr.Blocks() as demo:
    
    gr.Markdown("# 🏠 숙소 리뷰 난독화 및 복원")

    with gr.Tabs():
        # 🔒 리뷰 난독화 탭
        with gr.TabItem("리뷰 난독화"):
            with gr.Column():
                gr.Markdown("### 원본 리뷰 입력")
                obfuscation_input = gr.Textbox(lines=5, placeholder="여기에 리뷰를 입력하세요.")

                # 난독화 설정 슬라이더
                transform_hangul_slider = gr.Slider(0, 1, value=0.6, step=0.1, label="자모를 비슷한 발음으로 변환")
                add_random_jongseong_slider = gr.Slider(0, 1, value=0.7, step=0.1, label="종성 랜덤 추가")
                apply_liaison_slider = gr.Slider(0, 1, value=0.5, step=0.1, label="연음법칙 적용")
                cho_to_jong_slider = gr.Slider(0, 1, value=0.6, step=0.1, label="초성을 종성으로 변환")

                # 난독화 버튼 및 출력
                obfuscation_btn = gr.Button("난독화")
                gr.Markdown("### 난독화된 리뷰 출력")
                obfuscation_output = gr.Textbox(lines=5, interactive=False)

                # 난독화 실행
                obfuscation_btn.click(
                    fn=obfuscate_review, 
                    inputs=[
                        obfuscation_input, 
                        transform_hangul_slider,
                        add_random_jongseong_slider,
                        apply_liaison_slider,
                        cho_to_jong_slider
                    ], 
                    outputs=obfuscation_output
                )

        # 🔄 리뷰 복원 탭
        with gr.TabItem("리뷰 복원"):
            with gr.Column():
                gr.Markdown("### 난독화된 리뷰 입력")
                restore_input = gr.Textbox(lines=5, placeholder="난독화된 리뷰를 입력하세요.")

                # 복원 버튼 및 출력
                restore_btn = gr.Button("복원")
                gr.Markdown("### 복원된 리뷰 출력")
                restore_output = gr.Textbox(lines=5, interactive=False)

                # 복원 실행
                restore_btn.click(fn=restore_review, inputs=restore_input, outputs=restore_output)

# 앱 실행
demo.launch(show_error=True)