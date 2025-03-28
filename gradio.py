import gradio as gr
import random
import string
from sft.data_aug import obfuscate_korean

def obfuscate_review(review, 
                     transform_hangul, 
                     add_random_jongseong, 
                     apply_liaison, 
                     cho_to_jong):

    
    if not review:
        return "리뷰를 입력해주세요."
    
    settings = {
        "transform_hangul": transform_hangul,
        "add_random_jongseong": add_random_jongseong,
        "apply_liaison": apply_liaison,
        "cho_to_jong": cho_to_jong
    }  
    
    # 난독화
    words = obfuscate_korean(review, settings)
    
    return words


with gr.Blocks() as demo:
    
    gr.Markdown("# 🏠숙소 리뷰 난독화 및 복원 ")
    
    with gr.Tabs():
        with gr.TabItem("리뷰 난독화"):
            with gr.Column():

                transform_hangul_slider = gr.Slider(
                    minimum=0, maximum=1, value=0.6, step=0.1, 
                    label="자모를 비슷한 발음으로 변환"
                )
                add_random_jongseong_slider = gr.Slider(
                    minimum=0, maximum=1, value=0.7, step=0.1, 
                    label="종성 랜덤 추가"
                )
                apply_liaison_slider = gr.Slider(
                    minimum=0, maximum=1, value=0.5, step=0.1, 
                    label="연음법칙 적용"
                )
                cho_to_jong_slider = gr.Slider(
                    minimum=0, maximum=1, value=0.6, step=0.1, 
                    label="초성을 종성으로 변환"
                )

                # 원본 리뷰 입력란
                gr.Markdown("# 원본 리뷰")
                obfuscation_input = gr.Textbox(lines=5)

                # 난독화 버튼
                obfuscation_btn = gr.Button("난독화")

                # 난독화된 리뷰 출력란
                gr.Markdown("# 난독화된 리뷰")
                obfuscation_output = gr.Textbox(lines=5)
    
            # 이벤트 핸들러
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

# 앱 실행
demo.launch(show_error=True)