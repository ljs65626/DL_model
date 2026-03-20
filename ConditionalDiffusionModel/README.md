# Conditional Diffusion Model을 활용한 Image2Image 데이터 증강 모델

<p align="center">
  <img align="middle" src="https://github.com/user-attachments/assets/e86fd578-3597-44a8-80d3-75edbcb76acc" width="40%" />
  <img align="middle" src="https://raw.githubusercontent.com/googlefonts/noto-emoji/main/svg/emoji_u27a1.svg" width="60" hspace="30" />
  <img align="middle" src="https://github.com/user-attachments/assets/409657e4-5f4d-4eb2-a1e5-0f7f8fa00e1f" width="40%" />
</p>


```python
from image2image_diffusion import ConditionalDiffusion, AugmentConfig

input_dir = "./data/original_images"
output_dir = input_dir

# strength: 가우시안 노이즈를 얼마나 추가할 것인가?(0.0 ~ 1.0) -> 값이 낮을 수록 원본과 거의 비슷하게 나오고, 높을 수록 원본과 많이 다르게 나온다.
# guidance_scale: 사용자가 입력한 프롬프트를 얼마나 엄격하게 따를 것인가? -> 값이 낮을 수록 프롬프트를 무시하고, 높을 수록 프롬프트를 엄격하게 따른다.
# num_inference_steps: 가우시안 노이즈 제거 과정을 몇 번 반복할 것인가? -> 값이 낮을 수록 노이즈를 적게 제거하고, 높을 수록 노이즈를 많이 제거한다.
augment_configs = [
    AugmentConfig("name1", strength=0.30, guidance_scale=7.0, num_inference_steps=32),
    AugmentConfig("name2", strength=0.38, guidance_scale=7.2, num_inference_steps=36),
    AugmentConfig("name3", strength=0.44, guidance_scale=7.5, num_inference_steps=40),
]

augmentor = ConditionalDiffusion(
    model_id="runwayml/stable-diffusion-v1-5",
    seed=42,
)

saved_paths = augmentor.augment_directory(
    input_dir=input_dir,
    output_dir=output_dir,
    augment_configs=augment_configs,
    verbose=True,
)

print(f"총 생성 파일 수: {len(saved_paths)}")
```
