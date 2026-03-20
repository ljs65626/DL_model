# Conditional Diffusion Model을 활용한 Image2Image 데이터 증강 모델

<p align="center">
  <img align="middle" src="https://github.com/user-attachments/assets/e86fd578-3597-44a8-80d3-75edbcb76acc" width="40%" />
  <img align="middle" src="https://raw.githubusercontent.com/googlefonts/noto-emoji/main/svg/emoji_u27a1.svg" width="60" hspace="30" />
  <img align="middle" src="https://github.com/user-attachments/assets/409657e4-5f4d-4eb2-a1e5-0f7f8fa00e1f" width="40%" />
</p>
<br>
사용법: <br>
from image2image_diffusion import ConditionalDiffusion, AugmentConfig

input_dir = ""
output_dir = input_dir

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
