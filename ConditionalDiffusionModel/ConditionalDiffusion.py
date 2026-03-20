from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import gc
import random

import numpy as np
import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image


@dataclass(frozen=True)
class AugmentConfig:

    name: str
    strength: float
    guidance_scale: float
    num_inference_steps: int


class ConditionalDiffusion:

    DEFAULT_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    DEFAULT_EXCLUDE_TAGS = {"_mild_color_shift", "_texture_variation", "_detail_enhanced"}

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        # base prompt text를 vectorize 한 값을 이미지와 함께 넣어 해당 값과 비슷하게 가우시안 노이즈를 제거한다. (모델이 도달해야 할 목표 지점의 벡터)
        base_prompt: str = (
            "cold color palette, muted colors, detailed, 8k, "
            "high detail, number focus, texture variation, 8k number, another number"
        ),
        # negative prompt text를 vectorize한 값을 함께 넣어 해당 값과 반대되게 가우시안 노이즈를 제거한다. (모델이 피해야 할 기피 지점의 벡터)
        negative_prompt: str = (
            "nsfw, nude, low quality, blurry, noisy, grainy, "
            "artifacts, distorted, cartoon, painting, text, watermark"
        ),
        retry_prompt: str = "detailed, 8k, detailed texture",
        retry_strength_delta: float = 0.08,
        retry_guidance_delta: float = 1.0,
        retry_steps_delta: int = 6,
        retry_min_strength: float = 0.22,
        retry_min_guidance: float = 5.5,
        retry_min_steps: int = 28,
        target_long_side: int = 768,
        black_threshold: int = 8,
        black_ratio: float = 0.98,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.base_prompt = base_prompt
        self.negative_prompt = negative_prompt
        self.retry_prompt = retry_prompt

        self.retry_strength_delta = retry_strength_delta
        self.retry_guidance_delta = retry_guidance_delta
        self.retry_steps_delta = retry_steps_delta
        self.retry_min_strength = retry_min_strength
        self.retry_min_guidance = retry_min_guidance
        self.retry_min_steps = retry_min_steps

        self.target_long_side = target_long_side
        self.black_threshold = black_threshold
        self.black_ratio = black_ratio
        self.seed = seed

        self.device = device or self._detect_device()
        self.torch_dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32

        self._set_seed(self.seed)

        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.enable_attention_slicing()
        self.pipe = self.pipe.to(self.device)

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)

    def resize_to_multiple_of_8(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        scale = self.target_long_side / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        nw = max(512, (nw // 8) * 8)
        nh = max(512, (nh // 8) * 8)
        return image.resize((nw, nh), Image.Resampling.LANCZOS)

    def is_mostly_black(self, image: Image.Image) -> bool:
        arr = np.array(image.convert("RGB"), dtype=np.uint8)
        dark = np.all(arr <= self.black_threshold, axis=-1)
        return float(dark.mean()) >= self.black_ratio

    def collect_image_files(
        self,
        input_dir: str | Path,
        image_exts: Iterable[str] | None = None,
        exclude_tags: Iterable[str] | None = None,
    ) -> list[Path]:
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"입력 폴더가 존재하지 않습니다: {input_path}")

        exts = {e.lower() for e in (image_exts or self.DEFAULT_IMAGE_EXTS)}
        tags = set(exclude_tags or self.DEFAULT_EXCLUDE_TAGS)

        image_files = [
            p
            for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in exts and not any(tag in p.stem for tag in tags)
        ]

        if not image_files:
            raise FileNotFoundError(f"입력 이미지가 없습니다: {input_path}")

        return sorted(image_files)

    def _generator(self, seed: int) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(seed)

    def _infer_one(self, init_image: Image.Image, cfg: AugmentConfig, seed: int) -> Image.Image:
        out = self.pipe(
            prompt=self.base_prompt,
            negative_prompt=self.negative_prompt,
            image=init_image,
            strength=cfg.strength,
            guidance_scale=cfg.guidance_scale,
            num_inference_steps=cfg.num_inference_steps,
            generator=self._generator(seed),
        ).images[0]

        if self.is_mostly_black(out):
            out = self.pipe(
                prompt=self.retry_prompt,
                negative_prompt=self.negative_prompt,
                image=init_image,
                strength=max(self.retry_min_strength, cfg.strength - self.retry_strength_delta),
                guidance_scale=max(self.retry_min_guidance, cfg.guidance_scale - self.retry_guidance_delta),
                num_inference_steps=max(self.retry_min_steps, cfg.num_inference_steps - self.retry_steps_delta),
                generator=self._generator(seed + 10_000),
            ).images[0]

        return out

    def augment_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path | None = None,
        augment_configs: list[AugmentConfig] | None = None,
        image_exts: Iterable[str] | None = None,
        exclude_tags: Iterable[str] | None = None,
        verbose: bool = True,
    ) -> list[Path]:

        input_path = Path(input_dir)
        output_path = Path(output_dir) if output_dir is not None else input_path
        output_path.mkdir(parents=True, exist_ok=True)

        configs = augment_configs or [
            AugmentConfig("mild_color_shift", 0.30, 7.0, 32),
            AugmentConfig("texture_variation", 0.38, 7.2, 36),
            AugmentConfig("detail_enhanced", 0.44, 7.5, 40),
        ]

        image_files = self.collect_image_files(
            input_dir=input_path,
            image_exts=image_exts,
            exclude_tags=exclude_tags,
        )

        if verbose:
            print(f"입력 폴더: {input_path}")
            print(f"출력 폴더: {output_path}")
            print(f"대상 이미지 수: {len(image_files)}")
            print(f"사용 디바이스: {self.device}")

        saved_paths: list[Path] = []

        for idx, img_path in enumerate(image_files):
            init_image = Image.open(img_path).convert("RGB")
            init_image = self.resize_to_multiple_of_8(init_image)

            for i, cfg in enumerate(configs):
                run_seed = self.seed + idx * 100 + i
                out = self._infer_one(init_image, cfg, run_seed)

                save_name = f"{img_path.stem}_{cfg.name}{img_path.suffix.lower()}"
                save_path = output_path / save_name
                out.save(save_path)
                saved_paths.append(save_path)

                if verbose:
                    print(f"[{idx + 1}/{len(image_files)}] 저장 완료: {save_path}")

        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        if verbose:
            print("\n완료: 증강 저장 위치")
            print(output_path)

        return saved_paths
