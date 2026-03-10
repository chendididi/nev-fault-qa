"""
Qwen2-VL-7B-Instruct 推理客户端

封装本地 Qwen2-VL 模型的文本生成和图像理解能力。
支持：
- 纯文本对话
- 图像 + 文本多模态输入（电路图 OCR）
- 流式输出（SSE 场景）
"""

import base64
from pathlib import Path
from typing import Generator

import torch
from loguru import logger
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


class QwenClient:
    """Qwen2-VL-7B-Instruct 本地推理封装。"""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda:0",
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        do_sample: bool = False,
    ):
        """
        Args:
            model_path: 模型路径或 HuggingFace Hub ID
            device: 推理设备（RTX 4090 用 cuda:0）
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度（do_sample=True 时生效）
            do_sample: 是否采样（False 为贪心解码）
        """
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._do_sample = do_sample

        logger.info(f"加载 Qwen2-VL 模型: {model_path}")
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
        )
        self._processor = AutoProcessor.from_pretrained(model_path)
        logger.info("Qwen2-VL 模型加载完成")

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        纯文本生成。

        Args:
            prompt: 用户输入
            system_prompt: 系统提示词（可选）

        Returns:
            生成的文本字符串
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], return_tensors="pt").to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=self._do_sample,
                temperature=self._temperature if self._do_sample else None,
            )

        # 只取新生成的 token
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        return self._processor.decode(generated, skip_special_tokens=True)

    def describe_image(self, image_bytes: bytes, prompt: str) -> str:
        """
        多模态：根据图像字节和文本 prompt 生成描述。

        Args:
            image_bytes: PNG/JPEG 图像字节
            prompt: 文本指令

        Returns:
            模型生成的图像描述
        """
        from PIL import Image
        import io

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )

        generated = output_ids[0][inputs.input_ids.shape[1]:]
        return self._processor.decode(generated, skip_special_tokens=True)

    def stream_generate(
        self, prompt: str, system_prompt: str | None = None
    ) -> Generator[str, None, None]:
        """
        流式文本生成（适合 SSE 场景）。

        使用 TextIteratorStreamer 逐 token yield。

        Args:
            prompt: 用户输入
            system_prompt: 系统提示词

        Yields:
            逐步生成的文本片段
        """
        import threading
        from transformers import TextIteratorStreamer

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], return_tensors="pt").to(self._device)

        streamer = TextIteratorStreamer(
            self._processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": self._max_new_tokens,
            "do_sample": self._do_sample,
        }

        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        for token in streamer:
            yield token

        thread.join()
