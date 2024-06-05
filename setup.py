from setuptools import setup, find_packages

setup(
    name="ChatUniVi",
    version="0.1.0",
    description="Towards GPT-4 like large language and visual assistant.",
    author="syw",
    author_email="syw95@mail.ustc.edu.cn.com",
    packages=find_packages(),
    install_requires=[
        "einops", "fastapi", "gradio==3.35.2", "markdown2[all]", "numpy",
        "requests", "sentencepiece", "tokenizers>=0.12.1",
        "torch", "torchvision", "uvicorn", "wandb",
        "shortuuid", "httpx==0.24.0",
        "deepspeed==0.9.5",
        "peft==0.4.0",
        "transformers==4.31.0",
        "accelerate==0.21.0",
        "bitsandbytes==0.41.0",
        "scikit-learn==1.2.2",
        "sentencepiece==0.1.99",
        "einops==0.6.1", "einops-exts==0.0.4", "timm==0.6.13",
        "gradio_client==0.2.9",
        "gradio==3.35.2",
        "imageio==2.31.3",
        "imageio-ffmpeg==0.4.9",
        "openai",
        "timm",
        "decord==0.6.0",
        "jsonlines"
    ],
)