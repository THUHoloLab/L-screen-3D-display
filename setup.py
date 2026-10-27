"""L-Splat: Real-time monocular-to-anamorphic 3DGS engine for L-shaped 90-degree
outward-fold glasses-free 3D LED displays.

Install in editable mode:
    pip install -e .

After install, run:
    python -m l_splat.core.pipeline --image ~/Desktop/hypercar.jpg
"""

from setuptools import setup, find_packages

setup(
    name="l_splat",
    version="0.3.0",
    description="Real-Time Monocular-to-Anamorphic 3DGS Engine for L-Shaped Glasses-Free 3D Displays",
    author="L-Splat Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1",
        "torchvision>=0.16",
        "numpy<2",
        "opencv-contrib-python>=4.8",
        "pillow>=10.0",
        "tqdm>=4.65",
        "matplotlib>=3.7",
        "transformers>=4.40",
        "huggingface-hub>=0.20",
        "omegaconf>=2.3",
        "einops>=0.7",
        "safetensors>=0.4",
        "plyfile>=1.0",
    ],
)
