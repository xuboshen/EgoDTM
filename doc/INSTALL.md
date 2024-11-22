# Installation

We use `python=3.10`, as well as `torch >= 2.3.1`, `torchvision>=0.18.1` and `cuda-12.1` in our environment.

```bash
conda create -n instancehoi python=3.10
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
```

Install detectron
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Install linters
```bash
python -m pip install pre-commit
pre-commit install
```


For AVION, install necessary requirements from https://github.com/zhaoyue-zephyrus/AVION/blob/main/docs/INSTALL.md
```
pip install ninja==1.11.1
pip install kornia==0.6.10
pip install pytorchvideo==0.1.5
pip install submitit==1.4.5
pip install timm==0.6.12
pip install git+https://github.com/openai/CLIP.git
```


