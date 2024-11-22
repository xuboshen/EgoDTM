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


