# GPT-2 

This repo follows the receipes from Andrej Karpathy in reproducing a minimal GPT-2 (124M) pretraining setting, see his [video](https://www.youtube.com/watch?v=l8pRSuU81PU&ab_channel=AndrejKarpathy) and [codebase](https://github.com/karpathy/build-nanogpt).


## Installation 

```bash
conda create -n gpt2 python=3.10
conda activate gpt2 

conda install ipykernel
python -m ipykernel install --user --name=gpt2

# torch using 12.4, my nvidia driver is 550.120
pip install torch torchvision torchaudio
pip install transformers datasets tiktoken matplotlib 

# (optional) take conda env snapshot 
conda env export > environment.yml
# (optional) create conda env from snapshot 
conda env create -f environment.yml 
```


# Run

Download the datasets 
```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python fineweb.py 
python hellaswag.py 
```

run gpt-2 training script 
```bash
# simple launch:
python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```


