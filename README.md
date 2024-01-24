# VQA Synthetic Data Generation with Large Multimodal Models

## Setup
---
### LLaVa
- Export environment
    ```
    conda env export > environment.yml
    ```
- Create environment
    ```
    conda create --name my_env python=x.xx
    ```
    NB : ALWAYS SPECIFY PYTHON VERSION TO CREATE `PIP` AND TO AVOID GLOBAL PACKAGE INSTALLMENT
- Create environment from file
    ```
    conda env create -f environment.yml -n llava_env
    ```

- Create `ipykernel`
    ```
    conda install ipykernel # or pip install ipykernel
    python -m ipykernel install --user --name llava_env --display-name "Patrick (LLaVa)"
    ```
- Model Sources
    - [LLaVa-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b)
    - [LLaVa-7B-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
    - [bakLLaVa-v1-hf](https://huggingface.co/llava-hf/bakLlava-v1-hf)
---
### MiniGPT4
_(TO BE CONTINUED)_

---
### BLIP
_(TO BE CONTINUED)_
