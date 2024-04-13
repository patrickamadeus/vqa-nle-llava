# VQA Synthetic Data Generation with Large Multimodal Models

## Introduction
This project focuses on the creation of synthetic data tailored for Visual Question Answering (VQA) reasoning. Leveraging the capabilities of multiple Large Vision Language Models (LVLMs), the aim is to generate diverse and comprehensive datasets that can enhance the training and evaluation of VQA systems. By integrating both visual and textual modalities, this research thesis endeavors to advance the state-of-the-art in VQA through innovative data synthesis techniques.

## Setup
---
### LLaVa Environment
To ensure reproducibility and ease of setup, an environment management approach using Conda is adopted. The environment specifications are captured in `environment.yml`, facilitating the recreation of the working environment. Key steps include:

- Exporting the environment:
    ```
    conda env export > environment.yml
    ```

- Creating a new environment:
    ```
    conda create --name my_env python=x.xx
    ```

- Creating the environment from the file:
    ```
    conda env create -f environment.yml -n llava_env
    ```

- Installing IPython kernel for Jupyter notebook:
    ```
    conda install ipykernel # or pip install ipykernel
    python -m ipykernel install --user --name llava_env --display-name "Patrick (LLaVa)"
    ```

### Model Sources
Utilizing pre-trained LVLMs is integral to this project. The following models are employed:

- [LLaVa-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b)
- [LLaVa-7B-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [bakLLaVa-v1-hf](https://huggingface.co/llava-hf/bakLlava-v1-hf)
- [Mixtral](https://huggingface.co/cloudyu/Mixtral_7Bx4_MOE_24B)

## MiniGPT4
_(TO BE CONTINUED)_

## BLIP
_(TO BE CONTINUED)_

## Conclusion
This readme provides an overview of the project's objectives, setup, and key components. By harnessing the capabilities of LVLMs and innovative data generation techniques, the aim is to contribute to the advancement of VQA systems through the creation of high-quality synthetic datasets.
