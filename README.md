# Synthetic VQA with NLE Data Generation via LVLM

## Introduction
This project focuses on creating synthetic VQA (Visual Question Answering) with Explanation data. By leveraging the capabilities of LLaVA model, the aim is to generate diverse and comprehensive data samples that are able mimic the data created by humans.

## Setup
```bash
$ cd src/scripts
$ source setup.sh
```
**NB: Any confirmation prompt(s) may be present (e.g. `[Y/n]` confirmation)**

## Hyperparameters Configuration

This `.yml` file defines the key parameters for controlling experiment setups, including dataset details, model configurations, and inference behaviors. The results for each experiment will be saved under `/result/{test_name}`.

### General Parameters

- **`test_name`** (string):  
  The name of the dataset being used. Results will be stored in the `/result/{test_name}` directory.
  
- **`seed`** (int):  
  The random seed for the experiment, used for reproducibility.

### Dataset Parameters

- **`image_count`** (int, must be > 0):  
  Defines the number of images to generate during the experiment.
  
- **`use_scene_graph`** (bool: `0/1`):  
  Flag for whether to incorporate scene graph annotations.

## Model Parameters

- **`name`** (string):  
  The name of the Large Vision-Language Model (LVLM), following [Huggingface](https://huggingface.co/) tag format.

- **`path`** (string):  
  The path to the LVLM being used.

- **`family`** (string, choices: `llava` or `vip_llava`):  
  Specifies the LVLM family. The default is `llava`. Use `vip_llava` if the ViP-LLaVA series is required.

- **`params`**:  
  - **`use_8_bit`** (bool: `0/1`):  
    Enables or disables 8-bit quantization to reduce memory usage.
    
  - **`device`** (string, default: `cuda`):  
    Defines the computation device, such as `cuda` or `cpu`.
    
  - **`low_cpu`** (bool: `0/1`):  
    Enables the low CPU usage mode.

## Prompt Parameters

- **`prompt`** (string):  
  Specifies the instruction prompt to be used, formatted as `<dirname>-<filename>`.  
  Example: If using `/prompt/naive/optim.txt`, the value should be `naive-optim`.

## Inference Run Parameters

- **`num_per_inference`** (int):  
  The number of data points generated per image.

- **`use_img_ext`** (bool: `0/1`):  
  Flag to indicate whether to include image extensions in `img_id` during data processing.

- **`q_prefix`** (list of strings):  
  List of question prefixes for question generation.

- **`q_prefix_prop`** (list of ints):  
  The proportion corresponding to each question prefix in `q_prefix`.

Configurations discussed in the accompanying paper, used to construct the sample datasets, can be found in `/config/sample`. Each sample provides an illustrative example of how to structure datasets and experiments according to the guidelines above.

## LVLM Sources

- [LLaVA-1.5-7B](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [LLaVA-1.5-13B](https://huggingface.co/llava-hf/llava-1.5-13b-hf)
- [ViP-LLaVA-13B](https://huggingface.co/llava-hf/vip-llava-13b-hf)
