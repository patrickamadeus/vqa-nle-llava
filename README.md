# 👁️ ⛏️ Towards Efficient and Robust VQA-NLE Data Generation with Large Vision-Language Models

[Paper](https://arxiv.org/abs/2409.14785) &nbsp; [🤗 Datasets](https://huggingface.co/datasets/patrickamadeus/vqa-nle-llava)

## Setup
```bash
$ cd src/scripts
$ yes | source setup.sh
```

## IMPORTANT NOTES :warning: :warning:

We only provide some image samples in `/dataset/img/`. Please refer to our [dataset hub](https://huggingface.co/datasets/patrickamadeus/vqa-nle-llava) for complete data.


## Hyperparameters Configuration

```yaml
# This .yml file defines the key parameters for controlling experiment setups, including dataset details,
# model configurations, and inference behaviors. The results for each experiment will be saved under /result/{test_name}.

test_name:  # string: The name of the dataset being used. Results will be stored in the /result/{test_name} directory.
seed:       # int: The random seed for the experiment, used for reproducibility.

dataset:
  image_count:      # int, must be > 0: Defines the number of images to generate during the experiment.
  use_scene_graph:  # bool (0/1): Flag for whether to incorporate scene graph annotations.

model:
  name:             # string: The name of the Large Vision-Language Model (LVLM), following Huggingface tag format.
  path:             # string: The path to the LVLM being used.
  family:           # string, choices: ['llava', 'vip_llava']: Specifies the LVLM family. Default is 'llava'.
                    # Use 'vip_llava' if the ViP-LLaVA series is required.
  params:
    use_8_bit:      # bool (0/1): Enables or disables 8-bit quantization to reduce memory usage.
    device:         # string, default: 'cuda': Defines the computation device, such as 'cuda' or 'cpu'.
    low_cpu:        # bool (0/1): Enables the low CPU usage mode.

prompt:           # string: Specifies the instruction prompt to be used, formatted as '<dirname>-<filename>'.
                  # Example: If using '/prompt/naive/optim.txt', the value should be 'naive-optim'.

run_params:
  num_per_inference:  # int: The number of data points generated per image.
  use_img_ext:        # bool (0/1): Flag to indicate whether to include image extensions in img_id during data processing.
  q_prefix:           # list of strings: List of question prefixes for question generation.
  q_prefix_prop:      # list of ints: The proportion corresponding to each question prefix in q_prefix.
```

The configs that were used to generate huggingface datasets can be found in `/src/config/sample`.

## LVLM Sources

- [LLaVA-1.5-7B](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [LLaVA-1.5-13B](https://huggingface.co/llava-hf/llava-1.5-13b-hf)
- [ViP-LLaVA-13B](https://huggingface.co/llava-hf/vip-llava-13b-hf)

## Citation
```bib
@article{irawan2024towards,
  title={Towards Efficient and Robust VQA-NLE Data Generation with Large Vision-Language Models},
  author={Irawan, Patrick Amadeus and Winata, Genta Indra and Cahyawijaya, Samuel and Purwarianti, Ayu},
  journal={arXiv preprint arXiv:2409.14785},
  year={2024}
}
```
