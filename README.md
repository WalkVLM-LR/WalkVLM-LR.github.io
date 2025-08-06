# WalkVLM-LR

A walking assistance VLM with reduced redundancy for blind and low vision individuals.

## Overview

WalkVLM-LR is a walking assistance model designed to improve navigation for individuals with visual impairments. It reduces both output and temporal redundancy compared to existing models. By using human-preference-based custom reward functions and an environment awareness discriminator, WalkVLM-LR generates concise, accurate, and context-appropriate guidance while minimizing unnecessary reminders. Experimental results show it outperforms other models, particularly in output conciseness and reducing temporal redundancy.

## Installation

### Prerequisites

- CUDA 12.4
- Python 3.11
- PyTorch 2.6.0

### Setup

```bash
# Clone the repository
git clone https://github.com/huggingface/open-r1.git

# Install dependencies
pip install --upgrade pip
pip install vllm==0.8.5.post1
pip install setuptools && pip install flash-attn --no-build-isolation
GIT_LFS_SKIP_SMUDGE=1 pip install -e ".[dev]"
```
<!-- Next, log into your Hugging Face and Weights and Biases accounts as follows:

```shell
huggingface-cli login
wandb login
```

Finally, check whether your system has Git LFS installed so that you can load and push models/datasets to the Hugging Face Hub:

```shell
git-lfs --version
```

If it isn't installed, run:

```shell
sudo apt-get install git-lfs
``` -->

## Training

We use GRPO to fine-tune WalkVLM-LR.

### GRPO Training Command

```bash
# Run the training script
cd vlm_grpo_template
bash run_grpo_query_gene.sh
```

### GRPO Training Configuration

You can modify the training parameters in the `run_grpo_query_gene.sh` script, including:
- output_dir
- max_prompt_length
- num_train_epochs
- dataset_path 

### EAD Training Command
Configure the data path in trainEAD.py, then run the training code to train the EAD model.
```bash
python -m torch.distributed.launch --nproc_per_node=8 train_EAD.py
```

## Testing Command
Modify the checkpoint_path and image_paths in test.py, and then proceed with the testing. The pre-trained weights for CLIP and the related parameters for the GPT-4 API need to be configured manually.
```bash
python test.py
```


## Inference Command
Modify the checkpoint_path and image_paths (with a limit of three images per input) in inference.py, and then perform the inference.

```bash
python inference.py
```

## Contact

For questions and feedback, please open an issue and contact.
