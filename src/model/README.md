---
license: apache-2.0
base_model: EleutherAI/pythia-410m
tags:
- generated_from_trainer
model-index:
- name: lamini_docs_100_steps
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# lamini_docs_100_steps

This model is a fine-tuned version of [EleutherAI/pythia-410m](https://huggingface.co/EleutherAI/pythia-410m) on an unknown dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-06
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 4
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 1
- training_steps: 100

### Training results



### Framework versions

- Transformers 4.37.2
- Pytorch 2.5.1+cpu
- Datasets 2.14.6
- Tokenizers 0.15.2
