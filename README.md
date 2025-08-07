# Instruction Tuning with Pythia

Ludy Hasby Aulia

[[Project Page](https://huggingface.co/ludyhasby/lamini_docs_instruct)] [[Notebook](waiting)] 
<p align="center">
    <img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*SwMMluhfo_YW1-9Mwpb8kg.png" width="80%"> <br>
    Instruction Tuning, Image take from<a href="https://medium.com/@lmpo/an-overview-instruction-tuning-for-llms-440228e7edab"> LM PRO</a>
</p>


<!-- 
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE) -->


This project focuses on fine-tuning the large language model 'EleutherAI/pythia-410m' to enhance its ability to generate accurate and relevant responses to instruction-based prompts. By leveraging instruction-tuning techniques, we aim to:

- Reduce hallucinations and unwanted outputs
- Improve consistency and reliability in generated answers
- Enhance data privacy for company-specific use cases
- Lower operational costs by optimizing model performance

Fine-tuning also enables the model to better align with domain-specific requirements and organizational standards.

**Key Libraries Used:**
- PyTorch: For efficient deep learning model training and optimization
- Transformers: For state-of-the-art NLP model architectures and utilities
- LLama Library (Lamini): For streamlined instruction-tuning workflows

This repo contains: 
- Fine Tune Model Tokenization
- Fine Tune Model Trainer 
- Lamini Docs Dataset
- Notebook Model Development 
- Inference App with HuggingFace 

**Usage and License Notices**:  The dataset is CC BY [Lamini](https://huggingface.co/datasets/lamini/lamini_docs)

- [Overview](#overview)
- [LLM Selected](#base-large-language-model)
- [Dataset Design and Preparation](#data-design-preparation)
- [Fine Tuning Strategy](#fine-tune-strategy)
- [Evaluation and Benchmarking](#evaluation-benchmarking)
- [Practical Implementation](#practical-implementation)

## Overview
Large Language Models (LLMs) have shown impressive generalization capabilities such as in-context-learning and chain-of-thoughts reasoning. To enable LLMs to follow natural language instructions and complete real-world tasks, we have been exploring methods of instruction-tuning of LLMs. 
This project demonstrates the process of instruction-tuning a large language model (LLM), specifically EleutherAI/pythia-410m, to improve its ability to follow natural language instructions and generate high-quality, relevant responses. By leveraging the [lamini_docs](https://huggingface.co/datasets/lamini/lamini_docs) dataset, we fine-tune the base model to better align with real-world instruction-following tasks, reduce hallucinations, and enhance reliability.

## Base Large Language Model
For this project, **EleutherAI/pythia-410m** was chosen due to the following reasons:
- **Accessibility & Licensing:** Pythia is fully open-source and available on Hugging Face, making it easy to use, modify, and deploy without restrictive licenses.
- **Architecture:** It is based on the transformer architecture, which is well-suited for understanding and generating coherent, context-aware text.
- **Community Support:** Pythia has strong community backing, with pre-trained weights, documentation, and integration with popular libraries like `transformers`.
- **Performance:** While smaller than some models, Pythia-410m offers a good balance between computational efficiency and output quality, making it suitable for experimentation and prototyping.
- **Instruction-Tuning Compatibility:** The model can be fine-tuned on instruction datasets (such as lamini_docs) to improve its ability to follow prompts and generate relevant, structured responses.

Other models like LLaMA, Mistral, or DeepSeek may offer higher performance or larger parameter sizes, but Pythia is a practical choice for projects focused on open-source, reproducibility, and ease of deployment.

Here is `EleutherAI/pythia-410m` architectures:
```
GPTNeoXForCausalLM(
  (gpt_neox): GPTNeoXModel(
    (embed_in): Embedding(50304, 1024)
    (emb_dropout): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0-23): 24 x GPTNeoXLayer(
        (input_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (post_attention_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (post_attention_dropout): Dropout(p=0.0, inplace=False)
        (post_mlp_dropout): Dropout(p=0.0, inplace=False)
        (attention): GPTNeoXAttention(
          (rotary_emb): GPTNeoXRotaryEmbedding()
          (query_key_value): Linear(in_features=1024, out_features=3072, bias=True)
          (dense): Linear(in_features=1024, out_features=1024, bias=True)
          (attention_dropout): Dropout(p=0.0, inplace=False)
        )
        (mlp): GPTNeoXMLP(
          (dense_h_to_4h): Linear(in_features=1024, out_features=4096, bias=True)
          (dense_4h_to_h): Linear(in_features=4096, out_features=1024, bias=True)
          (act): GELUActivation()
        )
      )
    )
    (final_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
  )
  (embed_out): Linear(in_features=1024, out_features=50304, bias=False)
)
```

## Data Design Preparation
### Dataset Information
* [`lamini_docs.jsonl`](https://huggingface.co/datasets/lamini/lamini_docs) contains 1260 instruction-following preferrable response regarding Lamini information. 
This JSON file has the format as belom:

    - `question`: `str`, A natural language instruction or prompt describing the task. 
    - `answer`: `str`, The preferred answer to the instruction, generated by Lamini.

**Data Testing Example**

`Question input (test)`: Can Lamini generate technical documentation or user manuals for software projects?

`Prefer answer from Lamini docs`: Yes, Lamini can generate technical documentation and user manuals for software projects. It uses natural language generation techniques to create clear and concise documentation that is easy to understand for both technical and non-technical users. This can save developers a significant amount of time and effort in creating documentation, allowing them to focus on other aspects of their projects.

### Data Preprocessing
Data is first loaded and then processed using the base model's tokenizer. The preprocessing steps include:

- **Tokenization:** Each question and answer is converted into tokens using the tokenizer from the pretrained model.
- **Padding and Truncation:**  
  - Questions are padded or truncated to a fixed length of 1000 tokens.
  - Answers are padded or truncated to a fixed length of 100 tokens.
  This ensures all inputs and outputs have consistent shapes for efficient training.
- **Train-Test Split:**  
  After preprocessing, the dataset is split into training and testing sets to evaluate model performance.

This workflow prepares the data for fine-tuning and ensures compatibility with. Then we make pipelines to inference each input to output. with steps and function as follow: 

1. **Generate Tokenization from Prompt**: using model tokenizer
2. **Padding and Truncating** : Since models expect inputs of fixed length, tokenized sequences are padded (adding special tokens to reach the required length) or truncated (cutting off tokens that exceed the maximum length). This ensures uniform input size for efficient batch processing.
3. **Generate Model Response**
4. **Decode the Result from Tokenization**: The output tokens produced by the model are converted back into human-readable. 
5. **Strip the Prompt**  
   The decoded output often contains the original prompt followed by the model’s response. To isolate the model’s answer, the prompt portion is removed, leaving only the generated response for evaluation or further processing.

```python
    def inference(prompt, model, tokenizer, max_input_token=1000, max_output_token=100):
        """
        Function to generate model response from prompt
        """
        # Generate Tokenization from prompt
        inputs = tokenizer.encode(
            prompt, 
            return_tensors="pt",
            truncation=True, 
            max_length=max_input_token
        )
        # Generate Response
        device = model.device
        generate_token = model.generate(
            inputs.to(device), 
            max_new_tokens=max_output_token
        )
        # Decode the result from tokenization
        response = tokenizer.batch_decode(generate_token, 
                                        skip_special_tokens=True)    
        # Strip the prompt
        response = response[0][len(prompt):]
        return response
```

### Handle Unrelevant Information
To handle questions that are outside the scope of Lamini Docs, the dataset includes examples specifically designed to teach the model to respond appropriately. For instance:

- `Question:`
  *Why do we shiver when we're cold?*

- `Answer:`
  *Let’s keep the discussion relevant to Lamini.*

- `Question:`
  *Why do we dream?*

- `Answer:`
  *Let’s keep the discussion relevant to Lamini.*

This approach helps the model avoid answering unrelated questions and maintain focus on Lamini-

## Fine Tune Strategy
### Key Hyperparameters to Tune
- `learning_rate=1e-6`, # learning rate, we reduce it because avoiding overfitting
- `max_steps=100`, # steps can take up to 100 because of cost of computation
- `per_device_train_batch_size=1`, # batch size per device during training, we dont use GPU
- `warmup_steps=1`, # warmup steps, to be stable
- `per_device_eval_batch_size=1`, # we dont use GPU
- `optim="adamw_torch"`, # optimizer, I think state of art
- `gradient_accumulation_steps = 4`, # beneficial to minimum GPU
- `gradient_checkpointing=False`,
- `load_best_model_at_end=True`,
- `metric_for_best_model="eval_loss"`
### Training Result
Here is our [logs](/src/logs.txt) that you can evaluate. Or you can Check our [Notebook](/src/instructionTuning.ipynb). 
### Potential Challenge
- Computational Resources : limitation of RAM and GPU, solution : Choose small LLM base model (400m - 1B)
- Repeating answer, solution : truncating response
- Bahasa Indonesia Context, solution: translating model before preprocessing, vice verse before sending

## Evaluation Benchmarking
To assess the effectiveness of instruction tuning, we compare the responses generated by the baseline (pretrained) model and the fine-tuned model on both training and testing datasets. This benchmarking process highlights improvements in the model's ability to follow instructions and generate relevant answers.

**Evaluation Steps:**
1. **Select Sample Questions:**  
   Use representative questions from both the training and testing sets.
2. **Generate Responses:**  
   Obtain answers from the baseline model and the fine-tuned model for each question.
3. **Compare Outputs:**  
   Evaluate the quality, relevance, and alignment of the generated responses against the preferred answers from the Lamini Docs dataset.

<p align="center">
    <img src="https://github.com/user-attachments/assets/54dcd733-1eb7-4f54-a2d4-c4917e9e749f" width="100%">
</p>

## Implementation for Basic Fine Tuning Pipeline
These fine tuning model can be sketch as below
#### Fine-Tuning Pipeline
You should write a simplified implementation flow of how you would:
- Load a pre-trained open-source LLM 'EleutherAI/pythia-410m'
- Load instruction dataset `lamini_docs.jsonl`
- Tokenize and preprocess the data `base_model_tokenizer`
- Training Config `TrainingArguments(...)`
- Run the training using Hugging Face’s `Trainer(...)` or similar API

#### Our Fine Tune Model 
#### 🏆 Our Fine-Tuned Model

You can explore and use our fine-tuned model directly on Hugging Face:  
[**Lamini Docs Instruction-Tuned Model**](https://huggingface.co/ludyhasby/lamini_docs_100_steps/tree/main)

---

#### 🚀 Workflow: Generate Instructions with the Fine-Tuned Model

**Step-by-step guide to run inference from scratch:**

1. **Load the Fine-Tuned Model**
    ```python
    from transformers import AutoModelForCausalLM

    fine_model_id = "ludyhasby/lamini_docs_100_steps"
    fine_model = AutoModelForCausalLM.from_pretrained(fine_model_id)
    ```

2. **Load the Tokenizer**
    ```python
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(fine_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    ```

3. **Prepare Your Instruction/Question**
    - Write your prompt or instruction as a string.

4. **Preprocess the Input**
    - Tokenize your prompt using the loaded tokenizer.

5. **Generate Model Response**
    - Pass the tokenized input to the fine-tuned model for inference.

6. **Decode the Output**
    - Convert the model's output tokens back to human-readable text.

---

**Example Usage:**
```python
prompt = "How does Lamini handle background jobs?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = fine_model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

---

### 🖥️ Run the Ready-to-Use Program

For a streamlined experience, simply run our provided script:
```bash
python src/load_fine_tune.py
```
This will automatically load the fine-tuned model and tokenizer, and prompt you for instructions.

---

> **Tip:**  
> You can further customize the inference pipeline for batch processing, web API integration, or evaluation

## Acknowledgement
This project benefits from [Lamini](https://huggingface.co/datasets/lamini/lamini_docs), [EleutherAI/phythia](https://huggingface.co/EleutherAI/pythia-410m)