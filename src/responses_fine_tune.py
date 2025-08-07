from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from rich.markdown import Markdown
from rich import print
import torch

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

if __name__ == "__main__":
  # static
  fine_model_id = "ludyhasby/lamini_docs_100_steps"
  # load tokenizer model
  tokenizer = AutoTokenizer.from_pretrained(fine_model_id)
  tokenizer.pad_token = tokenizer.eos_token
  # load fine tune model
  fine_model = AutoModelForCausalLM.from_pretrained(fine_model_id) 

  # --- MAIN ---
  test_text = input("Input Question/Chat Regarding Lamini\n").strip()
  print("\n--Ready to Response--\nWait for minutes..\n")
  response = inference(test_text, fine_model, tokenizer)
  md = Markdown(response)
  print("\n")
  print(md)
