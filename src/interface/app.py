import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# --- Load Model ---
model_repo = "ludyhasby/lamini_docs_100_steps"
model_subfolder = "final"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_repo, subfolder=model_subfolder, use_auth_token=token)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_repo, subfolder=model_subfolder, device_map="auto", use_auth_token=token)
model.to(device)
model.eval()

def clean_response(text, prompt):
    # Ambil isi setelah prompt
    content = text[len(prompt):].strip()

    # Pisah berdasarkan titik
    sentences = [s.strip() for s in content.split(".") if s.strip()]

    # Hapus kalimat duplikat
    seen = set()
    unique_sentences = []
    for s in sentences:
        if s not in seen:
            unique_sentences.append(s)
            seen.add(s)

    # Gabung kembali menjadi satu string
    return '. '.join(unique_sentences) + '.'

# --- Inference Function ---
def inference(prompt):
    max_input_token = 512
    max_output_token = 100
    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=max_input_token).to(device)
    with torch.no_grad():
        output = model.generate(inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_output_token)
    decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
    return clean_response(decoded[0], prompt)

# --- UI Setup ---
with gr.Blocks(title="Chatbot Lamini 100 Steps") as demo:
    gr.Markdown("""
    # 🤖 Chat with Pythia LLM Fine Tune By Lamini Docs
    Ask Me Anything About Lamini and I will Give My Response/Instruction with the best of my knowledge!
    """)
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(
                label="📝 Prompt",
                placeholder="Example: How do I report a bug or issue with the Lamini documentation?",
                lines=4
            )
            submit_btn = gr.Button("🚀 Generate Response")
            gr.Markdown("Built by : Ludy Hasby x Bajau Escorindo")
        with gr.Column(scale=4):
            response = gr.Textbox(label="📤 Response", lines=10, interactive=True)

    submit_btn.click(fn=inference, inputs=prompt, outputs=response)

demo.launch()