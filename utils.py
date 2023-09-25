import subprocess
from typing import TypeAlias, Literal, Optional

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


device_type: TypeAlias = Literal['cpu', 'cuda']


def launch_tensorboard(log_dir: str, port: int = 6666):
    try:
        cmd = f"tensorboard --logdir {log_dir} --port={port}"
        process = subprocess.Popen(cmd, shell=True)
        print(f"TensorBoard is running at http://localhost:{port}/")
        process.wait()

    except Exception as e:
        print(f"Error launching TensorBoard: {e}")


def get_model_tokenizer(model_name: str, map_device: bool = True):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if map_device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    return (model, tokenizer)


def generate_text(model, tokenizer, input_text: str, max_length: int = 200, no_repeat_ngram_size: int = 2,
                  temperature: float = 0.7, top_p: float = 0.95, device: Optional[device_type] = None) -> str:
    if device:
        model.to(device)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output = model.generate(input_ids,
                            top_p=top_p, # top_k is inferior
                            pad_token_id=50256,
                            max_length=max_length,
                            num_return_sequences=1,
                            temperature=temperature,
                            no_repeat_ngram_size=no_repeat_ngram_size,
                            attention_mask=input_ids.new_ones(input_ids.shape))

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


if __name__ == '__main__':
    model_name = "shakespeare_gpt2_finetuned"  # Use the valid directory name
    model, tokenizer = get_model_tokenizer(model_name)
    input_text = "To be or not to be, that is the question:"
    generated_text = generate_text(model, tokenizer, input_text, max_length=400)
    print(generated_text)
