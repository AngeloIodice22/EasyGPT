import torch
from train import ChatConfig, ChatModel, ChatTokenizer

def load_model_and_tokenizer(model_path):
    config = ChatConfig.from_pretrained(model_path)
    model = ChatModel.from_pretrained(model_path, config)
    tokenizer = ChatTokenizer.from_pretrained(model_path)
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

def chat(text, model, tokenizer, device, max_length=1024, temperature=0.7, top_p=0.95, repetition_penalty=1.2, presence_penalty=-1.0):
    tokens = tokenizer.tokenize(f"<|user|>{text}<|assistant|>")
    input_ids = tokenizer.convert_tokens_to_ids(tokens, update_vocab=False)
    generated = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        while generated.size(1) < max_length:
            outputs = model(generated)
            logits = outputs["logits"][0, -1, :]

            for token in set(generated.squeeze().tolist()):
                logits[token] = logits[token] / repetition_penalty - presence_penalty

            scaled_logits = logits / temperature
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[0] = False
            scaled_logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')
            probabilities = torch.softmax(scaled_logits, dim=-1)

            if torch.isnan(probabilities).any() or probabilities.sum() == 0:
                next_token = torch.argmax(scaled_logits, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probabilities, num_samples=1)

            token_str = tokenizer.reverse_vocab.get(next_token.item(), "<unknown>")
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            if token_str == "<|end|>":
                break
            elif token_str == "\\n":
                print()
            else:
                print(token_str, end='', flush=True)

if __name__ == "__main__":
    print("EasyGPT Beta V1.2 Torch Inference (Dev)")
    model, tokenizer, device = load_model_and_tokenizer("./model/pretrain_epoch_30")
    while True:
        print("="*50)
        text = input("input: ")
        if text.strip().lower() == "exit":
            break
        chat(text, model, tokenizer, device)
        print()
