import torch 
from transformers import (
    AutoTokenizer,
    AutoModel,
    BartForConditionalGeneration,
)
from bart_latent_model import BARTForConditionalGenerationLatent
from diffusers import SanaTransformer2DModel
import os 
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# 1. Set the device to GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- You should have your custom class defined or imported here ---
# from your_module import BARTForConditionalGenerationLatent 
# This script assumes 'BARTForConditionalGenerationLatent' is available in your environment.

# --- Model and Tokenizer Loading ---

model_name = "facebook/bart-base"
tokenizer_lm = AutoTokenizer.from_pretrained(model_name)
config = BartForConditionalGeneration.from_pretrained(model_name).config

text_encoder = AutoModel.from_pretrained("HuggingFaceTB/SmolLM2-360M")
# Move text_encoder to the selected device
text_encoder.to(device)

model = BARTForConditionalGenerationLatent.from_pretrained(
    model_name,
    config=config,
    num_encoder_latents=32, 
    num_decoder_latents=32,
    dim_ae=64,
    num_layers=3, 
    l2_normalize_latents=True
)

# Load state dict with map_location set to the device
model.perceiver_ae.load_state_dict(torch.load("perceiver_ae_weights3.pth", map_location=device))

# Move the main model to the selected device
model.to(device)
model.eval()

def encode_prompt(prompt, tokenizer, text_encoder, max_length=50, add_special_tokens=False, **kwargs):
    tokenizer.padding_side = "right"
    if isinstance(prompt, list):
        prompt = [p.lower().strip() for p in prompt]
    elif isinstance(prompt, str):
        prompt = prompt.lower().strip()
    else:
        raise Exception(f"Unknown prompt type {type(prompt)}")
    
    # Move tokenized inputs to the text_encoder's device
    prompt_tok = tokenizer(prompt, return_tensors="pt", return_attention_mask=True, padding="max_length", truncation=True, max_length=max_length, add_special_tokens=add_special_tokens).to(text_encoder.device)
    
    with torch.no_grad():
        prompt_encoded = text_encoder(**prompt_tok)
    return prompt_encoded.last_hidden_state, prompt_tok.attention_mask

te_repo = "HuggingFaceTB/SmolLM2-360M"
tokenizer_encode = AutoTokenizer.from_pretrained(te_repo)
tokenizer_encode.pad_token = tokenizer_encode.eos_token

class PoemDiffusionDataset(Dataset):
    def __init__(self, poem_dirs, tokenizer_lm=tokenizer_lm, tokenizer_prompt=tokenizer_encode, text_encoder=text_encoder, max_length_poem=64, max_length_prompt=50, p_uncond=0.1):
        self.tokenizer_lm = tokenizer_lm
        self.tokenizer_prompt = tokenizer_prompt
        self.text_encoder = text_encoder
        self.max_length_poem = max_length_poem
        self.max_length_prompt = max_length_prompt
        self.p_uncond = p_uncond
        self.file_paths = []
        for poem_dir in poem_dirs:
            for root, _, files in os.walk(poem_dir):
                for file_name in files:
                    if file_name.endswith('.txt'):
                        self.file_paths.append(os.path.join(root, file_name))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, 'r', encoding='utf-8') as f:
            poem_text = f.read()

        encodings_dict = self.tokenizer_lm(
            poem_text,
            truncation=True,
            max_length=self.max_length_poem,
            padding="max_length",
            return_tensors='pt'
        )
        # Tensors from the tokenizer are moved to the model's device
        input_ids = encodings_dict['input_ids'].to(model.device)
        attention_mask = encodings_dict['attention_mask'].to(model.device)
        
        with torch.no_grad():
            encoder_outputs = model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
            diffusion_latent = model.get_diffusion_latent(encoder_outputs, attention_mask)
        
        if torch.rand(1) < self.p_uncond:
            prompt_text = "" 
        else:
            filename = os.path.basename(file_path)
            prompt_text = os.path.splitext(filename)[0]

        encoded_prompts, prompt_attention_mask = encode_prompt(
            prompt_text,
            self.tokenizer_prompt,
            self.text_encoder,
            max_length=self.max_length_prompt
        )
        
        # Tensors are returned on the CPU. They will be moved to GPU in the training loop batch-wise for efficiency.
        return {
            'diffusion_latent': diffusion_latent.squeeze(0).cpu(),
            'prompt_embeds': encoded_prompts.squeeze(0).cpu(),
            'prompt_attention_mask': prompt_attention_mask.squeeze(0).cpu()
        }

def add_random_noise(latents, timesteps=1000, dist="uniform"):
    assert dist in ["normal", "uniform"], f"Requested sigma dist. {dist} not supported"
    bs = latents.size(0)
    noise = torch.randn_like(latents) # Creates noise on the same device as latents

    if dist == "normal":
        sigmas = torch.randn((bs,)).sigmoid().to(latents.device)
    else:
        sigmas = torch.rand((bs,)).to(latents.device)
    
    timesteps = (sigmas * timesteps).to(latents.device)
    sigmas = sigmas.view([latents.size(0), *([1] * len(latents.shape[1:]))])
    
    latents_noisy = (1 - sigmas) * latents + sigmas * noise
    return latents_noisy.to(latents.dtype), timesteps, noise

# --- Transformer and Training Setup ---

config = SanaTransformer2DModel.load_config("Efficient-Large-Model/Sana_600M_1024px_diffusers", subfolder="transformer")
config["num_layers"] = 12
config["num_attention_heads"] = 12
config["attention_head_dim"] = 64
config["cross_attention_dim"] = 768
config["num_cross_attention_heads"] = 12
config["cross_attention_head_dim"] = 64
config["caption_channels"] = 960

transformer = SanaTransformer2DModel.from_config(config)
# Move the main training model to the selected device
transformer.to(device)

poem_dirs = ["forms", "topics"]
data = PoemDiffusionDataset(poem_dirs=poem_dirs)
optimizer = torch.optim.AdamW(transformer.parameters(), lr=5e-5)
data_loader = DataLoader(data, batch_size=32)

# --- Training Loop ---
epochs = 1
for e in range(epochs):
    transformer.train()
    for step, batch in enumerate(data_loader):
        # 2. Move data batch to the selected device
        latents = batch["diffusion_latent"].to(device)
        prompt_embeds = batch["prompt_embeds"].to(device)
        prompt_attention_mask = batch["prompt_attention_mask"].to(device)
        
        # Use latents.shape[0] to handle the last batch which might be smaller than 32
        batch_size = latents.shape[0]
        latents = latents.view(batch_size, 32, 16, 4)
        
        latents_noisy, timestep, noise = add_random_noise(latents)
        
        noise_pred = transformer(
            hidden_states=latents_noisy, 
            encoder_hidden_states=prompt_embeds, 
            encoder_attention_mask=prompt_attention_mask,
            timestep=timestep, 
        ).sample
        
        optimizer.zero_grad()
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()    
        grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
        optimizer.step()
    
        if step % 10 == 0:
            # .item() moves the loss value from GPU to CPU for printing
            print(f"Epoch {e+1}, Step {step+1}, Loss: {loss.item():.4f}")