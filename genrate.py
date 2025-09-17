import torch
from transformers import AutoTokenizer, AutoModel, BartForConditionalGeneration
from diffusers import SanaTransformer2DModel, DDPMScheduler
from tqdm.auto import tqdm
from bart_latent_model import BARTForConditionalGenerationLatent

from transformers.modeling_outputs import BaseModelOutput


print("Setting up models and tokenizers...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "facebook/bart-base"
config = BartForConditionalGeneration.from_pretrained(model_name).config

model = BARTForConditionalGenerationLatent.from_pretrained(
    model_name,
    config=config,
    num_encoder_latents=32,
    num_decoder_latents=32,
    dim_ae=64,
    num_layers=3,
    l2_normalize_latents=True
)
model.perceiver_ae.load_state_dict(torch.load("model/perceiver_ae_weights3.pth", map_location=device))
model.to(device)
model.eval()


tokenizer_lm = AutoTokenizer.from_pretrained(model_name)
te_repo = "HuggingFaceTB/SmolLM2-360M"
tokenizer_encode = AutoTokenizer.from_pretrained(te_repo)
tokenizer_encode.pad_token = tokenizer_encode.eos_token

text_encoder = AutoModel.from_pretrained(te_repo)
text_encoder.to(device)
text_encoder.eval()

# --- Load the Fine-Tuned Transformer ---
config = SanaTransformer2DModel.load_config("Efficient-Large-Model/Sana_600M_1024px_diffusers", subfolder="transformer")
config["num_layers"] = 12
config["num_attention_heads"] = 12
config["attention_head_dim"] = 64
config["cross_attention_dim"] = 768
config["num_cross_attention_heads"] = 12
config["cross_attention_head_dim"] = 64
config["caption_channels"] = 960

transformer = SanaTransformer2DModel.from_config(config)

# Define the path to your saved weights
LOAD_PATH = "model/Text_diffusion.pth" # Or "Text_diffusion.pth"
transformer.load_state_dict(torch.load(LOAD_PATH, map_location=device))
transformer.to(device)
transformer.eval()

print("All models loaded successfully!")


def encode_prompt(prompt, tokenizer=tokenizer_encode, text_encoder=text_encoder, max_length=50, add_special_tokens=False, **kwargs):
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



def generate(prompt, transformer = transformer, tokenizer=tokenizer_lm,num_steps = 10, latent_dim = [1, 32, 16, 4], guidance_scale = 3, neg_prompt = "", seed=None, max_prompt_tok=50, add_special_tokens=False):
    device, dtype = transformer.device, transformer.dtype
    do_cfg = guidance_scale is not None

    # Encode the prompt, +neg. prompt if classifier free guidance (CFG)
    prompt_encoded, prompt_atnmask = encode_prompt(
        [prompt, neg_prompt] if do_cfg else prompt, 
    )
        
    # Divide 1000 -> 0 in equally sized steps
    timesteps = torch.linspace(1000, 0, num_steps + 1, device=device, dtype=dtype)
    
    # Noise level. 1.0 -> 0.0 in equally sized steps
    sigmas = timesteps / 1000
    
    latent = torch.randn(
        latent_dim, 
        generator=torch.manual_seed(seed) if seed else None
    ).to(dtype).to(device)
    
    for t, sigma_prev, sigma_next, steps_left in zip(
        timesteps, 
        sigmas[:-1], 
        sigmas[1:], 
        range(num_steps, 0, -1)
    ):
        t = t[None].to(device)

        # DiT predicts noise
        with torch.no_grad():
            noise_pred = transformer(
                hidden_states = torch.cat([latent] * 2) if do_cfg else latent,
                timestep = torch.cat([t] * 2) if do_cfg else t,
                encoder_hidden_states=prompt_encoded,
                encoder_attention_mask=prompt_atnmask,
                return_dict=False
            )[0]
        
        if do_cfg:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        latent = latent + (sigma_next - sigma_prev) * noise_pred 
        b = latent.shape[0]
        latent = latent.view(b,32,64)
        
        with torch.no_grad():
            reconstructed_hidden_states = model.get_decoder_input(
                latent
            )
            
            encoder_attention_mask = torch.ones(
            reconstructed_hidden_states.shape[:-1],
            dtype=torch.long,
            device=device
        )
            encoder_outputs = BaseModelOutput(last_hidden_state=reconstructed_hidden_states)
            
            generated_ids = model.generate(
                encoder_outputs = encoder_outputs,
                attention_mask=encoder_attention_mask,
                num_beams = 4,
                max_length =64,
                early_stopping = True
                
            )
            reconstructed_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return reconstructed_text

print("\n"*4)
print(generate("Romance"))


