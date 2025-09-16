from bart_latent_model import BARTForConditionalGenerationLatent
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
)
from transformers.models.bart.modeling_bart import (
    BartForConditionalGeneration,

)
import torch 

model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

config = BartForConditionalGeneration.from_pretrained(
            model_name).config



model  = BARTForConditionalGenerationLatent.from_pretrained(
    model_name,
    config = config,
    num_encoder_latents=32, 
    num_decoder_latents=32, # Typically same as encoder latents
    dim_ae=64,
    num_layers=3, # Number of layers in compression/reconstruction networks 
    l2_normalize_latents=True
)

model.perceiver_ae.load_state_dict(torch.load("perceiver_ae_weights3.pth",map_location = torch.device('cpu')))

model.eval()

def reconstruct_text(text, model, tokenizer):
    """Encodes a text string to a latent vector and decodes it back to text."""
    print("-" * 50)
    # print(f"Original Text:\n'{text}'")

    # Prepare inputs
    inputs = tokenizer(text, return_tensors="pt", max_length=768, truncation=True, padding="max_length")
    print("the length",len(inputs.input_ids))
    print("\n"*10)
    print("reconstructed text ")
    out = tokenizer.decode(inputs.input_ids[0],skip_special_tokens=True)
    print(out)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Perform the full autoencoding process within a no_grad() block for efficiency
    with torch.no_grad():
        # Get initial encoder outputs from the frozen BART encoder
        encoder_outputs = model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        

        # 1. Get the compressed latent vector (32x64) from the encoder's output
        diffusion_latent = model.get_diffusion_latent(
            encoder_outputs, attention_mask
        )
        print(diffusion_latent.shape)

        # 2. Reconstruct the features for the decoder from the latent vector
        reconstructed_hidden_states = model.get_decoder_input(
            diffusion_latent
        )
        print(reconstructed_hidden_states.shape)
        # 3. Overwrite the original encoder_outputs' hidden state with the reconstructed one.
        # This is what the decoder will use for cross-attention.
        encoder_outputs['last_hidden_state'] = reconstructed_hidden_states

        # 4. Generate text using these new reconstructed features
        generated_ids = model.generate(
            encoder_outputs=encoder_outputs, # Pass the modified encoder_outputs object
            num_beams=4,
            max_length=64,
            early_stopping=True
        )
    print(len(generated_ids))
    reconstructed_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

test_sentence_1 = "absdasdfasf232324231324.']"
test_sentence_2 = "In the heart of the ancient forest, a hidden waterfall cascades into a crystal-clear pool."
with open("AbecedarianPoemsManShapedWindTheAbecedarianPoembyJaeyPeele.txt") as f:
    poem = f.read()

reconstruct_text(poem, model, tokenizer ) 
