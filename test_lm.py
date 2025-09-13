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
    print(f"Original Text:\n'{text}'")

    # Prepare inputs
    inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Perform the full autoencoding process within a no_grad() block
    with torch.no_grad():
        # Get initial encoder outputs from the frozen BART encoder
        encoder_outputs = model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        

        reconstructed_encoder_outputs = model.encoder_output_to_decoder_input(
            encoder_outputs, attention_mask
        )

        # Generate text using the reconstructed encoder outputs
        generated_ids = model.generate(
            encoder_outputs=reconstructed_encoder_outputs,
            num_beams=4,
            max_length=64,
            early_stopping=True
        )

    reconstructed_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\nReconstructed Text:\n'{reconstructed_text}'")
    print("-" * 50)


test_sentence_1 = "absdasdfasf232324231324.']"
test_sentence_2 = "In the heart of the ancient forest, a hidden waterfall cascades into a crystal-clear pool."
test_sentence_3 = "Artificial intelligence is rapidly transforming every aspect of modern technology and society."

reconstruct_text(test_sentence_1, model, tokenizer )
reconstruct_text(test_sentence_2, model, tokenizer )
reconstruct_text(test_sentence_3, model, tokenizer )
