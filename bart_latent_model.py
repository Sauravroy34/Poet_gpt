import torch
import os 
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from transformers.models.bart.modeling_bart import (
    BartForConditionalGeneration,

)
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    BartConfig,
    get_scheduler,
)
from perceiver_ae import PerceiverAutoEncoder

from transformers import AutoTokenizer
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from torch.utils.data import DataLoader, Dataset


class BARTForConditionalGenerationLatent(BartForConditionalGeneration):
    def __init__(self, config, num_encoder_latents, num_decoder_latents, dim_ae, num_layers=2, l2_normalize_latents=False):
        super().__init__(config)
        self.num_encoder_latents = num_encoder_latents
        self.dim_ae = dim_ae
        self.l2_normalize_latents = l2_normalize_latents

        self.perceiver_ae = PerceiverAutoEncoder(dim_lm=config.d_model, num_encoder_latents=num_encoder_latents, num_decoder_latents=num_decoder_latents, dim_ae=dim_ae, depth=num_layers, transformer_decoder=True, l2_normalize_latents=l2_normalize_latents)

    def get_diffusion_latent(self, encoder_outputs, attention_mask):
        hidden_state = encoder_outputs[0]
        latent = self.perceiver_ae.encode(hidden_state, attention_mask.bool())
        return latent
        
    def get_decoder_input(self, diffusion_latent):
        return self.perceiver_ae.decode(diffusion_latent)
    
    # Map encoder outputs to decoder inputs
    def encoder_output_to_decoder_input(self, encoder_outputs, attention_mask):
        diffusion_latent = self.get_diffusion_latent(encoder_outputs, attention_mask)
            
        encoder_outputs['last_hidden_state'] = self.get_decoder_input(diffusion_latent)
        
        return encoder_outputs
    


# # Example setup
# model_name = "facebook/bart-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# config = BartForConditionalGeneration.from_pretrained(
#             model_name).config

# autoencoder = BARTForConditionalGenerationLatent.from_pretrained(
#     model_name,
#     config = config,
#     num_encoder_latents=32, 
#     num_decoder_latents=32, # Typically same as encoder latents
#     dim_ae=64,
#     num_layers=3, # Number of layers in compression/reconstruction networks 
#     l2_normalize_latents=True
# )

# for name, param in autoencoder.model.named_parameters():
#     param.requires_grad = False
    
# class PoemAutoencoderDataset(Dataset):

#     def __init__(self, poem_dirs, tokenizer=tokenizer, max_length=64):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.file_paths = []

#         for poem_dir in poem_dirs:
#             for root, _, files in os.walk(poem_dir):
#                 for file_name in files:
#                     if file_name.endswith('.txt'): # Ensure we only read text files
#                         self.file_paths.append(os.path.join(root, file_name))

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         file_path = self.file_paths[idx]
#         with open(file_path, 'r', encoding='utf-8') as f:
#             poem_text = f.read()

#         encodings_dict = self.tokenizer(
#             poem_text,
#             truncation=True,
#             max_length=self.max_length,
#             padding="max_length",
#             return_tensors='pt'
#         )

#         input_ids = encodings_dict['input_ids'].squeeze(0)
#         attention_mask = encodings_dict['attention_mask'].squeeze(0)

#         return {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'labels': input_ids.clone()  
#         }
        
# NUM_EPOCHS = 5
# MAX_GRAD_NORM = 1.0
     
        
# poem_dirs = ["forms","topics"]
# data = PoemAutoencoderDataset(poem_dirs=poem_dirs)
# optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=5e-5)

# data_loader = DataLoader(data,batch_size =32)
# lr_scheduler = get_scheduler(
#             'cosine',
#             optimizer=optimizer,
#             num_warmup_steps=200,
#             num_training_steps=1000,
#         )
# for epoch in range(NUM_EPOCHS):
#     autoencoder.train()
#     total_loss = 0
    
#     for step,batch in enumerate(data_loader):
        
#         optimizer.zero_grad()
        
#         with torch.no_grad():
#                 encoder_outputs = autoencoder.get_encoder()(
#                     input_ids=batch['input_ids'],
#                     attention_mask=batch['attention_mask']
#                 )
#         reconstructed_outputs = autoencoder.encoder_output_to_decoder_input(
#                 encoder_outputs, batch['attention_mask']
                
#             )

#         loss = autoencoder(
#                 labels=batch['labels'], 
#                 encoder_outputs=reconstructed_outputs
#             ).loss
        
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, autoencoder.parameters()), MAX_GRAD_NORM)
#         optimizer.step()
#         lr_scheduler.step()
#         total_loss += loss.item()
            
#         if step % 100 == 0:
#             print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
        
#     avg_loss = total_loss / len(data_loader)
#     print(f"--- End of Epoch {epoch} --- Average Loss: {avg_loss:.4f}")