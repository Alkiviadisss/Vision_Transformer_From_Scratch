from transformers import ViTModel
from 01_ViT_Original import ViT
import torch

model = ViT(in_channels=3, patch_size=16, emb_size=768, img_size=224, num_heads=12, mlp_dim=3072,num_layers=12, num_classes=10)

hf_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", add_pooling_layer=False)


with torch.no_grad():

    model.patch_embedding.proj.weight.copy_(hf_model.embeddings.patch_embeddings.projection.weight)
    model.patch_embedding.proj.bias.copy_(hf_model.embeddings.patch_embeddings.projection.bias)
    model.cls_token.copy_(hf_model.embeddings.cls_token)
    model.pos_embedding.copy_(hf_model.embeddings.position_embeddings)

    for i in range(12):

        custom_block = model.transformer_blocks[i]
        hf_block = hf_model.encoder.layer[i]
        custom_block.norm1.weight.copy_(hf_block.layernorm_before.weight)
        custom_block.norm1.bias.copy_(hf_block.layernorm_before.bias)
        custom_block.attn.q_proj.weight.copy_(hf_block.attention.attention.query.weight)
        custom_block.attn.q_proj.bias.copy_(hf_block.attention.attention.query.bias)
        custom_block.attn.k_proj.weight.copy_(hf_block.attention.attention.key.weight)
        custom_block.attn.k_proj.bias.copy_(hf_block.attention.attention.key.bias)
        custom_block.attn.v_proj.weight.copy_(hf_block.attention.attention.value.weight)
        custom_block.attn.v_proj.bias.copy_(hf_block.attention.attention.value.bias)
        custom_block.attn.out_proj.weight.copy_(hf_block.attention.output.dense.weight)
        custom_block.attn.out_proj.bias.copy_(hf_block.attention.output.dense.bias)
        custom_block.norm2.weight.copy_(hf_block.layernorm_after.weight)
        custom_block.norm2.bias.copy_(hf_block.layernorm_after.bias)
        custom_block.mlp.fc1.weight.copy_(hf_block.intermediate.dense.weight)
        custom_block.mlp.fc1.bias.copy_(hf_block.intermediate.dense.bias)
        custom_block.mlp.fc2.weight.copy_(hf_block.output.dense.weight)
        custom_block.mlp.fc2.bias.copy_(hf_block.output.dense.bias)
    
    model.norm.weight.copy_(hf_model.layernorm.weight)
    model.norm.bias.copy_(hf_model.layernorm.bias)
