# Vision Transformer (ViT-B/16) from Scratch: Implementation & CIFAR-10 Fine-Tuning

A from-scratch PyTorch implementation of the Vision Transformer from the paper
[*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929)
(Dosovitskiy et al., 2020).

The architecture is written by hand (patch embedding, multi-head self-attention, MLP, encoder blocks) with no `timm` or
Hugging Face model classes. Pretrained ImageNet-21k weights from `google/vit-base-patch16-224-in21k` are then mapped
tensor by tensor into the custom model and fine-tuned on CIFAR-10 using the paper's fine-tuning recipe.

---

## Highlights

- **Faithful ViT-B/16 implementation:** pre-LayerNorm encoder, learnable `[CLS]` token, learnable 1D position embeddings, GELU MLP.
- **Manual pretrained-weight transfer:** every parameter of the custom model is copied explicitly from the Hugging Face checkpoint, which doubles as a check that the architecture matches the reference.
- **Paper-style fine-tuning recipe:** SGD with momentum, effective batch size 512, cosine schedule with warmup, 10k steps, gradient clipping.
- **Clean evaluation pipeline:** accuracy, macro precision/recall/F1, and a confusion matrix on the CIFAR-10 test set.

---

## Project Structure

```
.
├── ViT_Original.py             # ViT architecture implemented from scratch
├── Load_Pretrained_Weights.py  # Maps HF ViT-B/16 (ImageNet-21k) weights into the custom model
├── Fine_Tune.py                # CIFAR-10 datasets, transforms, stratified train/val split, dataloaders
├── Train.py                    # Fine-tuning with Hugging Face Trainer
├── Evaluate.py                 # Test-set metrics + confusion matrix
└── README.md
```

---

## Architecture

The model in `ViT_Original.py` follows the ViT-B/16 configuration:

| Component | Details |
|---|---|
| Input | 224 x 224 x 3 |
| Patch embedding | `Conv2d(3, 768, kernel=16, stride=16)` giving 196 patches |
| Tokens | 196 patch tokens + 1 learnable `[CLS]` token = 197 |
| Position embedding | Learnable, shape `(1, 197, 768)` |
| Encoder | 12 x pre-LN Transformer blocks |
| Attention | 12 heads, head dim 64, scaled dot-product, separate Q/K/V/output projections |
| MLP | `Linear(768 -> 3072)`, GELU, `Linear(3072 -> 768)` |
| Normalization | `LayerNorm` (eps = 1e-12, matching the HF reference) |
| Head | `Linear(768 -> num_classes)` on the final `[CLS]` token |
| Parameters | about 86M |

**Encoder block**

```
x = x + MHSA(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

**Classes**

| Class | Role |
|---|---|
| `PatchEmbedding` | Splits the image into 16x16 patches and linearly projects them via a strided convolution |
| `MultiHeadSelfAttention` | Manual scaled dot-product attention with per-head reshaping |
| `MLPBlock` | Two-layer feed-forward network with GELU |
| `TransformerBlock` | Pre-LN residual block combining attention and MLP |
| `ViT` | Full model; `forward(x, labels=None)` returns `{"logits"}` or `{"loss", "logits"}` |

The `forward` signature returning a dict with `loss` makes the model directly compatible with the Hugging Face `Trainer`.

---

## Pretrained Weight Transfer

`Load_Pretrained_Weights.py` loads `google/vit-base-patch16-224-in21k` and copies weights into the custom model:

| Custom model | Hugging Face `ViTModel` |
|---|---|
| `patch_embedding.proj` | `embeddings.patch_embeddings.projection` |
| `cls_token` | `embeddings.cls_token` |
| `pos_embedding` | `embeddings.position_embeddings` |
| `transformer_blocks[i].norm1` | `encoder.layer[i].layernorm_before` |
| `transformer_blocks[i].attn.{q,k,v}_proj` | `encoder.layer[i].attention.attention.{query,key,value}` |
| `transformer_blocks[i].attn.out_proj` | `encoder.layer[i].attention.output.dense` |
| `transformer_blocks[i].norm2` | `encoder.layer[i].layernorm_after` |
| `transformer_blocks[i].mlp.fc1` / `fc2` | `encoder.layer[i].intermediate.dense` / `output.dense` |
| `norm` | `layernorm` |

The classification head is **not** transferred. It is newly initialized for the 10 CIFAR-10 classes.

---

## Data Pipeline

`Fine_Tune.py` prepares CIFAR-10 for a model pretrained at 224 x 224:

- **Resize** 32 x 32 images to 224 x 224
- **Normalize** with mean = std = 0.5 per channel (matching the in21k checkpoint's preprocessing)
- **Augmentation** (training only): random horizontal flip
- **Split:** the 50,000 training images are split 90 / 10 into train / validation (45,000 / 5,000), **stratified** by class with `random_state=42`. Separate dataset objects are used so validation data gets no augmentation.
- **Test set:** the official 10,000 CIFAR-10 test images

---

## Training Configuration

`Train.py` uses the Hugging Face `Trainer` with hyperparameters that follow the fine-tuning setup in the ViT paper:

| Setting | Value |
|---|---|
| Optimizer | SGD, momentum 0.9 |
| Peak learning rate | 0.01 |
| LR schedule | Cosine decay, 500 warmup steps |
| Per-device batch size | 64 |
| Gradient accumulation | 8 steps (**effective batch size 512**) |
| Total steps | 10,000 |
| Gradient clipping | Global norm 1.0 (Trainer default) |
| Mixed precision | FP16 when CUDA is available |
| Evaluation / checkpointing | Every 500 steps, keep the best 2 checkpoints |
| Model selection | Best checkpoint by validation loss (`load_best_model_at_end`) |
| Seed | 42 |

The final weights are saved to `vit_finetuned.pth`.

---

## Installation

```bash
git clone https://github.com/alkiviadisss/Vision_Transformer_From_Scratch.git
cd Vision_Transformer_From_Scratch

pip install torch torchvision transformers accelerate scikit-learn matplotlib seaborn
```

A CUDA-capable GPU is strongly recommended. ViT-B/16 at 224 x 224 for 10k steps is slow on CPU.

---

## Usage

**1. Fine-tune** (downloads CIFAR-10 and the pretrained weights automatically on first run):

```bash
python Train.py
```

**2. Evaluate** on the CIFAR-10 test set:

```bash
python Evaluate.py
```

This prints the metrics and saves `confusion_matrix.png`.

---

## Results

> **Status: training and evaluation not yet run due to lack of GPU.**
> Fine-tuning ViT-B/16 (about 86M parameters) at 224x224 for 10,000 steps with an effective batch size of 512
> requires a GPU with substantial memory, which I did not have access to during development. The full pipeline
> (model, weight transfer, data loading, training, evaluation) is implemented, but no metrics have been
> produced yet, so they are reported here as "Pending".

### What has been verified
- The architecture is implemented from scratch and follows the ViT-B/16 configuration.
- All pretrained parameters map one-to-one from the Hugging Face checkpoint into the custom model.
- The training configuration follows the fine-tuning recipe from the paper.

### Reproducing the results
Run `python Train.py` followed by `python Evaluate.py` on a CUDA GPU. The metrics table below will be
filled in once a run completes.

| Metric | Score |
|---|---|
| Accuracy | Pending |
| Precision (macro) | Pending |
| Recall (macro) | Pending |
| F1 (macro) | Pending |

---

## Implementation Notes

- **Attention is written explicitly** (`softmax(QK^T / sqrt(d_k)) V`) instead of using `nn.MultiheadAttention` or `F.scaled_dot_product_attention`, to keep the math transparent.
- **Pre-LN** placement (LayerNorm before attention/MLP) matches the original ViT, with the final LayerNorm before the head.
- **LayerNorm eps = 1e-12** is used so outputs line up numerically with the Hugging Face reference model.
- **Dropout is omitted**, consistent with the paper's fine-tuning setup.
- **Separate Q/K/V projections** (rather than a fused QKV layer) make the weight mapping from Hugging Face one-to-one.

---

## Reference

```bibtex
Article by dosovitskiy2020vit,
Title: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,
Author: Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and others,
Journal: arXiv preprint arXiv:2010.11929,
Year: 2020
```
---

## Author

**Alkiviadis Agrogiannhs**  
Data Scientist & Machine Learning Engineer 
[LinkedIn](https://www.linkedin.com/in/alkiviadis-agrogiannhs/)
[Email](mailto:alkiviadisagrogiannhs@gmail.com)
[GitHub](https://github.com/alkiviadisss)

---
