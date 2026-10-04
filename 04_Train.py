import torch
from transformers import Trainer, TrainingArguments
import accelerate
from Load_Pretrained_Weights import model
from Fine_Tune import train_dataset, val_dataset

def collate_fn(features):

    images = torch.stack([feature[0] for feature in features])
    labels = torch.tensor([feature[1] for feature in features])
    return {"x": images, "labels": labels}

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=8,
    max_steps=10_000,
    learning_rate=0.01,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    optim="sgd",
    optim_args={"momentum": 0.9},
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    logging_steps=100,
    seed=42,
    remove_unused_columns=False,
    report_to="none")

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=collate_fn)

trainer.train()
torch.save(model.state_dict(), "vit_finetuned.pth")
