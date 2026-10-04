import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from ViT_Original import ViT
from Fine_Tune import test_loader
import matplotlib.pyplot as plt
import seaborn as sns


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "vit_finetuned.pth"
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

model = ViT(in_channels=3, patch_size=16, emb_size=768, img_size=224, num_heads=12, mlp_dim=3072, num_layers=12, num_classes=10)

state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:
        images = images.to(DEVICE, non_blocking=True)
        outputs = model(images)
        logits = outputs["logits"]
        predictions = logits.argmax(dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

print("Test Results:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("CIFAR-10 Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()