# Transformer Example

Example with transformer model.

```python
import torch
import torch.nn as nn
import tensight
from transformers import BertModel

# Wrap transformer
base_model = BertModel.from_pretrained('bert-base-uncased')
classifier = nn.Linear(768, 2)

class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = base_model
        self.classifier = classifier
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)

model = TransformerClassifier()
watched_model = tensight.watch(model)

# Pre-check before fine-tuning
report = tensight.pre_check(
    model, 
    train_loader, 
    config={'lr': 2e-5, 'batch_size': 16}
)
```

