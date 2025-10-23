# -*- coding: utf-8 -*-
"""sentment_analysis.ipynb


"""

!pip install huggingface_hub
import huggingface_hub

from huggingface_hub import HfApi

api = HfApi(token=ENV_VAR)

!pip install transformers
!pip install tokenizers

""" now lets make a sentece sentiment classifier"""

sentence = input()

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # already fine-tuned for sentiment

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

input = tokenizer(sentence,return_tensors = 'pt', padding=True)

with torch.no_grad():
  output = model(**input)

output.logits
probs = torch.nn.functional.softmax(output.logits,dim = -1 )
probs

pred_label = torch.argmax(probs, dim=1).item()

pred_label

"""In binary sentiment classification, it’s almost always:
               
0	   --->     Negative         
1	   --->     Positive

That’s the standard for most Hugging Face sentiment models (like distilbert-base-uncased-finetuned-sst-2-english).
"""

if pred_label == 1 :
  print('Positive Sentiment')

else:
  print('Negative Sentiment')

