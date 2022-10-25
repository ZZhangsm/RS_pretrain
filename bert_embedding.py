import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
import pandas as pd
import gc

gc.collect()
torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

data = []
with open('./dataset/item_reviews.csv', 'r', encoding='utf-8-sig') as f_input:
# with open('./dataset/user_reviews.csv', 'r', encoding='utf-8-sig') as f_input:
    for line in f_input:
        l = line.strip().split(',')
        l = l[1:]
        # 如果l中元素个数大于1，合并l中的元素，以空格分隔
        if len(l) > 1:
            data.append(' '.join(l[1:]))
        else:
            data.append(l[0])
        # data.append(list(line.strip().split(',')))
text = data

# load the pre-trained model
BERT_PATH = './bert_localpath'
output_dim = 64
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
# model_config = BertConfig.from_pretrained(BERT_PATH)
model = BertModel.from_pretrained(BERT_PATH).to(device)
fc = nn.Linear(768, output_dim).to(device)

bert_out = []
dataloader = DataLoader(text, batch_size=16, shuffle=False)
for batch in dataloader:
    with torch.no_grad():
        encoded_input = tokenizer(batch, return_tensors='pt', max_length=128,
                              padding=True, truncation=True).to(device)
        output = model(**encoded_input).pooler_output
        bert_out.append(fc(output))
        print(len(bert_out), "has been processed")
bert_out = torch.cat(bert_out, dim=0)
print(bert_out.shape)

# save the bert embedding
# np.save('./dataset/user_embed.npy', bert_out.cpu().detach().numpy())
np.save('./dataset/item_embed.npy', bert_out.cpu().detach().numpy())
