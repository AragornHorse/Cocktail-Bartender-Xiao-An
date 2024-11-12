import json
import numpy as np
from transformers import BertTokenizer, BertModel
import tqdm
import torch


ct_to_idx_pth = r"D:\Users\DELL\Desktop\datasets\cocktail\npy\ct2idx.json"
bert_pth = r"D:\Users\DELL\Desktop\models\bert"
des_pth = r"D:\Users\DELL\Desktop\datasets\cocktail\npy\describe.npy"

tokenizer = BertTokenizer.from_pretrained(bert_pth)
model = BertModel.from_pretrained(bert_pth)
model.eval()

with open(ct_to_idx_pth, 'r', encoding='utf-8') as f:
    ct_to_idx = json.load(f)

cts = list(ct_to_idx.keys())

rsts = []

with torch.no_grad():
    for ct in tqdm.tqdm(cts):
        encoded_input = tokenizer(ct, return_tensors='pt')
        output = model(**encoded_input).last_hidden_state[0, 0, :].cpu().numpy()
        rsts.append(output)

rst = np.stack(rsts)

np.save(des_pth, rst)



