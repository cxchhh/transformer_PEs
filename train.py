import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataloader import get_training_dataset
from modules import tokenizer
from transformer_models import VanillaTransformer, RPETransformer

def train(model, dataset: DataLoader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    loss_sum = 0
    for iter, data in enumerate(tqdm(dataset, position=0, desc="training")):
        src_text = data['en']
        tgt_text = data['fr']
        src_tokens = tokenizer.batch_encode_plus(
            src_text, add_special_tokens=True, padding=True, return_tensors='pt')['input_ids'].to("cuda")
        tgt_tokens = tokenizer.batch_encode_plus(
            tgt_text, add_special_tokens=True, padding=True, return_tensors='pt')['input_ids'].to("cuda")
        
        output = model(src_tokens, tgt_tokens).transpose(0,1).contiguous()
        loss = loss_fn(output.view(-1, output.size(-1)), tgt_tokens.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss
        if iter % 500 == 0:
            print(f"iter {iter} loss: {(loss_sum / 500) :.4f}")
            loss_sum = 0

if __name__ == "__main__":
    wordVec_dim = 16
    batch_size = 8
    train_dataset = get_training_dataset(0, batch_size)
    model = VanillaTransformer(d_model=wordVec_dim).cuda()
    train(model, train_dataset)
