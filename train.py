import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataloader import get_training_dataset, get_test_dataset
from modules import tokenizer
from transformer_models import MultiPETransformer
from torch.utils.tensorboard import SummaryWriter

LOSS_INTERVAL = 500
SAVE_INTERVAL = 20000
CACHE_INTERVAL = 50
TEST_SIZE = 50

def train(model: MultiPETransformer, train_dataset: DataLoader, writer: SummaryWriter):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    loss_sum = 0
    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    for iter, data in enumerate(tqdm(train_dataset, position=0, desc="training",ncols=150)):
        src_text = data['en']
        tgt_text = data['fr']
        src_tokens = tokenizer.batch_encode_plus(
            src_text, add_special_tokens=True, padding=True, return_tensors='pt')['input_ids'].to("cuda")
        tgt_tokens = tokenizer.batch_encode_plus(
            tgt_text, add_special_tokens=True, padding=True, return_tensors='pt')['input_ids'].to("cuda")
        
        tgt_tokens_shifted_left = tgt_tokens[:,1:]
        tgt_tokens = tgt_tokens[:,:-1]

        optimizer.zero_grad()
        output = model(src_tokens, tgt_tokens)
        output = model.predictor(output).transpose(0,1).contiguous()
        #import pdb; pdb.set_trace()
        loss = loss_fn(output.view(-1, output.size(-1)), tgt_tokens_shifted_left.reshape(-1))

        
        loss.backward()
        optimizer.step()

        if iter % CACHE_INTERVAL == 0:
            torch.cuda.empty_cache()

        if iter % LOSS_INTERVAL == 0:
            # test
            optimizer.zero_grad()
            with torch.no_grad():
                loss_sum = 0
                test_dataset = get_test_dataset()
                for test_iter, test_data in enumerate(test_dataset):
                    if test_iter >= TEST_SIZE:
                        break
                    
                    test_src_text = test_data['en']
                    test_tgt_text = test_data['fr']
                    test_src_tokens = tokenizer.batch_encode_plus(
                        test_src_text, add_special_tokens=True, padding=True, return_tensors='pt')['input_ids'].to("cuda")
                    test_tgt_tokens = tokenizer.batch_encode_plus(
                        test_tgt_text, add_special_tokens=True, padding=True, return_tensors='pt')['input_ids'].to("cuda")
                    
                    test_tgt_tokens_shifted_left = test_tgt_tokens[:, 1:]
                    test_tgt_tokens = test_tgt_tokens[:, :-1]

                    test_output = model(test_src_tokens, test_tgt_tokens)
                    test_output = model.predictor(test_output).transpose(0, 1).contiguous()
                    test_loss = loss_fn(test_output.view(-1, test_output.size(-1)), test_tgt_tokens_shifted_left.reshape(-1))
                    loss_sum += test_loss

                loss_avg = (loss_sum / TEST_SIZE)
                # print(f"iter {iter} loss: {loss_avg :.4f}")
                writer.add_scalar("loss", loss_avg, iter)
                test_dataset = None

        if iter > 0 and iter % SAVE_INTERVAL == 0:
            torch.save({'model': model.state_dict()}, f'./checkpoints/{model.pe_type}/{model.pe_type}_model_{iter}.pth')

    torch.save({'model': model.state_dict()}, f'./checkpoints/{model.pe_type}/{model.pe_type}_model.pth')

if __name__ == "__main__":
    wordVec_dim = 128
    batch_size = 8
    train_dataset = get_training_dataset(1, batch_size)
    
    pe_type = 'sinpe'
    model = MultiPETransformer(pe_type, d_model=wordVec_dim).cuda()
    writer = SummaryWriter(f"./logs/{pe_type}_B_{batch_size}_D_{wordVec_dim}")
    train(model, train_dataset, writer)
