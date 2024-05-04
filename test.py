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

def test(model:MultiPETransformer, src_text):
    src_tokens = tokenizer.batch_encode_plus(
                [src_text], add_special_tokens=True, padding=True, return_tensors='pt')['input_ids'].to("cuda")
    
    start_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    end_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    tgt_len = 128
    out_tokens = torch.tensor([[start_id]],dtype=torch.int).cuda()
    model.eval()
    with torch.no_grad():
        for i in range(tgt_len):
            out = model(src_tokens, out_tokens)
            out_prob = model.predictor(out[-1, :]).transpose(0,1).contiguous()
            next_out_token = torch.argmax(out_prob, dim=0).unsqueeze(0)
            #import pdb; pdb.set_trace()
            out_tokens = torch.concat([out_tokens, next_out_token],dim=1)
            
            if next_out_token.item() == end_id:
                break

        out_text = tokenizer.decode(out_tokens.squeeze(), skip_special_tokens=True)

    return out_text

if __name__ == "__main__":
    wordVec_dim = 128
    
    pe_type = 'sinpe'
    model = MultiPETransformer(pe_type, d_model=wordVec_dim,nhead=8).cuda()
    model.load_state_dict(torch.load(f"./checkpoints/{pe_type}/{pe_type}_model_60000.pth")['model'])

    src_text = "It's a sunny day."
    out_text = test(model, src_text)
    print("fr:", out_text)

