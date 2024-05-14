import torch
from modules import tokenizer
from transformer_models import MultiPETransformer

def test(model:MultiPETransformer, src_text):
    src_tokens = tokenizer.batch_encode_plus(
                [src_text], add_special_tokens=True, padding=True, return_tensors='pt')['input_ids'].to("cuda")
    
    start_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    end_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    tgt_len = 512
    out_tokens = torch.tensor([[start_id]],dtype=torch.int).cuda()
    model.eval()
    memory = model.encode(src_tokens)
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        for i in range(tgt_len):
            out = model.decode(memory, out_tokens)
            out_prob = model.predictor(out[-1, :]).transpose(0,1).contiguous()
            next_out_token = torch.argmax(out_prob, dim=0).unsqueeze(0)
            #import pdb; pdb.set_trace()
            out_tokens = torch.concat([out_tokens, next_out_token],dim=1)
            
            if next_out_token.item() == end_id:
                break

        out_text = tokenizer.decode(out_tokens.squeeze(), skip_special_tokens=True)

    return out_text

if __name__ == "__main__":
    wordVec_dim = 64
    
    models = []

    pe_types = ['nonepe','sinpe','rpe','rope']
    for pe_type in pe_types:
        model = MultiPETransformer(pe_type, d_model=wordVec_dim).cuda()
        model.load_state_dict(torch.load(f"./checkpoints/{pe_type}/{pe_type}_model.pth")['model'])
        models.append(model)

    while(True):
        src_text = input("en: ")
        for model in models:
            out_text = test(model, src_text)
            print(f"{model.pe_type} fr:", out_text)
        print()

