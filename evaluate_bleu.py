from wsgiref import validate
import torch
from tqdm import tqdm
from modules import tokenizer
from transformer_models import MultiPETransformer
from dataloader import get_test_dataset
from torchtext.data.metrics import bleu_score

MAX_LEN = 512

def eval_models(models, dataset):
    for model in models:
        model:MultiPETransformer 
        bleu_total = 0
        bleu_avg = 0
        for iter, data in enumerate(tqdm(dataset, position=0, desc=f"evaluating model_{model.pe_type}",ncols=120)):
            src_text = data['en']
            tgt_text = data['fr']
            src_tokens = tokenizer.batch_encode_plus(
                src_text, add_special_tokens=True, padding=True, return_tensors='pt')['input_ids'].to("cuda")
            tgt_tokens = tokenizer.batch_encode_plus(
                tgt_text, add_special_tokens=True, padding=True, return_tensors='pt')['input_ids'].to("cuda")
            reference_text_tokens = tokenizer.tokenize(tgt_text[0])
            
            if (src_tokens.shape[1] > MAX_LEN):
                src_tokens = src_tokens[:, :MAX_LEN]
            if (tgt_tokens.shape[1] > MAX_LEN):
                tgt_tokens = tgt_tokens[:, :MAX_LEN]

            tgt_tokens_shifted_left = tgt_tokens[:,1:]
            tgt_tokens = tgt_tokens[:,:-1]

            start_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
            end_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
            

            out_tokens = torch.tensor([[start_id]],dtype=torch.int).cuda()
            
            for i in range(MAX_LEN):
                out = model(src_tokens, out_tokens)
                out_prob = model.predictor(out[-1, :]).transpose(0,1).contiguous()
                next_out_token = torch.argmax(out_prob, dim=0).unsqueeze(0)
                #import pdb; pdb.set_trace()
                out_tokens = torch.concat([out_tokens, next_out_token],dim=1)
                
                if next_out_token.item() == end_id:
                    break

            out_text = tokenizer.decode(out_tokens.squeeze(), skip_special_tokens=True)
            inference_text_tokens = tokenizer.tokenize(out_text)

            if len(inference_text_tokens) < len(reference_text_tokens):
                inference_text_tokens += [tokenizer.pad_token] * (
                    len(reference_text_tokens) - len(inference_text_tokens))
            elif len(inference_text_tokens) > len(reference_text_tokens):
                reference_text_tokens += [tokenizer.pad_token] * (
                    len(inference_text_tokens) - len(reference_text_tokens))


            
            bleu = bleu_score([inference_text_tokens], [[reference_text_tokens]])
            bleu_total += bleu
            bleu_avg = bleu_total / (iter + 1)
            # if iter % 10 == 0:
            #     print(bleu_avg)
        
        print(f"model {model.pe_type}'s bleu:", bleu_avg)
            

if __name__ == "__main__":
    wordVec_dim = 64
    
    models = []

    pe_types = ['nonepe','sinpe','rpe','rope']
    '''
    nonope: 0.02823680870810569
    sinpe: 0.08826489413242081
    rpe: 0.17756954479163692
    rope: 0.16575512542695434
    '''
    for pe_type in pe_types:
        model = MultiPETransformer(pe_type, d_model=wordVec_dim).cuda()
        model.load_state_dict(torch.load(f"./checkpoints/{pe_type}/{pe_type}_model.pth")['model'])
        model.eval()
        models.append(model)
    
    test_dataset = get_test_dataset()

    with torch.no_grad():
        eval_models(models, test_dataset)


