from wsgiref import validate
import torch
import torch.utils
import torch.utils.tensorboard
from tqdm import tqdm
from modules import tokenizer
from transformer_models import MultiPETransformer
from dataloader import get_test_dataset
from bert_score import BERTScorer
from transformers import BertModel
import matplotlib.pyplot as plt

model_name = 'bert-base-multilingual-cased'
bert_model = BertModel.from_pretrained(model_name).to('cuda')
scorer = BERTScorer(model_type=model_name, num_layers=4)

MAX_LEN = 512

def eval_models(models, dataset):
    colors = ['orange','y','g','b']
    
    for mi, model in enumerate(models):
        model:MultiPETransformer 
        bert_score_total = 0
        bert_score_avg = 0
        scatters_x = []
        scatters_y = []
        for iter, data in enumerate(tqdm(dataset, position=0, desc=f"evaluating model_{model.pe_type}",ncols=120)):
            src_text = data['en']
            tgt_text = data['fr']
            src_tokens = tokenizer.batch_encode_plus(
                src_text, add_special_tokens=True, padding=True, return_tensors='pt')['input_ids'].to("cuda")
            tgt_tokens = tokenizer.batch_encode_plus(
                tgt_text, add_special_tokens=True, padding=True, return_tensors='pt')['input_ids'].to("cuda")
            
            if (src_tokens.shape[1] > MAX_LEN):
                src_tokens = src_tokens[:, :MAX_LEN]
            if (tgt_tokens.shape[1] > MAX_LEN):
                tgt_tokens = tgt_tokens[:, :MAX_LEN]

            tgt_tokens_shifted_left = tgt_tokens[:,1:]
            tgt_tokens = tgt_tokens[:,:-1]

            start_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
            end_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
            

            out_tokens = torch.tensor([[start_id]],dtype=torch.int).cuda()

            memory = model.encode(src_tokens)

            for i in range(MAX_LEN):
                out = model.decode(memory, out_tokens)
                out_prob = model.predictor(out[-1, :]).transpose(0,1).contiguous()
                next_out_token = torch.argmax(out_prob, dim=0).unsqueeze(0)
               
                out_tokens = torch.concat([out_tokens, next_out_token],dim=1)
                
                if next_out_token.item() == end_id:
                    break

            out_text = tokenizer.decode(out_tokens.squeeze(), skip_special_tokens=True)


            
            bert_score = scorer.score([out_text], tgt_text)[2] # f1 score
            bert_score_total += bert_score
            bert_score_avg = bert_score_total / (iter + 1)
            scatters_x.append(mi)
            scatters_y.append(bert_score.item())
            
            if iter % 100 == 0:
                plt.scatter(scatters_x, scatters_y, c=colors[mi], alpha=0.01)
                plt.savefig('eval_logs/bert_scores.png')
                scatters_x.clear()
                scatters_y.clear()
                #plt.scatter([mi], [bert_score_avg], c='r',marker='+')
                #plt.savefig('eval_logs/bert_scores.png')
            # 
            #     print(bert_score_avg)
        
        print(f"model {model.pe_type}'s mean bert_score:", bert_score_avg)
        plt.scatter([mi], [bert_score_avg], c='r',marker='+')
        plt.savefig('eval_logs/bert_scores.png')
            

if __name__ == "__main__":
    wordVec_dim = 64
    
    models = []

    pe_types = ['nonepe','sinpe','rpe','rope']
    '''
    nonope: 0.6686
    sinpe: 0.7074
    rpe: 0.7584
    rope: 0.7500
    '''
    for pe_type in pe_types:
        model = MultiPETransformer(pe_type, d_model=wordVec_dim).cuda()
        model.load_state_dict(torch.load(f"./checkpoints/{pe_type}/{pe_type}_model.pth")['model'])
        model.eval()
        models.append(model)
    
    test_dataset = get_test_dataset()

    with torch.no_grad():
        eval_models(models, test_dataset)


