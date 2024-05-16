# Positional Encodings in transformers

## Preparation
download wmt14 dataset in `./wmt`
```
conda create -n multiPE python=3.10
conda activate multiPE
```
```
pip install -r ./requirements.txt
```
## Usage

### Training
modify the `pe_type` inside `./train.py` and run
```
python train.py
```
you can see the logs by running
```
tensorboard --logdir "./logs"
```
during training in another terminal.
### Evalutions
evaluate bleu:
```
python evaluate_bleu.py
```
evaluate bert score:
```
python evaluate_bert_score.py
```

### Inference

```
python test.py
```

