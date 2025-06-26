## Environment setup
1. create a new environment using conda or pip (We use Python 3.8.10)
2. ```pip install -r requirements.txt```
## Download Data
The three datasets (CMU-MOSI, CMU-MOSEI, and CH-SIMS) are available from this link: https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk

## Data Directory
To run our preprocessing and training codes directly, please put the necessary files from downloaded data in separate folders as described below.

```
/data/
    mosi/
        raw/
        label.csv
    mosei/
        raw/
        label.csv
    sims/
        raw/
        label.csv
```

## Train
```
python run.py   （以下为默认）
    --seed 1
    --batch_size 8
    --lr 5e-6
    --dataset mosi
    --num_hidden_layers 5
    --use_context True
    --use_attnFusion True
    --use_cme True
```