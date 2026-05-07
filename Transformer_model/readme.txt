Folder structure

Keep these in one folder:

translation_project/
├── train.py
├── model.py
├── dataset.py
├── config.py
├── dataset_train.csv(input parallel dataset)

After running, these will be created:

translation_project/
└── artifacts/
    ├── corpus/
    │   ├── hindi_corpus.txt
    │   └── malayalam_corpus.txt
    ├── tokenizers/
    │   ├── tokenizer_hindi.model
    │   ├── tokenizer_hindi.vocab
    │   ├── tokenizer_malayalam.model
    │   └── tokenizer_malayalam.vocab
    ├── weights/
    │   ├── tmodel_00.pt
    │   ├── tmodel_01.pt
    │   └── tmodel_latest.pt
    └── final/
        └── final_model.pt
        
Install packages
pip install torch pandas datasets tqdm sentencepiece

If not installed already:

pip install tensorboard
Run from terminal

Inside the project folder:

python3 train.py
