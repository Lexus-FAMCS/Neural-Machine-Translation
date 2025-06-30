# Neural Machine Translation 🇬🇧➡️🇷🇺

An **Encoder-Decoder Transformer-based** translator from English to Russian.

---

## 📚 Description

This repository contains a Transformer implementation for the machine translation task.  
The model is trained on a dataset consisting of sentence pairs **English ↔ Russian**.

---

## 📂 Dataset

The dataset is located in the `data/raw_data` folder and consists of two files:  
- `data.en` — English sentences
- `data.ru` — Russian sentences

Each line in these files is a translation pair.

---

## 🗂️ Preparing the dataset

For training you will need:
- Tokenizers (English and Russian)
- Files split into train/val/test in the format:
  `train.en`, `train.ru`, `val.en`, `val.ru`, `test.en`, `test.ru`

---

## 🚀 Quick Start

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Lexus-FAMCS/Neural-Machine-Translation.git
cd Neural-Machine-Translation
```

### 2️⃣ Create an environment and install dependencies

```bash
python3 -m venv your_env
source your_env/bin/activate
pip install -r requirements.txt
```

---

## 3️⃣ Download the pretrained model weights

Download the pretrained model weights from the [link](https://drive.google.com/file/d/1avuO6Tz3G4Xy-8PfA0-ltIoF2A9UfYM4/view?usp=drive_link) and run translation generation:

```bash
python3 translate.py --model_path path_to_pretrained_model --tokenizers_path your_repo/data/processed_data --model_max_len 64 --device your_device --num_layers 8 --d_model 1024 --num_heads 8 --d_hid 4096
```

You can view the training results of the pretrained model with:
```bash
tensorboard --logdir your_repo/runs/pretrained_model
```

---

## ⚙️ Advanced: Train on your own dataset

You can use the existing processed data and tokenizers in the `data/processed_data` folder.

Or, if you have your own dataset with English ↔ Russian sentence pairs, you can preprocess it yourself:

```bash
python3 data_prepare.py --data_dir your_repo/data/your_raw_data --output_dir your_repo/data/your_processed_data
```

Run training:
```bash
python3 train.py --data_dir your_repo/data/your_processed_data --batch_size 64 --epochs 10 --learning_rate 1e-4 --device your_device --num_layers 6 --d_model 512 --num_heads 8 --d_hid 2048 --dropout 0.1 --train_log_interval 150 --val_log_interval 500 --output_dir your_output_dir
```

The trained model and logs will be saved to `your_repo/runs/your_output_dir`.

Launch TensorBoard to view training progress: 
```bash
tensorboard --logdir your_repo/runs/your_output_dir
```

---

## 🔍 More about each script

For help on each script:
```bash
python3 script.py --help
```

---

## ✅ Results

Saved in the `translations.txt` file.
---

## ⚡ Conclusions

The model struggles with individual words and short phrases (3–5 words).
I believe this is due to the fact that it was trained on sentences of mean length ~7 words.
In addition, the model has only ~265M parameters and was trained on a small dataset.
---

## 🔭 Next steps

- Increase the model size
- Use a larger and more diverse dataset
- Pre-train the model on dictionary word pairs

---

📄 More about the architecture:: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
