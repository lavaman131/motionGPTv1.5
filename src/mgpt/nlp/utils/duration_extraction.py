import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import pytorch_lightning as pl
import torch.utils.data
from mgpt.nlp import PRECOMPUTED_DIR
import pickle
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import DistilBertTokenizer, DistilBertModel
import torch.nn.functional as F

UNK_TOKEN = "<unk>"
UNK_INDEX = 0
torch.set_float32_matmul_precision("high")


class MotionDurationModel(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.bert = DistilBertModel.from_pretrained(
            "distilbert/distilbert-base-uncased"
        )
        self.lstm = nn.LSTM(self.bert.config.hidden_size, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0]
        x, _ = self.lstm(last_hidden_states)
        x = self.fc(x)
        # enforce non-negative duration
        x = nn.ReLU()(x)
        return x.squeeze(-1)

    def calculate_loss(self, preds, labels, lengths):
        mask = torch.zeros_like(preds, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        masked_preds = preds[mask]
        masked_labels = labels[mask]
        loss = nn.MSELoss()(masked_preds, masked_labels)
        return loss

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        lengths = batch["lengths"]
        preds = self(input_ids, attention_mask)
        loss = self.calculate_loss(preds, labels, lengths)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        lengths = batch["lengths"]
        preds = self(input_ids, attention_mask)
        self.eval()
        with torch.no_grad():
            loss = self.calculate_loss(preds, labels, lengths)
        self.log("val_loss", loss, prog_bar=True)
        self.train()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)


def collate_fn(batch):
    # Separate the input features and labels
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = [item["labels"] for item in batch]

    # Determine the maximum label length in the batch
    max_length = input_ids[0].size(0)

    # Pad the labels and store the original label lengths
    padded_labels = []
    lengths = []
    for label in labels:
        padded_label = label + [0] * (max_length - len(label))
        padded_labels.append(padded_label)
        lengths.append(len(label))

    padded_labels = torch.tensor(padded_labels)
    lengths = torch.tensor(lengths)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": padded_labels,
        "lengths": lengths,
    }


def tokenize_and_preserve_labels(
    sentence, text_labels, tokenizer, add_special_tokens=True
):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    if add_special_tokens:
        tokenized_sentence = (
            [f"{tokenizer.cls_token}"] + tokenized_sentence + [f"{tokenizer.sep_token}"]
        )
        labels = [0] + labels + [0]

    return tokenized_sentence, labels


def prepare_data(stats):
    phrases = stats["action_sequences"]
    duration_labels = stats["duration_sequences"]

    # Tokenize the phrases and create a mapping
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert/distilbert-base-uncased"
    )

    tokenized_texts = []
    tokenized_labels = []
    for phrase, label in zip(phrases, duration_labels):
        sentence = [stats["id_to_action"][action] for action in phrase]
        tokenized_sentence, labels = tokenize_and_preserve_labels(
            sentence, label, tokenizer
        )
        assert len(tokenized_sentence) == len(labels)
        tokenized_texts.append(tokenized_sentence)
        tokenized_labels.append(labels)

    output = tokenizer(
        tokenized_texts,
        is_split_into_words=True,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )

    dataset = [
        {
            "input_ids": output["input_ids"][i],
            "attention_mask": output["attention_mask"][i],
            "labels": tokenized_labels[i],
        }
        for i in range(len(tokenized_labels))
    ]

    return dataset


def main():
    with open(PRECOMPUTED_DIR.joinpath("babel_action_stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    bs = 32
    model = MotionDurationModel()

    dataset = prepare_data(stats)

    train_frac, val_frac = 0.8, 0.2
    # default collate and splits into train and val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_frac, val_frac]
    )

    # Training
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, collate_fn=collate_fn, num_workers=4
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=bs, collate_fn=collate_fn, num_workers=4
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=PRECOMPUTED_DIR.joinpath("duration_extraction_model"),
            monitor="val_loss",
        )
    ]

    # Training
    trainer = pl.Trainer(
        max_epochs=20,
        callbacks=callbacks,
        log_every_n_steps=5,
        check_val_every_n_epoch=5,
        precision="bf16-mixed",
    )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    pl.seed_everything(42)
    main()
