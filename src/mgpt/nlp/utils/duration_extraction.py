import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import pytorch_lightning as pl
import torch.utils.data
from mgpt.nlp import PRECOMPUTED_DIR
import pickle
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from transformers import (
    get_linear_schedule_with_warmup,
    DataCollatorForTokenClassification,
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from typing import Dict, List
import torchmetrics

torch.set_float32_matmul_precision("high")


class MotionDurationModel(pl.LightningModule):
    def __init__(self, lr=2e-5):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.loss = torchmetrics.MeanSquaredError()
        self.val_loss = torchmetrics.MeanSquaredError()
        self.bert = AutoModel.from_pretrained("distilbert/distilbert-base-cased")
        self.lstm = nn.LSTM(self.bert.config.hidden_size, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0]
        x, _ = self.lstm(last_hidden_states)
        x = self.fc(x)
        # # enforce non-negative duration
        x = nn.ReLU()(x)
        return x.squeeze(-1)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        preds = self(input_ids, attention_mask)
        mask = labels != -100
        loss = self.loss(preds[mask], labels[mask])
        self.log("train_loss", self.loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        preds = self(input_ids, attention_mask)
        self.eval()
        mask = labels != -100
        with torch.no_grad():
            self.val_loss.update(preds[mask], labels[mask])
        self.log("val_loss", self.val_loss, on_step=True, prog_bar=True)
        self.train()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.trainer.max_steps,
        )
        return [optimizer], [scheduler]


class DataCollatorForTokenRegression(DataCollatorForTokenClassification):
    def __call__(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )

        no_labels_features = [
            {k: v for k, v in feature.items() if k != label_name}
            for feature in features
        ]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label)
                + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label))
                + to_list(label)
                for label in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.float32)
        return batch


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(tokens, all_labels, tokenizer):
    tokenized_inputs = tokenizer(tokens, truncation=True, is_split_into_words=True)
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def prepare_data(stats, tokenizer):
    phrases = stats["action_sequences"]
    duration_labels = stats["duration_sequences"]

    sentences = []
    all_labels = []
    for phrase, label in zip(phrases, duration_labels):
        sentence = [stats["id_to_action"][action] for action in phrase]
        sentences.append(sentence)
        all_labels.append(label)

    tokenized_inputs = tokenize_and_align_labels(sentences, all_labels, tokenizer)

    dataset = [
        {
            "input_ids": tokenized_inputs["input_ids"][i],
            "attention_mask": tokenized_inputs["attention_mask"][i],
            "labels": tokenized_inputs["labels"][i],
        }
        for i in range(len(tokenized_inputs["labels"]))
    ]

    return dataset


def main():
    with open(PRECOMPUTED_DIR.joinpath("babel_action_stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    bs = 32
    model = MotionDurationModel()

    # Tokenize the phrases and create a mapping
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased")

    dataset = prepare_data(stats, tokenizer)

    train_frac, val_frac = 0.8, 0.2
    # default collate and splits into train and val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_frac, val_frac]
    )

    collate_fn = DataCollatorForTokenRegression(tokenizer)

    # Training
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, collate_fn=collate_fn
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=bs, collate_fn=collate_fn
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=PRECOMPUTED_DIR.joinpath("duration_extraction_model"),
            monitor="val_loss",
        )
    ]

    # Training
    max_steps = 2000
    trainer = pl.Trainer(
        max_steps=max_steps,
        callbacks=callbacks,
        log_every_n_steps=5,
        check_val_every_n_epoch=5,
        precision="bf16-mixed",
    )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    pl.seed_everything(42)
    main()
