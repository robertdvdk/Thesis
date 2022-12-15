"""
Author: Robert van der Klis

This module finetunes a pretrained DNABERT model using the Huggingface
framework.

Usage: python3 huggingface.py
"""

# Import statements
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, load_metric
import numpy as np
import os
import torch.nn.functional

# Function definitions
class BCEWithLogitsLossWeighted(torch.nn.Module):
    """From: https://stackoverflow.com/questions/57021620/
    how-to-calculate-unbalanced-weights-for-bcewithlogitsloss-in-pytorch"""
    def __init__(self, weight, *args, **kwargs):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(*args, **kwargs, reduction="none")
        self.weight = weight

    def forward(self, logits, labels):
        loss = self.bce(logits, labels)
        binary_labels = labels.bool()
        loss[binary_labels] *= labels[binary_labels] * self.weight
        return torch.mean(loss)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        balance = 1/torch.mean(labels).item()
        loss_fct = BCEWithLogitsLossWeighted(balance)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

def seq2kmer(seq, k):
    """From DNABERT codebase
    Convert original sequence to kmers

    Args:
        seq::str
            original sequence
        k::int
            the length of the kmers

    Returns:
        kmers::str
            kmers separated by space
    """
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers

def domain_occurrence_cutoff(domains_dict, cutoff):
    """Removes domains from the dataset if they occur less than a specified
        number of times

    Args:
        domains_dict::dict
            dictionary containing all processed data:
                {UniProt ID: (promoter sequence: [domains])}

    Returns:
        (domains_dict, num_domains)::tuple
            tuple with dictionary containing all processed data:
                {UniProt ID: (promoter sequence: [domains])} and the total
                number of unique domains
    """
    count_dict = {}
    for uniprot_id, (promseq, domains) in domains_dict.items():
        for domain in domains:
            count_dict.setdefault(domain, [0, []])
            count_dict[domain][0] += 1
            count_dict[domain][1].append(uniprot_id)
    total = 0
    lower = 0
    for domain, (count, uniprot_ids) in count_dict.items():
        total += count
        if count < cutoff:
            lower += count
            for uniprot_id in uniprot_ids:
                if len(domains_dict[uniprot_id][1]) <= 1:
                    # 1 domain or fewer left: delete entire protein
                    del domains_dict[uniprot_id]
                else:
                    # Delete only the particular domain
                    index = domains_dict[uniprot_id][1].index(domain)
                    del domains_dict[uniprot_id][1][index]
    num_datapoints = len(domains_dict)

    print(f'Domains deleted: {lower} out of {total}')
    print(f'Average n domains per promoter: '
          f'{round((total - lower)/num_datapoints, 2)}')
    return domains_dict, num_datapoints

def load_domains(processed_dir, begin_base, end_base):
    """Loads promoter-domain pairs from the directory with processed files

    Args:
        processed_dir::str
            the directory containing the processed files
        begin_base::int
            the base at which to begin (1000 is Transcription Start Site)
        end_base::int
            the base at which to end (1000 is Transcription Start Site)

    Returns:
        domains_dict::dict
            dictionary containing all processed data:
                {UniProt ID: (promoter sequence: [domains])}
    """
    domains_dict = {}
    for file in os.listdir(f'{processed_dir}/domains{begin_base}-{end_base}'):
        with open(f'{processed_dir}/domains{begin_base}-{end_base}/{file}') \
                as fopen:
            curr_upid, curr_promseq, curr_domains = '', '', []
            for line in fopen:
                words = line.split()
                if words[0].startswith('UPID'):
                    curr_upid = words[1]
                if words[0].startswith('PROMSEQ'):
                    curr_promseq = words[1]
                if words[0].startswith('DOMAINS'):
                    curr_domains = words[1:]
                if curr_upid and curr_promseq and curr_domains:
                    domains_dict[curr_upid] = (curr_promseq, curr_domains)
                    curr_upid = ''
                    curr_promseq = ''
                    curr_domains = []
    return domains_dict

def create_dataset():
    """Make dataset in correct format for Huggingface framework

    Args:
        None

    Returns:
        trainset::dict
            training set in format used by Huggingface
        testset::dict
            test set in format used by Huggingface

    """
    processed_dir = '/home/klis004/nbk_lustre/processed_data'
    begin_base = 500
    end_base = 1000
    dataset = load_domains(processed_dir, begin_base, end_base)
    cutoff = 10
    dataset, numdatapoints = domain_occurrence_cutoff(dataset, cutoff)
    index = round(0.80*len(list(dataset.keys())))
    traindataset = list(dataset.keys())[:index]
    testdataset = list(dataset.keys())[index:]
    domain_ids = {}
    curr_id = 0
    trainset = {'text': [], 'labels': []}
    testset = {'text': [], 'labels': []}

    # Add all domains to a set to be able to make tensor of length n_domains
    domain_set = set()
    for promseq, domains in dataset.values():
        domain_set.update(domains)

    n_domains = len(domain_set)

    print(n_domains)
    for key in traindataset:
        promseq, domains = dataset[key]
        kmers = seq2kmer(promseq, 6)
        encoded_domains = []
        for domain in domains:
            if domain not in domain_ids:
                domain_ids[domain] = curr_id
                encoded_domains.append(curr_id)
                curr_id += 1
            else:
                encoded_domains.append(domain_ids[domain])
        onehot = torch.zeros(n_domains)
        onehot[encoded_domains] = 1.0

        # Huggingface format
        trainset['text'].append(kmers)
        trainset['labels'].append(onehot.tolist())

    for key in testdataset:
        promseq, domains = dataset[key]
        kmers = seq2kmer(promseq, 6)
        encoded_domains = []
        for domain in domains:
            if domain not in domain_ids:
                domain_ids[domain] = curr_id
                # encoded_domains = curr_id
                encoded_domains.append(curr_id)
                curr_id += 1
            else:
                # encoded_domains = curr_id
                encoded_domains.append(domain_ids[domain])
        onehot = torch.zeros(n_domains)
        onehot[encoded_domains] = 1.0

        # Huggingface format
        testset['text'].append(kmers)
        testset['labels'].append(onehot.tolist())

    return trainset, testset

def main():
    trainset, testset = create_dataset()
    tokenizer = AutoTokenizer.from_pretrained("armheb/DNA_bert_6")
    #cutoff 1500: 2 labels, cutoff 100: 238 labels, cutoff 10: 3681 labels
    model = AutoModelForSequenceClassification.from_pretrained("/home/klis004/nbk_lustre/dnabert/cutoff_10/epochs_20", num_labels=3681, problem_type="multi_label_classification")
    train_dataset = Dataset.from_dict(trainset)
    test_dataset = Dataset.from_dict(testset)

    # metric = load_metric('precision', 'recall', 'accuracy')

    def tokenize_function(examples):
        """Huggingface tokenizer"""
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)

    def compute_metrics(eval_pred):
        """Custom accuracy metric, same as used in CNN"""
        logits, labels = eval_pred
        predictions = np.where(logits < 0, 0, 1)
        TP = np.sum(
            np.where(((predictions == 1) & (labels == 1)), 1, 0))
        FP = np.sum(
            np.where(((predictions == 1) & (labels == 0)), 1, 0))
        FN = np.sum(
            np.where(((predictions == 0) & (labels == 1)), 1, 0))
        TN = np.sum(
            np.where(((predictions == 0) & (labels == 0)), 1, 0))
        acc = (TP + TN) / (TP + FP + TN + FN)
        if TP + FP != 0:
            precision = round(TP / (TP + FP), 3)
        else:
            precision = 0
        if TP + FN != 0:
            recall = round(TP / (TP + FN), 3)
        else:
            recall = 0
        if TN + FP != 0:
            specificity = round(TN / (TN + FP), 3)
        else:
            specificity = 0
        print(f'Accuracy: {acc}, Precision: {precision}, Recall: {recall}, Specificity: {specificity}')
        return {'accuracy': acc, 'precision': precision, 'recall': recall, 'specificity': specificity}
        # acc = np.mean(np.where(predictions == labels, 1, 0))
        # precision = np.mean(np.where((predictions == 1) & (labels == 1), 1, 0))
        # recall = np.mean(np.where((predictions == )))
        # acc = torch.Tensor([acc])
        # return {'accuracy': acc, 'precision': }

    print(train_tokenized)
    training_args = TrainingArguments(output_dir="/home/klis004/nbk_lustre/dnabert/cutoff_10/epochs_20", evaluation_strategy="epoch", num_train_epochs=20)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics,
    )
    # trainer.train()
    # trainer.evaluate()

if __name__ == "__main__":
    main()