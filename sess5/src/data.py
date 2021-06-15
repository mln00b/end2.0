import random
import numpy as np
import torch, torchtext
from torchtext.legacy import data

from tqdm import tqdm
from collections import Counter


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def _clean_phrase_labels(phrase_labels):
  cleaned = dict()
  for k, v in phrase_labels.items():
    if len(list(set(k.strip()))) < 4: continue  # remove if string is too short
    if not any(c.isalpha() for c in k): continue  # remove if no alphabets
    cleaned[k] = v
  return cleaned


def _get_sentence_labels_from_phrases(sentences, phrase_labels):
    def float_val_to_bucket(val):
        if val <= 0.2: return 1
        elif val <=0.4: return 2
        elif val <= 0.6: return 3
        elif val <= 0.8: return 4
        else: return 5
    
    # TODO: This looks highly in-efficient
    sentence_labels = dict()
    skipped_sentences = []
    for sentence in tqdm(sentences):
        sentiments = []
        for k in phrase_labels:
            if k in sentence: sentiments.append(phrase_labels[k])
        
        if sentiments:
            sentence_labels[sentence] = float_val_to_bucket(sum(sentiments)/len(sentiments))
        else:
            skipped_sentences.append(sentence) # skip because no phrase found
    
    if skipped_sentences:
        print("Skipped some sentences while cleaning")
        print("Len, sentences: ", len(skipped_sentences), skipped_sentences)
    
    return sentence_labels


def _get_sentence_labels(base_path):
    
    with open("stanfordSentimentTreebank/datasetSentences.txt") as f:
        sentences = [x.split("\t")[1].replace("\n","") for x in f.readlines()[1:]]
    
    with open("stanfordSentimentTreebank/dictionary.txt") as f:
        phrase_to_id = {x.split("|")[0].strip(): int(x.split("|")[1].strip()) for x in f.readlines()}
        id_to_phrase = {v: k for k, v in phrase_to_id.items()}

    with open("stanfordSentimentTreebank/sentiment_labels.txt") as f:
        raw_phrase_labels = {id_to_phrase[int(x.split("|")[0])]: float(x.split("|")[1]) for x in f.readlines()[1:]}
    
    print("Len of original phrase labels: ", len(raw_phrase_labels))
    phrase_labels = _clean_phrase_labels(raw_phrase_labels)
    print("Len of cleaned phrase labels: ", len(raw_phrase_labels))

    return _get_sentence_labels_from_phrases(sentences, phrase_labels)

def get_train_valid_dataset(base_path, train_split=0.7,val_split=0.3):
    sentence_labels = _get_sentence_labels(base_path)
    label_ct = Counter(list(sentence_labels.values()))
    print("Label distribution: ", label_ct.items())

    Review = data.Field(sequential = True, tokenize = 'spacy', batch_first =True, include_lengths=True)
    Label = data.LabelField(tokenize ='spacy', is_target=True, batch_first =True, sequential =False)

    fields = [('review', Review),('label',Label)]
    example = [data.Example.fromlist([k,v], fields) for k,v in sentence_labels.items()] 
    dataset = data.Dataset(example, fields)
    (train, valid) = dataset.split(split_ratio=[train_split, val_split], random_state=random.seed(SEED))

    print("Len of train, valid dataset: ", len(train), len(valid))
    return train, valid