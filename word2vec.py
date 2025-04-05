import nltk
from nltk.corpus import stopwords
import datasets
from datasets import DatasetDict
import re
import string
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import Counter
from tqdm import tqdm



class SkipGramDataset(torch.utils.data.Dataset):

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)



#! SkipGram Modell architecture:
#todo Eingabe: Index des Center-Worts
#todo Erzeuge: Embedding-Vektor für das Center-Wort
#todo Vergleiche: Mit allen Kontext-Embeddings (Dot-Produkt)
#todo Berechne: Wahrscheinlichkeiten (Softmax)
#todo Loss: Cross-Entropy

class SkipGramModel(nn.Module):

    def __init__(self, my_dim, vocab_size):
        super(SkipGramModel, self).__init__()
        self.input_emb = nn.Embedding(vocab_size, my_dim)
        self.output_emb = nn.Embedding(vocab_size, my_dim)

        def forward(self, center_words):
            center_emb = self.input_emb(center_words)
            scores = torch.matmul(center_emb, self.output_emb.weight.t())

            log_probs = F.log_softmax(scores, dim=1)

            return log_probs








def load_text(selection):
    if selection == "imdb":
        text = datasets.load_dataset("stanfordnlp/imdb")
    elif selection == "text8":
        text = datasets.load_dataset("afmck/text8")
    elif selection == "wikitext103":
        text = datasets.load_dataset("wikitext-103-v1")

    # load stopwords
    nltk.download('stopwords')
    stop_en = stopwords.words("english")
    return text, stop_en




def preprocessing(data, stopwords):
    text = data["train"][0]["text"]
    text_clean = re.sub(r"<[^>]+>", "", text)
    text_clean = re.sub(f"[{re.escape(string.punctuation)}]", "", text_clean)
    tokens = [t for t in text_clean.lower().split() if t not in stopwords and len(t) > 1]

    filtered_tokens = []
    for word in tokens: 
        if word not in stopwords and len(word) > 1:
            filtered_tokens.append(word)

    # count the words and build dicts
    counter = Counter(filtered_tokens)
    most_common = counter.most_common(2**14)
    word_to_index = {word: i for i, (word, _) in enumerate(most_common)}
    index_to_word = {i: word for word, i in word_to_index.items()}
    return filtered_tokens, word_to_index, index_to_word



def build_pairs(tokens, word_to_index, window_size=2):
    pairs = []
    for i in range(len(tokens)):
        center_word = tokens[i]
        if center_word not in word_to_index:
            continue
        center_idx = word_to_index[center_word]

        for j in range(i - window_size, i + window_size + 1):
            if j == i or j < 0 or j >= len(tokens):
                continue
            context_word = tokens[j]
            if context_word not in word_to_index:
                continue
            context_idx = word_to_index[context_word]
            pairs.append((center_idx, context_idx))
    return pairs



def main():
    # Get one of the texts
    valid_input = ["imdb", "text8", "wikitext103"]
    while True:
        # text_select = input("Which dataset do you want to load? (imdb, text8, wikitext103): ")
        text_select = "imdb" #! for testing
        if text_select in valid_input:
            break
        print("Not a valid input. Try again.\n")

    dataset, stopwords = load_text(text_select)
    if dataset is not None:
        print("Text loaded successfully!\n")
    else:
        print("Error loading text")

    # Preprocessing of the text
    #? Machen Stopwords überhaupt Sinn wenn man Wörter mit Kontext betrachtet???
    tokens, word_to_index, index_to_word = preprocessing(dataset, stopwords)
    
    # Build the pairs for skipgram   
    pairs = build_pairs(tokens, word_to_index)
    print(f"Generated {len(pairs)} training pairs.")

    # Build a dataset for pairs
    dataset = SkipGramDataset(pairs)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    print(f"Dataset size: {len(dataset)} samples.")















if __name__ == "__main__":
    main()