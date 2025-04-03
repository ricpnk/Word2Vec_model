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
    tokens, word_to_index, index_to_word = preprocessing(dataset, stopwords)
    

    # Build the pairs for skipgram   
    pairs = build_pairs(tokens, word_to_index)
    print(f"Generated {len(pairs)} training pairs.")




if __name__ == "__main__":
    main()