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
from sklearn.manifold import TSNE
import pandas as pd
import time


#! Classes
class SkipGramDataset(torch.utils.data.Dataset):

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)

#todo Loss: Cross-Entropy
class SkipGramModel(nn.Module):

    def __init__(self, my_dim, vocab_size):
        super(SkipGramModel, self).__init__()
        self.input_emb = nn.Embedding(vocab_size, my_dim) # every word gets a vector with size my_dim
        self.output_emb = nn.Embedding(vocab_size, my_dim) # train on different embeddings to differentiate between representations

    def forward(self, center_words):
        center_emb = self.input_emb(center_words) # get the matrix with all the embeddings for center_words (dim: batch_size, my_dim)
        #? Calculate a score for every word in the vocab
        scores = torch.matmul(center_emb, self.output_emb.weight.t()) # dot product multiply both matrixes and .t() transponse
        #? calculates log-softmax for every score
        log_probs = F.log_softmax(scores, dim=1) # dim=1: every row | "log" because of .NLLLoss() we need log there

        return log_probs




#! Preprocessing
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

def preprocessing(data, stopwords, use_stopwords=True):
    text = data["train"][0]["text"]
    text_clean = re.sub(r"<[^>]+>", "", text)
    text_clean = re.sub(f"[{re.escape(string.punctuation)}]", "", text_clean)

    if use_stopwords:
        tokens = [t for t in text_clean.lower().split() if t not in stopwords and len(t) > 1]
    else:
        tokens = [t for t in text_clean.lower().split() if len(t) > 1]

    # count the words and build dicts
    counter = Counter(tokens)
    most_common = counter.most_common(2**14)
    word_to_index = {word: i for i, (word, _) in enumerate(most_common)}
    index_to_word = {i: word for word, i in word_to_index.items()}
    return tokens, word_to_index, index_to_word

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




#! Evaluation
def evaluate(model, word_to_index, index_to_word, word, topk = 10):
    model.eval() # set model to evaluation-mode

    embeddings = model.input_emb.weight.detach().cpu()

    word_idx = word_to_index[word]
    word_emb = embeddings[word_idx]

    similarities = F.cosine_similarity(word_emb.unsqueeze(0), embeddings, dim=1)

    topk_sim = torch.topk(similarities, k=topk + 1) # +1 cause of word itself
    top_idx = topk_sim.indices[1:]

    similar_words = [(index_to_word[idx.item()], similarities[idx].item()) for idx in top_idx]

    return similar_words

def tsne_scatterplot(model, word_to_index, index_to_word, num_words, topn):
    embeddings = model.input_emb.weight.detach().cpu() #? nochmal fragen

    if num_words < len(word_to_index):
        selected_indices = list(range(num_words))
    else:
        selected_indices = list(range(len(word_to_index)))

    selected_embeddings = embeddings[selected_indices]

    tsne_model = TSNE(n_components=2, random_state=42, n_iter=3000, perplexity=topn) #? random_state: for reproduction of code
    embeddings_2d = tsne_model.fit_transform(selected_embeddings.numpy())

    plt.figure(figsize=(12,10))
    for i, label in enumerate(selected_indices):
        x, y = embeddings_2d[i, 0], embeddings_2d[i, 1]
        plt.scatter(x, y)
        if label in index_to_word:
            plt.annotate(index_to_word[label], (x, y), textcoords="offset points", xytext=(2,2), ha='right', fontsize=8)

    plt.title("t-SNE visualization of word embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()




#! Training
def train_model(model, dataloader, num_epochs, device):
    # Define loss and optimizer
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(num_epochs):
        total_loss = 0

        if device == "mps":
            for center_batch, context_batch in dataloader:
                center_batch = center_batch.to(device)  #! changed to apple gpu!!!
                context_batch = context_batch.to(device)

                optimizer.zero_grad()
                log_probs = model(center_batch)
                loss = loss_function(log_probs, context_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
        else:
            for center_batch, context_batch in dataloader:
                optimizer.zero_grad()
                log_probs = model(center_batch)
                loss = loss_function(log_probs, context_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
    print()



#! Main function
def main():
    #todo set the parameters
    num_epochs = 10
    batch_size = 1024
    my_dim = 300
    performance_cut = 100000
    use_stopwords = False
    scatter_words = 150
    perplexity = 20


    #todo check if mps is available (gpu support for apple chips)
    if torch.backends.mps.is_available():
        answer = input("Do you want to use apple's gpu? (y|n): ")
        if answer == "y":
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    #todo text selection
    valid_input = ["imdb", "text8", "wikitext103"]
    while True:
        # text_select = input("Which dataset do you want to load? (imdb, text8, wikitext103): ")
        text_select = "text8"
        if text_select in valid_input:
            break
        print("Not a valid input. Try again.\n")

    dataset, stopwords = load_text(text_select)
    if dataset is not None:
        print("Text loaded successfully!\n")
    else:
        print("Error loading text")


    #todo Preprocessing of the text
    #? Machen Stopwords überhaupt Sinn wenn man Wörter mit Kontext betrachtet???
    tokens, word_to_index, index_to_word = preprocessing(dataset, stopwords, use_stopwords)
    
    tokens = tokens[:performance_cut] #! cut tokens for testing!! -> time


    #todo Build the pairs for skipgram   
    pairs = build_pairs(tokens, word_to_index)
    print(f"Generated {len(pairs)} training pairs.\n")


    #todo Build a dataset for pairs
    dataset = SkipGramDataset(pairs)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)


    #todo building the model
    vocab_size = len(word_to_index)
    if device == "mps":
        model = SkipGramModel(my_dim, vocab_size).to(device) #! changed to apple gpu!!
    else:
        model = SkipGramModel(my_dim, vocab_size)
    

    #todo train the model
    start_time = time.time()
    train_model(model, dataloader, num_epochs, device)
    stop_time = time.time()
    result_time = int(stop_time - start_time)
    print(f"Model trained with {num_epochs} epochs!\n It took {result_time} seconds.\n")


    #todo t-sne visualization
    tsne_scatterplot(model, word_to_index, index_to_word, scatter_words, perplexity)


    #todo test with a user input word
    #! implement a loop for endless inputting
    # test_word = input("Insert a word to get similarities: ").lower()
    test_word = "test"
    print("\nSimilar words:")
    if test_word not in word_to_index:
        print("Word not in vocabulary!")
    else:
        similar_words = evaluate(model, word_to_index, index_to_word, test_word)
        for word, score in similar_words:
            print(f"{word}: {score}")







if __name__ == "__main__":
    main()