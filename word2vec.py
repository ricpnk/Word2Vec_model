import nltk
from nltk.corpus import stopwords
import datasets
import re
import string
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
from sklearn.manifold import TSNE
import time
import os
from gensim.models import KeyedVectors
import random
import pickle

# Global variable
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
#! Main function
def main():
    # todo set seed for reproduction
    set_seed(42)

    # todo set the parameters
    model_type = "standard"  # or negative_sampling
    num_epochs = 5
    batch_size = 8192  # 1024 for testing 8192 for training
    my_dim = 300  #! dont change cause of testmodel.py
    performance_cut = 100000  # len of the training words
    use_stopwords = True
    # visualisation parameters
    scatter_words = 200
    perplexity = 35

    # todo user input for model type
    mode_input = input(
        "Which model do you want to use? (standard = 0, negative_sampling = 1): "
    )
    if mode_input == "0":
        model_type = "standard"
    elif mode_input == "1":
        model_type = "negative_sampling"
    else:
        print("Not a valid input. Using standard model.\n")
        model_type = "standard"

    # todo check if mps is available (gpu support for apple chips)
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("apple-gpu")
    if torch.cuda.is_available():
        devices.append("cuda")

    answer = input(f"Do you want to gpu? ({', '.join(devices)}): ")

    if answer == "cuda":
        device = torch.device("cuda")
    elif answer == "apple-gpu":
        device = torch.device("mps")
    else:
        print("Error: Invalid input using cpu")
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # todo text selection
    valid_input = ["imdb", "text8", "wikitext103"]
    while True:
        text_select = input(
            "Which dataset do you want to load? (imdb, text8, wikitext103): "
        )
        if text_select in valid_input:
            break
        print("Not a valid input. Try again.\n")

    dataset, stopwords = load_text(text_select)
    if dataset is not None:
        print("Text loaded successfully!\n")
    else:
        print("Error loading text")

    # todo Preprocessing of the text
    # ? Is using stopwords necessary for SkipGram?
    tokens, word_to_index, index_to_word = preprocessing(
        dataset, stopwords, use_stopwords, text_select
    )

    tokens = tokens[:performance_cut]  #! cut tokens for testing!! -> time

    # todo Build the pairs for skipgram
    pairs = build_pairs(tokens, word_to_index)
    print(f"Generated {len(pairs)} training pairs.\n")

    # todo Build a dataset for pairs
    dataset = SkipGramDataset(pairs)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    # todo building the model
    vocab_size = len(word_to_index)
    if model_type == "standard":
        model = SkipGramModel(my_dim, vocab_size, device).to(device)
    elif model_type == "negative_sampling":
        model = SkipGramModelNegativeSampling(my_dim, vocab_size, device).to(device)

    # todo train the model
    if model_type == "standard":
        start_time = time.time()
        final_loss = train_model(model, dataloader, num_epochs, device)
        stop_time = time.time()
        result_time = int(stop_time - start_time)
        print(
            f"Model used standard sampling and trained with {num_epochs} epochs!\n It took {result_time} seconds.\n"
        )
    elif model_type == "negative_sampling":
        start_time = time.time()
        final_loss = train_model_negative_sampling(
            model, dataloader, num_epochs, device, vocab_size
        )
        stop_time = time.time()
        result_time = int(stop_time - start_time)
        print(
            f"Model used negative sampling and trained with {num_epochs} epochs!\n It took {result_time} seconds.\n"
        )

    # todo save the model
    model_directory = f"outputs/{timestamp}/model"
    model_save_path = f"{model_directory}/trained_model_{timestamp}.pt"

    os.makedirs(model_directory, exist_ok=True)
    with open(f"{model_directory}/word_to_index.pkl", "wb") as f:
        pickle.dump(word_to_index, f)
    with open(f"{model_directory}/index_to_word.pkl", "wb") as f:
        pickle.dump(index_to_word, f)
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}\n")

    # todo t-sne visualization
    tsne_scatterplot(model, word_to_index, index_to_word, scatter_words, perplexity)

    # todo test with a user input word
    evaluate_gensim(model, index_to_word)

    # todo input all parameters and outputs to a .txt file
    export_output(
        num_epochs=num_epochs,
        batch_size=batch_size,
        my_dim=my_dim,
        performance_cut=performance_cut,
        use_stopwords=use_stopwords,
        scatter_words=scatter_words,
        perplexity=perplexity,
        training_time=result_time,
        final_loss=final_loss,
        dataset_name=text_select,
        model_type=model_type,
    )


#! Set the random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


#! Classes
class SkipGramDataset(torch.utils.data.Dataset):

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(
            context, dtype=torch.long
        )


class SkipGramModel(nn.Module):

    def __init__(self, my_dim, vocab_size, device):
        super(SkipGramModel, self).__init__()
        self.center_emb = nn.Embedding(
            vocab_size, my_dim, device=device
        )  # every word gets a vector with size my_dim
        self.context_emb = nn.Embedding(
            vocab_size, my_dim, device=device
        )  # train on different embeddings to differentiate between representations

    def forward(self, center_words):
        center_emb = self.center_emb(
            center_words
        )  # get the matrix with all the embeddings for center_words (dim: batch_size, my_dim)
        context_emb = (
            self.context_emb.weight
        )  # get the matrix with all the embeddings for context_words (dim: vocab_size, my_dim)
        # ? Calculate a score for every center word, with every word in the vocab
        probs = torch.matmul(
            center_emb, context_emb.t()
        )  # dot product multiply both matrixes and .t() transponse
        return probs


class SkipGramModelNegativeSampling(nn.Module):
    def __init__(self, my_dim, vocab_size, device):
        super(SkipGramModelNegativeSampling, self).__init__()
        self.center_emb = nn.Embedding(vocab_size, my_dim, device=device)
        self.context_emb = nn.Embedding(vocab_size, my_dim, device=device)

    def forward(self, center_words, pos_words, neg_words):
        center_emb = self.center_emb(center_words)
        pos_emb = self.context_emb(pos_words)
        neg_emb = self.context_emb(neg_words)
        # ? Calculate scores for positive and negative samples difference to the original SkipGramModel
        pos_probs = torch.sum(center_emb * pos_emb, dim=1)
        neg_probs = torch.bmm(
            neg_emb, center_emb.unsqueeze(2)
        ).squeeze()  # (batch_size, num_neg_samples)
        return pos_probs, neg_probs


#! Preprocessing
def load_text(selection):
    if selection == "imdb":
        text = datasets.load_dataset("stanfordnlp/imdb")
    elif selection == "text8":
        text = datasets.load_dataset("afmck/text8")
    elif selection == "wikitext103":
        text = datasets.load_dataset("wikitext", "wikitext-103-v1")

    # load stopwords
    nltk.download("stopwords")
    stop_en = stopwords.words("english")
    return text, stop_en


def preprocessing(data, stopwords, use_stopwords, text_select):
    if text_select == "text8":
        text = data["train"][0]["text"]
    elif text_select == "imdb":
        text_list = [example["text"] for example in data["train"]] + [
            example["text"] for example in data["test"]
        ]
        text = " ".join(text_list)
    elif text_select == "wikitext103":
        text_list = [example["text"] for example in data["train"]]
        text = " ".join(text_list)

    text_clean = re.sub(r"<[^>]+>", "", text)
    text_clean = re.sub(f"[{re.escape(string.punctuation)}]", "", text_clean)

    if use_stopwords:
        tokens = [
            t for t in text_clean.lower().split() if t not in stopwords and len(t) > 1
        ]
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
    for i in tqdm(range(len(tokens))):
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
def evaluate_gensim(model, index_to_word):
    model.eval()
    print("Gensim evaluation:")
    embeddings = (
        model.center_emb.weight.detach().cpu().numpy()
    )  # .weight: gets parameter | .detach(): cuts the tensor from comp | .cpu: copy's tensor to cpu from gpu | .numpy(): changes tensor to numpy array
    gensim_model = KeyedVectors(
        vector_size=embeddings.shape[1]
    )  # create a empty KeyedVectors object for gensim
    vocab_list = [
        index_to_word[i] for i in range(len(index_to_word))
    ]  # build a vocab list
    gensim_model.add_vectors(vocab_list, embeddings)  # add vectors to gensim model

    # testing on some words
    test_words = ["king", "queen", "war", "house", "germany"]
    for word in test_words:
        if word in gensim_model:
            print(f"\nWord: {word}\nMost similar words:")
            similar_words = gensim_model.most_similar(word, topn=5)
            for word, score in similar_words:
                print(f"{word}: {score:.4f}")
        else:
            print(f"Error: {word} not in vocabulary!")


#! Visualization
def tsne_scatterplot(model, word_to_index, index_to_word, num_words, perplexity_value):
    model.eval()
    embeddings = model.center_emb.weight.detach().cpu()

    if num_words < len(word_to_index):
        selected_indices = list(range(num_words))
    else:
        selected_indices = list(range(len(word_to_index)))

    selected_embeddings = embeddings[selected_indices]

    tsne_model = TSNE(
        n_components=2, random_state=42, max_iter=3000, perplexity=perplexity_value
    )  # ? random_state: for reproduction of code
    embeddings_2d = tsne_model.fit_transform(selected_embeddings.numpy())

    plt.figure(figsize=(12, 10))
    for i, label in enumerate(selected_indices):
        x, y = embeddings_2d[i, 0], embeddings_2d[i, 1]
        plt.scatter(x, y)
        if label in index_to_word:
            plt.annotate(
                index_to_word[label],
                (x, y),
                textcoords="offset points",
                xytext=(2, 2),
                ha="right",
                fontsize=8,
            )

    plt.title("t-SNE visualization of word embeddings")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Save the plot
    output_folder = f"outputs/{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    plot_filename = f"{output_folder}/tsne_plot_{timestamp}.png"

    plt.savefig(plot_filename)
    print(f"t-SNE plot saved to {plot_filename}\n")


#! Training
def train_model(model, dataloader, num_epochs, device):
    # Define loss and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()  # set model to training-mode

        for center_batch, context_batch in tqdm(dataloader):
            center_batch = center_batch.to(device)  #! changed to apple gpu!!!
            context_batch = context_batch.to(device)

            optimizer.zero_grad()
            probs = model(center_batch)
            loss = loss_function(probs, context_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)  # average loss per batch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    print()
    return avg_loss


def train_model_negative_sampling(
    model, dataloader, num_epochs, device, vocab_size, num_negatives=5
):
    loss_function = nn.BCEWithLogitsLoss()  #! changed to binary cross entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()

        for center_batch, pos_context_batch in tqdm(dataloader):
            center_batch = center_batch.to(device)
            pos_context_batch = pos_context_batch.to(device)

            batch_size = center_batch.size(0)

            neg_context_batch = torch.randint(
                0, vocab_size, (batch_size, num_negatives)
            ).to(device)
            optimizer.zero_grad()
            pos_score, neg_score = model(
                center_batch, pos_context_batch, neg_context_batch
            )

            pos_labels = torch.ones_like(pos_score)
            neg_labels = torch.zeros_like(neg_score)

            loss_pos = loss_function(pos_score, pos_labels)
            loss_neg = loss_function(neg_score, neg_labels)

            loss = loss_pos + loss_neg
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    print()
    return avg_loss


#! Export Outputs
def export_output(
    num_epochs,
    batch_size,
    my_dim,
    performance_cut,
    use_stopwords,
    scatter_words,
    perplexity,
    training_time,
    final_loss,
    dataset_name,
    model_type,
):

    filename = f"outputs/{timestamp}/output{timestamp}.txt"
    os.makedirs(f"outputs/{timestamp}", exist_ok=True)
    content = f"""Training Output - {timestamp}

Dataset: {dataset_name}
Model Type: {model_type}

Hyperparameters:
- Number of Epochs: {num_epochs}
- Batch Size: {batch_size}
- Embedding Dimension: {my_dim}
- Tokens used: {performance_cut}
- Stopwords Removed: {use_stopwords}
- Scatter Words (for t-SNE): {scatter_words}
- Perplexity (for t-SNE): {perplexity}

Training Results:
- Training Time: {training_time} seconds
- Final Loss per badge: {final_loss:.4f}
"""

    with open(filename, "w") as f:
        f.write(content)
    print(f"\nOutput saved to {filename}\n")


if __name__ == "__main__":
    main()
