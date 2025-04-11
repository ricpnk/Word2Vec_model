import torch
from gensim.models import KeyedVectors
from word2vec import SkipGramModel, SkipGramModelNegativeSampling
import pickle


def main():
    # parameters (have to be same as model)
    my_dim=300
    vocab_size=2**14
    device = torch.device("cpu")

    # open the saved functions and model
    with open("best_model/index_to_word.pkl", "rb") as f:
        index_to_word = pickle.load(f)

    if index_to_word != None:
        print("successfully loaded functions")
    else:
        print("Error: could not load functions")

    
    # load the model
    modelpath = "best_model/best_w2v_model.pt"
    model = SkipGramModelNegativeSampling(my_dim, vocab_size, device) # or negative sampling model
    model.load_state_dict(torch.load(modelpath, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    evaluate_gensim(model, index_to_word)
    print("\nGoodbye!")



def evaluate_gensim(model, index_to_word):
    print("Gensim evaluation:")
    embeddings = model.center_emb.weight.detach().cpu().numpy() # .weight: gets parameter | .detach(): cuts the tensor from comp | .cpu: copy's tensor to cpu from gpu | .numpy(): changes tensor to numpy array
    gensim_model = KeyedVectors(vector_size=embeddings.shape[1]) # create a empty KeyedVectors object for gensim
    vocab_list = [index_to_word[i] for i in range(len(index_to_word))] # build a vocab list
    gensim_model.add_vectors(vocab_list, embeddings) # add vectors to gensim model

    while True:
        test_word = input("Type in a center word (exit to quit): ")
        if test_word == "exit":
            break
        elif test_word in gensim_model:
            print(f"\nWord: {test_word}\nMost similar words:\n")
            similar_words = gensim_model.most_similar(test_word, topn=5)
            for context_word, score in similar_words:
                print(f"{context_word}: {score:.4f}")
        else:
            print(f"Error: {test_word} not in vocabulary!")



if __name__ == "__main__":
    main()