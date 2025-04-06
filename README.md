Homework 2 - Word2Vec Implementation

Project Description

This project is part of the Natural Language Processing course.
The task was to implement a custom Word2Vec model with two training methods:
	•	Standard Skip-Gram model (with Softmax)
	•	Skip-Gram model with Negative Sampling

Additionally, the project includes functions for evaluating and visualizing the trained word embeddings.

⸻

Repository

Official repository:
https://gitlab.lrz.de/ki430/homework2

⸻

Requirements
	•	Python 3.10
	•	A virtual environment is recommended (e.g., using uv or venv)
	•	Required packages:
	•	torch
	•	datasets
	•	nltk
	•	scikit-learn
	•	gensim
	•	matplotlib
	•	tqdm

⸻

Installation and Execution
	1.	Clone the repository:

git clone https://gitlab.lrz.de/ki430/homework2.git
cd homework2

	2.	Set up a virtual environment (recommended using uv):

uv venv
source .venv/bin/activate
uv pip install torch datasets nltk scikit-learn gensim matplotlib tqdm

	3.	Run the project:

uv run word2vec.py

During execution, you will be asked to select the model type (Standard or Negative Sampling) and whether you want to use the Apple MPS GPU if available.

⸻

Features
	•	Training Word2Vec models
	•	Choice between Standard Skip-Gram and Negative Sampling.
	•	Use of text datasets (imdb, text8, wikitext103).
	•	Device selection (CPU or Apple MPS GPU).
	•	Configurable hyperparameters (embedding dimension, batch size, number of epochs).
	•	Evaluation
	•	Custom implementation of Cosine Similarity evaluation.
	•	Integration of trained embeddings into Gensim for using the .most_similar() API.
	•	Visualization
	•	Generation of 2D t-SNE plots of the learned word embeddings.
	•	Saving plots into an output directory.
	•	Automatic saving of major training results, hyperparameters, and visualizations with timestamps.

⸻

Project Files

File	Description
word2vec.py	Main file containing training, evaluation, and visualization logic.
outputs/	Directory where all results (plots, training logs) are automatically stored.
.venv/	Virtual environment directory (locally created, not part of the repository).

