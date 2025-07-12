# Word2Vec Hello World

A simple demonstration of Word2Vec using the Gensim library. This project shows how to train a Word2Vec model on sample sentences and perform word similarity and analogy tasks.

## Features

- Train a Word2Vec model on custom sentences
- Find similar words using cosine similarity
- Perform word analogies (e.g., king - man + woman = queen)
- Visualize word vectors
- Easy-to-understand code with detailed comments

## Requirements

- Python 3.7+
- Gensim
- NumPy

## Installation

1. Clone this repository:

```bash
git clone <your-repo-url>
cd kom_scratch
```

2. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the example:

```bash
python word2vec_hello_world.py
```

## What the Example Demonstrates

1. **Training**: Creates a Word2Vec model on 15 sample sentences
2. **Word Similarity**: Finds words similar to "python" and "learning"
3. **Word Analogies**: Performs the classic "king - man + woman = queen" analogy
4. **Vector Visualization**: Shows the actual word vectors

## Model Parameters

- **Vector Size**: 100 dimensions
- **Window Size**: 5 (context window)
- **Training Method**: Skip-gram
- **Epochs**: 100
- **Minimum Word Count**: 1

## Example Output

```
Word2Vec Hello World Example
========================================
Created 15 sample sentences
Training Word2Vec model...
Model trained! Vocabulary size: 56

=== Word Similarity Examples ===
Words similar to 'python':
  woman: 0.286
  words: 0.239
  embeddings: 0.217
  this: 0.214
  learn: 0.207

=== Word Analogy Examples ===
king - man + woman ≈
  a: 0.341
  as: 0.273
  queen: 0.248
```

## Customization

You can easily modify the example by:

1. **Adding more sentences** to the `create_sample_sentences()` function
2. **Changing model parameters** in `train_word2vec_model()`
3. **Trying different words** in similarity searches
4. **Creating your own analogies**

## Learning Resources

- [Gensim Word2Vec Documentation](https://radimrehurek.com/gensim/models/word2vec.html)
- [Word2Vec Paper](https://arxiv.org/abs/1301.3781)
- [Word Embeddings Tutorial](https://www.tensorflow.org/tutorials/text/word2vec)

## License

This project is open source and available under the [MIT License](LICENSE).
