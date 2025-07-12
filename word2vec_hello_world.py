#!/usr/bin/env python3
"""
Word2Vec Hello World Example
A simple demonstration of word2vec using gensim
"""

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np

def create_sample_sentences():
    """Create sample sentences for training"""
    sentences = [
        ['hello', 'world', 'this', 'is', 'python'],
        ['python', 'is', 'a', 'programming', 'language'],
        ['machine', 'learning', 'is', 'fun'],
        ['word2vec', 'is', 'a', 'neural', 'network'],
        ['neural', 'networks', 'learn', 'word', 'embeddings'],
        ['embeddings', 'represent', 'words', 'as', 'vectors'],
        ['vectors', 'can', 'be', 'added', 'and', 'subtracted'],
        ['king', 'minus', 'man', 'plus', 'woman', 'equals', 'queen'],
        ['computer', 'science', 'is', 'interesting'],
        ['data', 'science', 'uses', 'python', 'and', 'machine', 'learning'],
        ['artificial', 'intelligence', 'is', 'the', 'future'],
        ['deep', 'learning', 'models', 'are', 'powerful'],
        ['natural', 'language', 'processing', 'is', 'amazing'],
        ['hello', 'there', 'how', 'are', 'you'],
        ['goodbye', 'world', 'see', 'you', 'later']
    ]
    return sentences

def train_word2vec_model(sentences):
    """Train a Word2Vec model on the given sentences"""
    print("Training Word2Vec model...")
    
    # Train the model
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,      # Size of word vectors
        window=5,             # Context window size
        min_count=1,          # Minimum word frequency
        workers=1,            # Number of CPU cores to use
        sg=1,                 # Skip-gram (1) or CBOW (0)
        epochs=100            # Number of training epochs
    )
    
    print(f"Model trained! Vocabulary size: {len(model.wv.key_to_index)}")
    print("=== Full Vocabulary with Vectors ===")
    for word in model.wv.key_to_index:
        vector = model.wv[word]
        print(f"{word}: {vector}")
    print(f"\nTotal vocabulary size: {len(model.wv.key_to_index)}")
    print(f"Vector dimensions: {model.wv.vector_size}")
    return model

def demonstrate_word_similarity(model):
    """Demonstrate finding similar words"""
    print("\n=== Word Similarity Examples ===")
    
    # Find words similar to 'python'
    print("Words similar to 'python':")
    try:
        similar_words = model.wv.most_similar('python', topn=5)
        for word, similarity in similar_words:
            print(f"  {word}: {similarity:.3f}")
    except KeyError:
        print("  'python' not found in vocabulary")
    
    # Find words similar to 'learning'
    print("\nWords similar to 'learning':")
    try:
        similar_words = model.wv.most_similar('learning', topn=5)
        for word, similarity in similar_words:
            print(f"  {word}: {similarity:.3f}")
    except KeyError:
        print("  'learning' not found in vocabulary")

def demonstrate_word_analogies(model):
    """Demonstrate word analogies (king - man + woman = queen)"""
    print("\n=== Word Analogy Examples ===")
    
    # Try the classic king - man + woman analogy
    try:
        result = model.wv.most_similar(
            positive=['king', 'woman'], 
            negative=['man'], 
            topn=3
        )
        print("king - man + woman ≈")
        for word, similarity in result:
            print(f"  {word}: {similarity:.3f}")
    except KeyError as e:
        print(f"Could not compute analogy: {e}")
    
    # Try another analogy: python - programming + science
    try:
        result = model.wv.most_similar(
            positive=['python', 'science'], 
            negative=['programming'], 
            topn=3
        )
        print("\npython - programming + science ≈")
        for word, similarity in result:
            print(f"  {word}: {similarity:.3f}")
    except KeyError as e:
        print(f"Could not compute analogy: {e}")

def show_word_vectors(model):
    """Show some word vectors"""
    print("\n=== Word Vectors ===")
    
    words_to_show = ['hello', 'world', 'python', 'learning']
    for word in words_to_show:
        try:
            vector = model.wv[word]
            print(f"{word}: vector shape {vector.shape}, first 5 values: {vector[:5]}")
        except KeyError:
            print(f"{word}: not found in vocabulary")

def main():
    """Main function to run the word2vec hello world example"""
    print("Word2Vec Hello World Example")
    print("=" * 40)
    
    # Create sample sentences
    sentences = create_sample_sentences()
    print(f"Created {len(sentences)} sample sentences")
    
    # Train the model
    model = train_word2vec_model(sentences)
    
    # Demonstrate various word2vec capabilities
    demonstrate_word_similarity(model)
    demonstrate_word_analogies(model)
    show_word_vectors(model)
    
    print("\n" + "=" * 40)
    print("Word2Vec Hello World Complete!")
    print("\nTry experimenting with:")
    print("- Different words in similarity searches")
    print("- Your own analogies")
    print("- Adding more training sentences")
    print("- Adjusting model parameters (vector_size, window, epochs)")

if __name__ == "__main__":
    main() 