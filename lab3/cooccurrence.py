from collections import Counter

import numpy as np


def tokenize_corpus(corpus):
    """
    Convert raw sentences into tokenized sentences.

    Input:
        corpus: list[str]

    Output:
        sentences: list[list[str]]
    """
    sentences = []

    for sentence in corpus:
        tokens = sentence.lower().split()
        sentences.append(tokens)

    return sentences


def build_vocabulary(sentences):
    """
    Input:
        sentences: list[list[str]]

    Output:
        vocab: dict {word: index}
    """
    all_words = []

    for sentence in sentences:
        all_words.extend(sentence)

    word_counts = Counter(all_words)

    vocab = {}

    for idx, word in enumerate(word_counts.keys()):
        vocab[word] = idx

    return vocab


def cosine_similarity(vector1, vector2):
    """
    Calculate cosine similarity between two vectors.
    """
    numerator = vector1 @ vector2

    denominator = (
        np.linalg.norm(vector1)
        * np.linalg.norm(vector2)
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


def build_cooccurrence_matrix(
    sentences,
    vocab,
    window_size=2,
):
    """
    Input:
        sentences:
            list[list[str]]

        vocab:
            dict {word: index}

        window_size:
            size of context window

    Output:
        matrix:
            numpy.ndarray
            shape = (vocab_size, vocab_size)
    """

    vocab_size = len(vocab)

    matrix = np.zeros(
        (vocab_size, vocab_size),
        dtype=np.float32,
    )
    for sentence in sentences:

        for i, word in enumerate(sentence):

            if word not in vocab:
                continue

            word_idx = vocab[word]

            start = max(0, i - window_size)
            end = min(
                len(sentence),
                i + window_size + 1,
            )

            for j in range(start, end):

                if i == j:
                    continue

                context_word = sentence[j]

                if context_word not in vocab:
                    continue

                context_idx = vocab[context_word]

                matrix[word_idx, context_idx] += 1

    return matrix


def most_similar(
    corpus,
    sentences,
    vocab,
    co_matrix,
    query,
):
    """
    Find the document most similar to query.

    Input:
        corpus:
            list[str]

        sentences:
            list[list[str]]

        vocab:
            dict {word: index}

        co_matrix:
            co-occurrence matrix

        query:
            str

    Output:
        best_doc:
            most similar document

        best_sim:
            similarity score
    """

    def get_sentence_vector(tokens):

        vocab_size = co_matrix.shape[1]

        vector = np.zeros(
            vocab_size,
            dtype=np.float32,
        )

        count = 0

        for word in tokens:

            if word not in vocab:
                continue

            word_idx = vocab[word]

            vector += co_matrix[word_idx]

            count += 1

        if count > 0:
            vector /= count

        return vector

    query_tokens = query.lower().split()

    query_vec = get_sentence_vector(query_tokens)

    best_sim = -1.0
    best_doc = None

    for doc, doc_tokens in zip(corpus, sentences):

        doc_vec = get_sentence_vector(doc_tokens)

        sim = cosine_similarity(
            query_vec,
            doc_vec,
        )

        if sim > best_sim:
            best_sim = sim
            best_doc = doc

    return best_doc, float(best_sim)


if __name__ == "__main__":

    corpus = [
        "I love machine learning",
        "I love deep learning",
        "machine learning is interesting",
        "deep learning is powerful",
        "machine learning is fun",
        "deep learning is amazing",
    ]

    sentences = tokenize_corpus(corpus)

    print("Sentences:")
    print(sentences)

    print("\nVocabulary:")

    vocab = build_vocabulary(sentences)

    print(vocab)

    print("Co-occurrence matrix:")

    co_matrix = build_cooccurrence_matrix(
        sentences,
        vocab,
        window_size=2,
    )

    print(co_matrix)

    query = "machine learning"

    best_doc, best_sim = most_similar(
        corpus,
        sentences,
        vocab,
        co_matrix,
        query,
    )

    print("Query:")
    print(query)

    print("Most similar document:")
    print(best_doc)

    print("Similarity:")
    print(best_sim)