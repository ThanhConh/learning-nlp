from collections import Counter
import math
def build_vocabulary(documents):
    vocab_set = set()
    for doc in documents:
        words = doc.split(" ")
        for word in words:
            vocab_set.add(word)
    return sorted(list(vocab_set))

def compute_counts(corpus, vocabulary):
    counts_list = []
    for doc in corpus:
        words = doc.split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
    
        doc_vector = []
        for word in vocabulary:
            doc_vector.append(word_counts.get(word, 0))
            
        counts_list.append(doc_vector)
    return counts_list

def compute_tf(counts_list, vocabulary):
    tf_list = []
    for doc_counts in counts_list:
        total_words = sum(doc_counts)
        doc_tf = []
        for count in doc_counts:
            tf_value = count / total_words if total_words > 0 else 0
            doc_tf.append(tf_value)

        tf_list.append(doc_tf)
    return tf_list

def compute_idf(counts_list, vocabulary):
    N = len(counts_list)
    idf_list = []

    num_words = len(vocabulary)
    for col_idx in range(num_words):
        docs_with_word = sum(1 for doc_counts in counts_list if doc_counts[col_idx] > 0)

        idf_value = math.log(N / docs_with_word) if docs_with_word > 0 else 0
        idf_list.append(idf_value)

    return idf_list

def compute_tfidf(tf_list, idf_list, vocabulary):
    tfidf_matrix = []
    for doc_tf in tf_list:
        doc_vector = []
        for col_idx in range(len(vocabulary)):
            doc_vector.append(doc_tf[col_idx] * idf_list[col_idx])
        tfidf_matrix.append(doc_vector)
    return tfidf_matrix

def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a ** 2 for a in vector_a))
    norm_b = math.sqrt(sum(b ** 2 for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)