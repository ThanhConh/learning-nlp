from collections import Counter
import math
import re


class NGramLanguageModel:

    def __init__(
        self,
        n=2,
        smoothing="mle"
    ):
        if n not in [1, 2, 3]:
            raise ValueError("n must be 1, 2, or 3")

        if smoothing not in ["mle", "laplace"]:
            raise ValueError(
                "smoothing must be 'mle' or 'laplace'"
            )

        self.n = n
        self.smoothing = smoothing

        self.vocabulary = set()

        self.ngram_counts = Counter()
        self.context_counts = Counter()

        self.total_tokens = 0
        self.total_ngrams = 0

    # --------------------------------------------------
    # Tokenization
    # --------------------------------------------------

    def tokenize(self, sentence):

        return re.findall(
            r"[a-zA-Z0-9]+",
            sentence.lower()
        )

    # --------------------------------------------------
    # Vocabulary
    # --------------------------------------------------

    def build_vocabulary(self, corpus):
        counter = Counter()
        for sentence in corpus:
            tokens = self.tokenize(sentence)
            counter.update(tokens)
        self.vocabulary = set(counter.keys())
        self.vocabulary.add("<END>")
        return self.vocabulary


    # --------------------------------------------------
    # Count n-grams
    # --------------------------------------------------

    def count_ngrams(self, corpus):

        self.ngram_counts.clear()
        self.context_counts.clear()

        for sentence in corpus:

            tokens = self.tokenize(sentence)

            if not tokens:
                continue

            tokens = (
                ["<START>"] * (self.n - 1)
                + tokens
                + ["<END>"]
            )

            for i in range(
                len(tokens) - self.n + 1
            ):

                ngram = tuple(
                    tokens[i:i + self.n]
                )

                self.ngram_counts[ngram] += 1

                if self.n > 1:

                    context = ngram[:-1]

                    self.context_counts[
                        context
                    ] += 1

        self.total_ngrams = sum(
            self.ngram_counts.values()
        )

    # --------------------------------------------------
    # Fit
    # --------------------------------------------------

    def fit(self, corpus):

        self.build_vocabulary(corpus)

        self.count_ngrams(corpus)

        self.total_tokens = sum(
            self.ngram_counts.values()
        )

        return self

    # --------------------------------------------------
    # Probability
    # --------------------------------------------------

    def probability(self, context, word):
        # Giữ nguyên chữ hoa cho token đặc biệt, chỉ lower các từ thường
        if word.upper() in ("<START>", "<END>"):
            word = word.upper()
        else:
            word = word.lower()

        context = tuple(
            w.upper() if w.upper() in ("<START>", "<END>") else w.lower()
            for w in context
        )

        # -----------------------------
        # Unigram
        # -----------------------------
        if self.n == 1:
            count = self.ngram_counts.get((word,), 0)

            if self.smoothing == "mle":
                if self.total_ngrams == 0:
                    return 0.0
                return count / self.total_ngrams
            else:
                vocab_size = len(self.vocabulary)
                denom = self.total_ngrams + vocab_size
                if denom == 0:
                    return 0.0
                return (count + 1) / denom

        # -----------------------------
        # Bigram / Trigram
        # -----------------------------
        ngram = context + (word,)

        count = self.ngram_counts.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)

        # MLE
        if self.smoothing == "mle":
            if context_count == 0:
                return 0.0
            return count / context_count

        # Laplace
        vocab_size = len(self.vocabulary)
        denom = context_count + vocab_size
        if denom == 0:
            return 0.0
        return (count + 1) / denom


    # --------------------------------------------------
    # Log probability
    # --------------------------------------------------

    def sentence_log_probability(
        self,
        sentence
    ):

        tokens = self.tokenize(sentence)

        if not tokens:
            return 0.0

        tokens = (
            ["<START>"] * (self.n - 1)
            + tokens
            + ["<END>"]
        )

        log_probability = 0.0

        for i in range(
            self.n - 1,
            len(tokens)
        ):

            word = tokens[i]

            if self.n == 1:

                context = ()

            else:

                context = tuple(
                    tokens[
                        i - self.n + 1:i
                    ]
                )

            probability = self.probability(
                context,
                word
            )

            if probability <= 0:

                return float("-inf")

            log_probability += math.log(
                probability
            )

        return log_probability

    # --------------------------------------------------
    # Sentence probability
    # --------------------------------------------------

    def sentence_probability(
        self,
        sentence
    ):

        log_prob = (
            self.sentence_log_probability(
                sentence
            )
        )

        if log_prob == float("-inf"):

            return 0.0

        return math.exp(log_prob)