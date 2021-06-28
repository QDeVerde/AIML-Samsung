from collections import Counter
from functools import reduce
from itertools import chain

import nltk


def get_basic_tokenizer(verbose=True):
    nltk.download('punkt', quiet=not verbose)

    return tokenize


def tokenize(text: str):
    return nltk.tokenize.word_tokenize(text.lower(), language='russian')


class SpecialToken:
    def __init__(self, name: str, value: int):
        """
        A special token used in vocabulary
        :param name: Token name
        :param value: Token id/value
        """
        self.name = name
        self.value = value

    def as_dict(self) -> dict:
        """
        :return: Token representation in dict format
        """
        return {self.name: self.value}


class MinHitVocab:
    def __init__(self, corpus: list[str], max_seq_len: int, min_hits: int, tokenize_function: callable):
        """
        Vocab with transforming and padding mechanics

        :param min_hits: Min hits of token in corpus to be added in vocabulary
        :param corpus:  Corpus to process
        :param tokenize_function: Tokenizing function (text aka str to list[str])
        :param max_seq_len: Amount of tokens that would be taking for alignment (without start|stop tokens)
        """

        self._min_hits = min_hits
        self.max_seq_len = max_seq_len
        self._tokenize = tokenize_function

        self.start_token = SpecialToken('<start>', 1)
        self.end_token = SpecialToken('<end>', 2)
        self.padding_token = SpecialToken('<empty>', 0)
        self.unk_token = SpecialToken('<unknown>', 3)
        self.sep_token = SpecialToken('<separator>', 4)

        # Adding special tokens to vocab
        self._token_to_value_dict = self.start_token.as_dict() | self.end_token.as_dict() | self.padding_token.as_dict() | self.unk_token.as_dict() | self.sep_token.as_dict()

        # Padding in vocab id list for first non-special token
        self._first_token_index = len(self._token_to_value_dict) + 1

        self._populate(corpus)

    def _populate(self, corpus: list[str]):
        # Tokenizing all texts in corpus
        tokenized_corpus = map(lambda text: self._tokenize(text), corpus)

        # Counting all tokens in corpus
        tokens_counter = Counter(chain(*tokenized_corpus))

        # Selecting only such tokens which occur more than min_hits times
        filtered_tokens = filter(lambda t: t[1] >= self._min_hits, tokens_counter.most_common())

        # Taking filtered tokens
        filtered_tokens = map(lambda t: t[0], filtered_tokens)

        # Expanding vocab with filtered tokens
        self._token_to_value_dict |= {filtered_token: i + self._first_token_index for i, filtered_token in
                                      enumerate(filtered_tokens)}

        self._value_to_token_dict = {v: k for k, v in self._token_to_value_dict.items()}

    def __len__(self):
        return self._value_to_token_dict.__len__() + 2  # Magic the gathering

    def _token_to_value(self, token: str) -> int:
        return self._token_to_value_dict.get(token, self.unk_token.value)

    def _value_to_token(self, value: int) -> str:
        return self._value_to_token_dict.get(value, self.unk_token.name)

    def extend_cut(self, data: list, max_seq_len) -> list:
        if len(data) <= max_seq_len:
            return data + [self.padding_token.value] * (max_seq_len - len(data))
        else:
            return data[:max_seq_len]

    def encode(self, texts: list[str]) -> list[int]:
        """
        Converting texts to tokens for nlp

        :param texts:
        :return: list of tokens
        """
        tokenized_text = list(map(lambda text: self._tokenize(text), texts))
        data = list(map(lambda it: list(map(lambda it: self._token_to_value_dict.get(it, self.unk_token.value), it)), tokenized_text))
        data = list(map(lambda it: self.extend_cut(it, self.max_seq_len) + [self.sep_token.value], data))
        data = [self.start_token.value] + list(chain(*data))[:-1] + [self.end_token.value]

        return list(data)

    def decode(self, text_values: list[list[int]]) -> list[str]:
        """
        Converting tokens to it's string representation

        :param text_values:
        :return: texts
        """
        text_tokens = map(
            lambda values: list(map(self._value_to_token, values)),
            text_values
        )

        texts = map(lambda it: reduce(lambda lhs, rhs: lhs + ' ' + rhs, it), text_tokens)

        return list(texts)
