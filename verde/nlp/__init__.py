from transformers import AutoTokenizer

import meta


class BertTokenizerWrapper:

    def __init__(self, max_seq_len=0, selected_model='bert-large-cased'):
        self.bert_tokenizer = AutoTokenizer.from_pretrained(selected_model)

        self.bert_tokenizer.add_special_tokens({
            'bos_token': '[SOS]',
            'eos_token': '[EOS]'
        })

        self.max_seq_len = max_seq_len

    def encode(self, texts: list[str]):
        data = self.bert_tokenizer.encode_plus(' '.join(texts), padding='max_length', truncation=True,
                                               max_length=meta.max_seq_len, return_tensors='pt')

        return {
            k: v[0] for k, v in data.items()
        }

    def decode(self, ids: list[list[int]]):
        return self.bert_tokenizer.batch_decode(ids)
