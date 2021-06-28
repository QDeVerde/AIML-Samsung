import math

import torch
import torch.nn as nn

import meta


def generate_square_subsequent_mask(sz: int):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, batch_first=True):
        if batch_first:
            x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0), :]

        if batch_first:
            x = x.permute(1, 0, 2)
        return self.dropout(x)


class Miss(nn.Module):
    def __init__(self, vocab_size, bos_token_id):
        super(Miss, self).__init__()

        self.bos_token_id = bos_token_id

        self.vocab_size = vocab_size

        self.embedding = nn.Sequential(
            nn.Embedding(num_embeddings=vocab_size, embedding_dim=meta.embedding_dim),
            nn.Dropout(meta.embedding_dropout),
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=2048, nhead=16),
            num_layers=1
        )

        self.image_positional_embedding = PositionalEncoding(d_model=2048)
        self.tgt_pos_embedding = PositionalEncoding(d_model=meta.embedding_dim)

        self.projection = nn.Sequential(
            nn.Linear(2048, meta.embedding_dim),
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=meta.embedding_dim, nhead=4),
            num_layers=4
        )

        self.fc = nn.Linear(meta.embedding_dim, vocab_size)

    def forward(self, image_features, input_ids, attention_mask):
        """

        :param image_features: batch x story_len x 2048 x 8 x 8
        :param input_ids: batch x max_seq_len
        :param attention_mask: batch x story_len x max_seq_len
        :return:
        """

        batch_size = image_features.shape[0]
        story_len = image_features.shape[1]

        # batch_size x story_len x seq_len=64 x features=2048
        image_features = image_features.view([batch_size, story_len, 2048, 64]).permute(0, 1, 3, 2)

        image_features = image_features.reshape([batch_size, story_len * 64, 2048])

        image_features = self.image_positional_embedding(image_features, batch_first=True).permute(1, 0, 2)
        image_features = self.encoder(image_features)
        image_features = self.projection(image_features)

        # batch_size x max_seq_len x embedding_dim
        embedded_inputs = self.embedding(input_ids)
        embedded_inputs = self.tgt_pos_embedding(embedded_inputs, batch_first=True)
        embedded_inputs = embedded_inputs.permute(1, 0, 2)

        attention_mask = generate_square_subsequent_mask(embedded_inputs.shape[0]).to(meta.device)

        outputs = self.decoder(
            embedded_inputs,
            image_features,
            tgt_mask=attention_mask
        )

        logits = self.fc(outputs.permute(1, 0, 2))

        return logits

    def _raw_inference(self, image_features, length=27 * 5):
        with torch.no_grad():
            batch_size = image_features.shape[0]
            story_len = image_features.shape[1]

            image_features = image_features.view([batch_size, story_len, 2048, 64]).permute(0, 1, 3, 2)
            image_features = image_features.reshape([batch_size, story_len * 64, 2048])
            image_features = self.image_positional_embedding(image_features, batch_first=True).permute(1, 0, 2)
            image_features = self.encoder(image_features)
            image_features = self.projection(image_features)

            output = torch.ones(batch_size, length).long().to(meta.device) * self.bos_token_id

            for t in range(1, length):
                tgt_emb = self.embedding(output[:, :t])
                tgt_emb = self.tgt_pos_embedding(tgt_emb)
                tgt_emb = tgt_emb.permute(1, 0, 2)

                tgt_mask = generate_square_subsequent_mask(tgt_emb.shape[0]).to(meta.device)

                decoder_out = self.decoder(tgt=tgt_emb, memory=image_features, tgt_mask=tgt_mask)

                logits = self.fc(decoder_out)[-1, :, :]
                output_t = logits.data.topk(1)[1].squeeze()
                output[:, t] = output_t

            return output


def inference(miss_model, image_features, tokenizer, length=27 * 5):
    with torch.no_grad():
        batch_size = image_features.shape[0]
        story_len = image_features.shape[1]

        image_features = image_features.view([batch_size, story_len, 2048, 64]).permute(0, 1, 3, 2)
        image_features = image_features.reshape([batch_size, story_len * 64, 2048])
        image_features = miss_model.image_positional_embedding(image_features, batch_first=True).permute(1, 0, 2)
        image_features = miss_model.encoder(image_features)
        image_features = miss_model.projection(image_features)

        output = torch.ones(batch_size, length).long().to(torch.device('cpu')) * tokenizer.bert_tokenizer.cls_token_id

        stop_index = -1
        for t in range(1, length):
            tgt_emb = miss_model.embedding(output[:, :t])
            tgt_emb = miss_model.tgt_pos_embedding(tgt_emb)
            tgt_emb = tgt_emb.permute(1, 0, 2)

            tgt_mask = generate_square_subsequent_mask(tgt_emb.shape[0]).to(torch.device('cpu'))

            decoder_out = miss_model.decoder(tgt=tgt_emb, memory=image_features, tgt_mask=tgt_mask)

            logits = miss_model.fc(decoder_out)[-1, :, :]
            output_t = logits.data.topk(1)[1].squeeze()
            output[:, t] = output_t

            if output_t == tokenizer.bert_tokenizer.sep_token_id:
                stop_index = t
                break

        words = output[:, 1:stop_index][0].tolist()

        return tokenizer.decode([words])[0]
