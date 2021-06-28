import logging
import os

import torch
import torch.nn.functional as F
from logdecorator import log_on_start, log_on_end
from torch.utils.data import DataLoader
from tqdm import tqdm

from verde.nlp import BertTokenizerWrapper
from verde.data import FolderIterableDataset
from verde.utils import configure_logging, log

from model import Miss

import meta

from transformers import get_constant_schedule_with_warmup


@log_on_start(logging.INFO, 'Starting training')
@log_on_end(logging.INFO, 'Training finished')
def main():
    tokenizer = BertTokenizerWrapper(meta.max_seq_len)
    vocab_size = tokenizer.bert_tokenizer.vocab_size

    dataloaders = get_dataloaders(tokenizer.encode)

    model = Miss(vocab_size=vocab_size, bos_token_id=tokenizer.bert_tokenizer.cls_token_id).to(meta.device)
    try_load_pretrained_weights(model)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=meta.lr
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    scheduler = get_constant_schedule_with_warmup(optimizer, meta.warmup)

    try:

        for epoch in range(meta.epochs):
            running_loss = []
            epoch_last_mean_loss = None

            for it, batch in enumerate(dataloaders['train']):
                ###
                ### Fetching data ...
                ###

                # batch x story_len x 2048 x 8 x 8
                images_features = batch[0].to(meta.device)

                # batch x story_len x max_seq_len
                input_ids = batch[1]['input_ids'].to(meta.device)

                # batch x story_len x max_seq_len
                attention_mask = batch[1]['attention_mask'].to(meta.device)

                ###
                ### Training ...
                ###

                outputs = model(images_features, input_ids[:, :-1], attention_mask[:, :-1])
                targets = input_ids[:, 1:]

                loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
                running_loss.append(loss.item())

                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

                ###
                ### Logging ...
                ###

                if it % meta.loss_verbose_k == 0:
                    epoch_last_mean_loss = sum(running_loss) / len(running_loss)
                    log(f'Epoch={epoch}, it={it}/{len(dataloaders["train"])}, loss={round(epoch_last_mean_loss, 4)}')
                    running_loss.clear()

                ###
                ### Inference
                ###

                if it % meta.inference_log_k == 0:
                    with open(f'./workdir/logs/log-{epoch}-{it}-{loss.item()}.txt', mode='w', encoding='utf-8') as f:
                        for i in range(5):
                            out = model._raw_inference(images_features[i].unsqueeze(0))
                            f.write('ORIGINAL:\n')
                            for line in tokenizer.decode(input_ids[i]):
                                f.write(line + ' ')

                            f.write('\n\n\nGENERATED:\n')
                            for line in tokenizer.decode(out.tolist()):
                                f.write(line + '\n\n')

                            f.write('\n\n\n\n')

            ###
            ### Saving on epoch end
            ###

            torch.save(model.state_dict(), meta.epoch_saving_path(epoch, epoch_last_mean_loss))
            torch.save(model.state_dict(), meta.last_saving_path)
            log(f'Saved model on epoch={epoch} with loss={epoch_last_mean_loss}')

    except KeyboardInterrupt:
        log('Saving model ...')
        torch.save(model.state_dict(), meta.last_saving_path)
        log('Model saved')


def get_dataloaders(encode_method):
    selections = ['train', 'val', 'test']
    datasets = {
        selection: FolderIterableDataset(os.path.join('./workdir', 'data', f'{selection}'), encode_method)
        for selection in selections
    }
    dataloaders = {
        selection: DataLoader(datasets[selection], batch_size=meta.batch_size)
        for selection in selections
    }
    return dataloaders


def try_load_pretrained_weights(model):
    if os.path.exists(meta.last_saving_path):
        try:
            log('Pretrained model found')
            model.load_state_dict(torch.load(meta.last_saving_path))
            log('Pretrained model loaded')
        except Exception:
            log('Pretrained model failed to load, creating new ...')
    else:
        log('Pretrained model not found, creating new ...')


if __name__ == '__main__':
    configure_logging()
    main()
