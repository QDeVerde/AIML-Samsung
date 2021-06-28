from torch import device

# Working device
device = device('cuda')

# Vocab creation
max_seq_len = 27 * 4
story_len = 5
min_hits = 1
path_to_vocab = './workdir/vocab.pytorch'

# Training
batch_size = 16
epochs = 100
lr = 1e-4
loss_verbose_k = 10
inference_log_k = 250

warmup = 250

# Model
embedding_dim = 512
embedding_dropout = 0.1

# Saving
last_saving_path = './workdir/model-last.pytorch'
epoch_saving_path = lambda epoch, loss: f'./workdir/model-epoch-{epoch}-loss-{loss}.pytorch'
