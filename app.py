import torch
import PIL.Image as Image
from urllib.parse import urlencode

from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename

from model import Miss, inference
from verde.extractor import InceptionExtractor
from verde.nlp import BertTokenizerWrapper

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = path_to_cache_folder = 'static/storage/'

device = torch.device('cpu')
extractor = InceptionExtractor(device)
tokenizer = BertTokenizerWrapper()

model = Miss(
    vocab_size=tokenizer.bert_tokenizer.vocab_size,
    bos_token_id=tokenizer.bert_tokenizer.bos_token_id
).to(device)

model.load_state_dict(torch.load('./workdir/model-last.pytorch'))


@app.route('/')
def index():
    images = request.args.getlist('img')
    desc = request.args.getlist('desc')

    return render_template('base.html', images=images, desc=desc)


@app.route('/api/v1/load_images', methods=['POST'])
def api_v1_load_images():
    images = request.files.getlist('files[]')

    selected = []

    for i, image in enumerate(images):
        filename = secure_filename(image.filename)
        image.save(path_to_cache_folder + filename)
        selected.append(filename)

    pimages = list(map(lambda it: Image.open(it.stream), images))
    features = extractor(pimages).unsqueeze(0)

    text = inference(
        miss_model=model,
        tokenizer=tokenizer,
        image_features=features,
    )

    selected = [('img', filename) for filename in selected]
    selected.extend([('desc', text)])

    return redirect('/?' + urlencode(selected))


if __name__ == '__main__':
    app.run(debug=True)
