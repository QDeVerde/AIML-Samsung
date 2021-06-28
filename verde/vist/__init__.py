import json
import logging
import os
from itertools import chain

from logdecorator import log_on_start, log_on_end

from verde.nlp.vocab import MinHitVocab, get_basic_tokenizer


def fetch_data(path_to_json_header: str, path_to_images: str) -> list[dict[str, list[str]]]:
    """
    Function for pulling specific data for VIST dataset.

    :param path_to_json_header: Path for the json header file
    :param path_to_images: Path for image folder
    :return: Iterable list with stories (texts, paths to images) in format {'texts': list[str], 'paths': list[str]}
    """

    # Opening and fetching data from json file
    with open(file=path_to_json_header, mode='r', encoding='utf-8') as file:
        raw = file.read()
        header = json.loads(raw)

    # Fetching unique stories ids
    stories_ids = {annotation_meta[0]['story_id'] for annotation_meta in header['annotations']}

    # Preforming stories list
    stories = {
        int(idx): {
            'texts': [],
            'paths': []
        }
        for idx in stories_ids
    }

    # Populating stories list
    for annotation_meta in header['annotations']:
        story_id = int(annotation_meta[0]['story_id'])

        text = annotation_meta[0]['original_text']
        image_id = annotation_meta[0]['photo_flickr_id']

        image_path = f'{path_to_images}/{image_id}.jpg'

        stories[story_id]['texts'].append(text)
        stories[story_id]['paths'].append(image_path)

    return list(stories.values())


@log_on_start(logging.INFO, 'Creating vocab')
@log_on_end(logging.INFO, 'Vocab created')
def create_vocab(data, max_seq_len, min_hits) -> MinHitVocab:
    texts = map(lambda story: story['texts'], data)
    corpus = list(chain(*texts))

    basic_tokenizer = get_basic_tokenizer()

    vocab = MinHitVocab(
        corpus=corpus,
        max_seq_len=max_seq_len,
        min_hits=min_hits,
        tokenize_function=basic_tokenizer
    )

    return vocab
