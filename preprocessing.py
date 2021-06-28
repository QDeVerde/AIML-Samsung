import logging
import os
import sys
import warnings

import torch
from PIL import Image

from verde.extractor import InceptionExtractor
from verde.utils import configure_logging, log_on_start, log_on_end, log, progress, lmap
from verde.vist import fetch_data

from meta import *


@log_on_start(logging.INFO, 'Starting preprocessing')
@log_on_end(logging.INFO, 'Preprocessing finished')
def main(argv):
    cutting = None
    if len(argv) == 2:
        cutting = int(argv[1])
        log(f'Set preprocessing dataset range to {cutting}')

    selections = ['train', 'val', 'test']
    data = {}

    for selection in progress(selections, 'Parsing raw dataset'):
        path_to_header = os.path.join('./dataset', 'sis', f'{selection}.story-in-sequence.json')
        path_to_images = os.path.join('./dataset', 'images', f'{selection}')

        raw_data = fetch_data(path_to_header, path_to_images)

        if cutting is not None:
            raw_data = raw_data[:cutting]

        data[selection] = raw_data

    extractor = InceptionExtractor(device=device)

    for selection in progress(selections, 'Preprocessing datasets'):
        errors = 0

        for index, story in enumerate(
                t := progress(data[selection],
                              desc=f'Parsing stories ({selection} selection), total {errors} corrupted stories')
        ):
            path_to_images = story['paths']
            texts = story['texts']

            try:
                images = lmap(path_to_images, lambda it: Image.open(it).convert('RGB'))
                lmap(images, lambda it: it.verify())

                features = extractor(images)

                sample = (features, texts)

                torch.save(sample, os.path.join('./workdir', 'data', f'{selection}', f'story-{index}.pytorch'))
            except:
                errors += 1
                t.desc = f'Parsing stories ({selection} selection), total {errors} corrupted stories'


if __name__ == '__main__':
    configure_logging()
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    main(sys.argv)
