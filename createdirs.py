from pathlib import Path
import logging

from logdecorator import log_on_start, log_on_end

from verde.utils import configure_logging


@log_on_start(logging.INFO, 'Creating working dirs')
@log_on_end(logging.INFO, 'Working dirs created')
def create_data_dirs():
    selections = ['train', 'test', 'val']

    # Creating dirs for all selections
    for selection in selections:
        Path(f'./workdir/data/{selection}/').mkdir(parents=True, exist_ok=True)

    Path(f'./workdir/logs/').mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    configure_logging()
    create_data_dirs()
