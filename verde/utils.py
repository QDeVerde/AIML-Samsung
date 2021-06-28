import logging
import sys

from logdecorator import log_on_end, log_on_start
from tqdm import tqdm


def configure_logging():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def ensure_is_list(target):
    if type(target) != list:
        return [target]
    else:
        return target


def log(msg: str, log_level=logging.INFO):
    logging.log(log_level, msg)


def progress(iterable, desc='', **kwargs):
    return tqdm(
        iterable,
        desc=desc,
        file=sys.stderr,
        bar_format='[INFO] {desc} {elapsed}/{remaining} - {percentage:3.2f}%',
        dynamic_ncols=True,
        **kwargs
    )


def lmap(target, function):
    return list(map(function, target))
