"""Apply one caller-supplied date consistently to native chat templates."""
from datetime import datetime
from functools import wraps


def bind(tokenizer, date_string):
    original = tokenizer.apply_chat_template

    @wraps(original)
    def render(*args, **kwargs):
        if 'date_string' in kwargs and kwargs['date_string'] != date_string:
            raise ValueError('Conflicting template date')
        kwargs['date_string'] = date_string
        return original(*args, **kwargs)

    tokenizer.apply_chat_template = render
    return tokenizer


def install(date_string):
    datetime.strptime(date_string, '%d %b %Y')
    from transformers import AutoTokenizer
    original = AutoTokenizer.from_pretrained

    def load(*args, **kwargs):
        return bind(original(*args, **kwargs), date_string)

    AutoTokenizer.from_pretrained = staticmethod(load)
