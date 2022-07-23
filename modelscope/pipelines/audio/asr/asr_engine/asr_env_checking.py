import ssl

import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download(
        'averaged_perceptron_tagger', halt_on_error=False, raise_on_error=True)

try:
    nltk.data.find('corpora/cmudict')
except LookupError:
    nltk.download('cmudict', halt_on_error=False, raise_on_error=True)
