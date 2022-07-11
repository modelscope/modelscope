import nltk

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download(
        'averaged_perceptron_tagger', halt_on_error=False, raise_on_error=True)

try:
    nltk.data.find('corpora/cmudict')
except LookupError:
    nltk.download('cmudict', halt_on_error=False, raise_on_error=True)
