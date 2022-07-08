'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''
import codecs
import os

_pad = '_'
_eos = '~'
_mask = '@[MASK]'


def load_symbols(dict_path, has_mask=True):
    _characters = ''
    _ch_symbols = []
    sy_dict_name = 'sy_dict.txt'
    sy_dict_path = os.path.join(dict_path, sy_dict_name)
    f = codecs.open(sy_dict_path, 'r')
    for line in f:
        line = line.strip('\r\n')
        _ch_symbols.append(line)

    _arpabet = ['@' + s for s in _ch_symbols]

    # Export all symbols:
    sy = list(_characters) + _arpabet + [_pad, _eos]
    if has_mask:
        sy.append(_mask)

    _characters = ''

    _ch_tones = []
    tone_dict_name = 'tone_dict.txt'
    tone_dict_path = os.path.join(dict_path, tone_dict_name)
    f = codecs.open(tone_dict_path, 'r')
    for line in f:
        line = line.strip('\r\n')
        _ch_tones.append(line)

    # Export all tones:
    tone = list(_characters) + _ch_tones + [_pad, _eos]
    if has_mask:
        tone.append(_mask)

    _characters = ''

    _ch_syllable_flags = []
    syllable_flag_name = 'syllable_flag_dict.txt'
    syllable_flag_path = os.path.join(dict_path, syllable_flag_name)
    f = codecs.open(syllable_flag_path, 'r')
    for line in f:
        line = line.strip('\r\n')
        _ch_syllable_flags.append(line)

    # Export all syllable_flags:
    syllable_flag = list(_characters) + _ch_syllable_flags + [_pad, _eos]
    if has_mask:
        syllable_flag.append(_mask)

    _characters = ''

    _ch_word_segments = []
    word_segment_name = 'word_segment_dict.txt'
    word_segment_path = os.path.join(dict_path, word_segment_name)
    f = codecs.open(word_segment_path, 'r')
    for line in f:
        line = line.strip('\r\n')
        _ch_word_segments.append(line)

    # Export all syllable_flags:
    word_segment = list(_characters) + _ch_word_segments + [_pad, _eos]
    if has_mask:
        word_segment.append(_mask)

    _characters = ''

    _ch_emo_types = []
    emo_category_name = 'emo_category_dict.txt'
    emo_category_path = os.path.join(dict_path, emo_category_name)
    f = codecs.open(emo_category_path, 'r')
    for line in f:
        line = line.strip('\r\n')
        _ch_emo_types.append(line)

    emo_category = list(_characters) + _ch_emo_types + [_pad, _eos]
    if has_mask:
        emo_category.append(_mask)

    _characters = ''

    _ch_speakers = []
    speaker_name = 'speaker_dict.txt'
    speaker_path = os.path.join(dict_path, speaker_name)
    f = codecs.open(speaker_path, 'r')
    for line in f:
        line = line.strip('\r\n')
        _ch_speakers.append(line)

    # Export all syllable_flags:
    speaker = list(_characters) + _ch_speakers + [_pad, _eos]
    if has_mask:
        speaker.append(_mask)
    return sy, tone, syllable_flag, word_segment, emo_category, speaker
