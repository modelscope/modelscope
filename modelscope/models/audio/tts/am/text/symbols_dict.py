import re
import sys

from .cleaners import (basic_cleaners, english_cleaners,
                       transliteration_cleaners)


class SymbolsDict:

    def __init__(self, sy, tone, syllable_flag, word_segment, emo_category,
                 speaker, inputs_dim, lfeat_type_list):
        self._inputs_dim = inputs_dim
        self._lfeat_type_list = lfeat_type_list
        self._sy_to_id = {s: i for i, s in enumerate(sy)}
        self._id_to_sy = {i: s for i, s in enumerate(sy)}
        self._tone_to_id = {s: i for i, s in enumerate(tone)}
        self._id_to_tone = {i: s for i, s in enumerate(tone)}
        self._syllable_flag_to_id = {s: i for i, s in enumerate(syllable_flag)}
        self._id_to_syllable_flag = {i: s for i, s in enumerate(syllable_flag)}
        self._word_segment_to_id = {s: i for i, s in enumerate(word_segment)}
        self._id_to_word_segment = {i: s for i, s in enumerate(word_segment)}
        self._emo_category_to_id = {s: i for i, s in enumerate(emo_category)}
        self._id_to_emo_category = {i: s for i, s in enumerate(emo_category)}
        self._speaker_to_id = {s: i for i, s in enumerate(speaker)}
        self._id_to_speaker = {i: s for i, s in enumerate(speaker)}
        print('_sy_to_id: ')
        print(self._sy_to_id)
        print('_tone_to_id: ')
        print(self._tone_to_id)
        print('_syllable_flag_to_id: ')
        print(self._syllable_flag_to_id)
        print('_word_segment_to_id: ')
        print(self._word_segment_to_id)
        print('_emo_category_to_id: ')
        print(self._emo_category_to_id)
        print('_speaker_to_id: ')
        print(self._speaker_to_id)
        self._curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')
        self._cleaners = {
            basic_cleaners.__name__: basic_cleaners,
            transliteration_cleaners.__name__: transliteration_cleaners,
            english_cleaners.__name__: english_cleaners
        }

    def _clean_text(self, text, cleaner_names):
        for name in cleaner_names:
            cleaner = self._cleaners.get(name)
            if not cleaner:
                raise Exception('Unknown cleaner: %s' % name)
            text = cleaner(text)
        return text

    def _sy_to_sequence(self, sy):
        return [self._sy_to_id[s] for s in sy if self._should_keep_sy(s)]

    def _arpabet_to_sequence(self, text):
        return self._sy_to_sequence(['@' + s for s in text.split()])

    def _should_keep_sy(self, s):
        return s in self._sy_to_id and s != '_' and s != '~'

    def symbol_to_sequence(self, this_lfeat_symbol, lfeat_type, cleaner_names):
        sequence = []
        if lfeat_type == 'sy':
            this_lfeat_symbol = this_lfeat_symbol.strip().split(' ')
            this_lfeat_symbol_format = ''
            index = 0
            while index < len(this_lfeat_symbol):
                this_lfeat_symbol_format = this_lfeat_symbol_format + '{' + this_lfeat_symbol[
                    index] + '}' + ' '
                index = index + 1
            sequence = self.text_to_sequence(this_lfeat_symbol_format,
                                             cleaner_names)
        elif lfeat_type == 'tone':
            sequence = self.tone_to_sequence(this_lfeat_symbol)
        elif lfeat_type == 'syllable_flag':
            sequence = self.syllable_flag_to_sequence(this_lfeat_symbol)
        elif lfeat_type == 'word_segment':
            sequence = self.word_segment_to_sequence(this_lfeat_symbol)
        elif lfeat_type == 'emo_category':
            sequence = self.emo_category_to_sequence(this_lfeat_symbol)
        elif lfeat_type == 'speaker':
            sequence = self.speaker_to_sequence(this_lfeat_symbol)
        else:
            raise Exception('Unknown lfeat type: %s' % lfeat_type)

        return sequence

    def text_to_sequence(self, text, cleaner_names):
        '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

          The text can optionally have ARPAbet sequences enclosed in curly braces embedded
          in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

          Args:
            text: string to convert to a sequence
            cleaner_names: names of the cleaner functions to run the text through

          Returns:
            List of integers corresponding to the symbols in the text
        '''
        sequence = []

        # Check for curly braces and treat their contents as ARPAbet:
        while len(text):
            m = self._curly_re.match(text)
            if not m:
                sequence += self._sy_to_sequence(
                    self._clean_text(text, cleaner_names))
                break
            sequence += self._sy_to_sequence(
                self._clean_text(m.group(1), cleaner_names))
            sequence += self._arpabet_to_sequence(m.group(2))
            text = m.group(3)

        # Append EOS token
        sequence.append(self._sy_to_id['~'])
        return sequence

    def tone_to_sequence(self, tone):
        tones = tone.strip().split(' ')
        sequence = []
        for this_tone in tones:
            sequence.append(self._tone_to_id[this_tone])
        sequence.append(self._tone_to_id['~'])
        return sequence

    def syllable_flag_to_sequence(self, syllable_flag):
        syllable_flags = syllable_flag.strip().split(' ')
        sequence = []
        for this_syllable_flag in syllable_flags:
            sequence.append(self._syllable_flag_to_id[this_syllable_flag])
        sequence.append(self._syllable_flag_to_id['~'])
        return sequence

    def word_segment_to_sequence(self, word_segment):
        word_segments = word_segment.strip().split(' ')
        sequence = []
        for this_word_segment in word_segments:
            sequence.append(self._word_segment_to_id[this_word_segment])
        sequence.append(self._word_segment_to_id['~'])
        return sequence

    def emo_category_to_sequence(self, emo_type):
        emo_categories = emo_type.strip().split(' ')
        sequence = []
        for this_category in emo_categories:
            sequence.append(self._emo_category_to_id[this_category])
        sequence.append(self._emo_category_to_id['~'])
        return sequence

    def speaker_to_sequence(self, speaker):
        speakers = speaker.strip().split(' ')
        sequence = []
        for this_speaker in speakers:
            sequence.append(self._speaker_to_id[this_speaker])
        sequence.append(self._speaker_to_id['~'])
        return sequence

    def sequence_to_symbol(self, sequence):
        result = ''
        pre_lfeat_dim = 0
        for lfeat_type in self._lfeat_type_list:
            current_one_hot_sequence = sequence[:, pre_lfeat_dim:pre_lfeat_dim
                                                + self._inputs_dim[lfeat_type]]
            current_sequence = current_one_hot_sequence.argmax(1)
            length = current_sequence.shape[0]

            index = 0
            while index < length:
                this_sequence = current_sequence[index]
                s = ''
                if lfeat_type == 'sy':
                    s = self._id_to_sy[this_sequence]
                    if len(s) > 1 and s[0] == '@':
                        s = s[1:]
                elif lfeat_type == 'tone':
                    s = self._id_to_tone[this_sequence]
                elif lfeat_type == 'syllable_flag':
                    s = self._id_to_syllable_flag[this_sequence]
                elif lfeat_type == 'word_segment':
                    s = self._id_to_word_segment[this_sequence]
                elif lfeat_type == 'emo_category':
                    s = self._id_to_emo_category[this_sequence]
                elif lfeat_type == 'speaker':
                    s = self._id_to_speaker[this_sequence]
                else:
                    raise Exception('Unknown lfeat type: %s' % lfeat_type)

                if index == 0:
                    result = result + lfeat_type + ': '

                result = result + '{' + s + '}'

                if index == length - 1:
                    result = result + '; '

                index = index + 1
            pre_lfeat_dim = pre_lfeat_dim + self._inputs_dim[lfeat_type]
        return result
