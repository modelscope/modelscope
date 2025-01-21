#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
# Part of this code is based on the source code of Transformers
# (arXiv:1910.03771)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re

import json
import numpy as np
import six
from tqdm import tqdm

logger = logging.getLogger(__name__)
USER_NAME = 'User'
SYSTEM_NAME = 'System'
DIALOG_ACT = 'Dialog_Act'


class DSTProcessor(object):
    ACTS_DICT = {
        'taxi-depart': 'taxi-departure',
        'taxi-dest': 'taxi-destination',
        'taxi-leaveat': 'taxi-leaveAt',
        'taxi-arriveby': 'taxi-arriveBy',
        'train-depart': 'train-departure',
        'train-dest': 'train-destination',
        'train-leaveat': 'train-leaveAt',
        'train-arriveby': 'train-arriveBy',
        'train-bookpeople': 'train-book_people',
        'restaurant-price': 'restaurant-pricerange',
        'restaurant-bookpeople': 'restaurant-book_people',
        'restaurant-bookday': 'restaurant-book_day',
        'restaurant-booktime': 'restaurant-book_time',
        'hotel-price': 'hotel-pricerange',
        'hotel-bookpeople': 'hotel-book_people',
        'hotel-bookday': 'hotel-book_day',
        'hotel-bookstay': 'hotel-book_stay',
        'booking-bookpeople': 'booking-book_people',
        'booking-bookday': 'booking-book_day',
        'booking-bookstay': 'booking-book_stay',
        'booking-booktime': 'booking-book_time',
    }

    LABEL_MAPS = {}  # Loaded from file

    def __init__(self):
        # Required for mapping slot names in dialogue_acts.json file
        # to proper designations.
        pass

    def _convert_inputs_to_utterances(self, inputs: dict,
                                      history_states: list):
        """This method is to generate the utterances with user, sys, dialog_acts and metadata,
         while metadata is from the history_states or the output from the inference pipline"""

        utterances = []
        user_inputs = []
        sys_gen_inputs = []
        dialog_acts_inputs = []
        for i, item in enumerate(inputs):
            name, turn = item.split('-')
            if name == USER_NAME:
                user_inputs.insert(int(turn) - 1, inputs[item])
            elif name == SYSTEM_NAME:
                sys_gen_inputs.insert(int(turn) - 1, inputs[item])
            else:
                dialog_acts_inputs.insert(int(turn) - 1, inputs[item])

        # user is leading the topic should aways larger than sys and dialog acts
        assert len(user_inputs) - 1 == len(sys_gen_inputs)
        assert len(user_inputs) - 1 == len(dialog_acts_inputs)
        # the history states record both user and sys states
        assert len(history_states) == len(user_inputs) + len(sys_gen_inputs)

        # the dialog_act at user turn is useless
        for i, item in enumerate(history_states):
            utterance = {}
            # the dialog_act at user turn is useless
            utterance['dialog_act'] = dialog_acts_inputs[
                i // 2] if i % 2 == 1 else {}
            utterance['text'] = sys_gen_inputs[
                i // 2] if i % 2 == 1 else user_inputs[i // 2]
            utterance['metadata'] = item
            utterance['span_info'] = []
            utterances.append(utterance)

        return utterances

    def _load_acts(self, inputs: dict, dialog_id='example.json'):
        dialog_acts_inputs = []
        for i, item in enumerate(inputs):
            name, turn = item.split('-')
            if name == DIALOG_ACT:
                dialog_acts_inputs.insert(int(turn) - 1, inputs[item])
        s_dict = {}

        for j, item in enumerate(dialog_acts_inputs):
            if isinstance(item, dict):
                for a in item:
                    aa = a.lower().split('-')
                    if aa[1] == 'inform' or aa[1] == 'recommend' or \
                            aa[1] == 'select' or aa[1] == 'book':
                        for i in item[a]:
                            s = i[0].lower()
                            v = i[1].lower().strip()
                            if s == 'none' or v == '?' or v == 'none':
                                continue
                            slot = aa[0] + '-' + s
                            if slot in self.ACTS_DICT:
                                slot = self.ACTS_DICT[slot]
                            key = dialog_id, str(int(j) + 1), slot
                            # In case of multiple mentioned values...
                            # ... Option 1: Keep first informed value
                            if key not in s_dict:
                                s_dict[key] = list([v])
                            # ... Option 2: Keep last informed value
                            # s_dict[key] = list([v])

        return s_dict


class multiwoz22Processor(DSTProcessor):

    def __init__(self):
        super().__init__()

    def normalize_time(self, text):
        text = re.sub(r'(\d{1})(a\.?m\.?|p\.?m\.?)', r'\1 \2',
                      text)  # am/pm without space
        text = re.sub(r'(^| )(\d{1,2}) (a\.?m\.?|p\.?m\.?)', r'\1\2:00 \3',
                      text)  # am/pm short to long form
        text = re.sub(
            r'(^| )(at|from|by|until|after) ?(\d{1,2}) ?(\d{2})([^0-9]|$)',
            r'\1\2 \3:\4\5', text)  # Missing separator
        text = re.sub(r'(^| )(\d{2})[;.,](\d{2})', r'\1\2:\3',
                      text)  # Wrong separator
        text = re.sub(r'(^| )(at|from|by|until|after) ?(\d{1,2})([;., ]|$)',
                      r'\1\2 \3:00\4', text)  # normalize simple full hour time
        text = re.sub(r'(^| )(\d{1}:\d{2})', r'\g<1>0\2',
                      text)  # Add missing leading 0
        # Map 12 hour times to 24 hour times
        text = \
            re.sub(
                r'(\d{2})(:\d{2}) ?p\.?m\.?',
                lambda x: str(int(x.groups()[0]) + 12
                              if int(x.groups()[0]) < 12 else int(x.groups()[0])) + x.groups()[1], text)
        text = re.sub(r'(^| )24:(\d{2})', r'\g<1>00:\2',
                      text)  # Correct times that use 24 as hour
        return text

    def normalize_text(self, text):
        text = self.normalize_time(text)
        text = re.sub("n't", ' not', text)
        text = re.sub('(^| )zero(-| )star([s.,? ]|$)', r'\g<1>0 star\3', text)
        text = re.sub('(^| )one(-| )star([s.,? ]|$)', r'\g<1>1 star\3', text)
        text = re.sub('(^| )two(-| )star([s.,? ]|$)', r'\g<1>2 star\3', text)
        text = re.sub('(^| )three(-| )star([s.,? ]|$)', r'\g<1>3 star\3', text)
        text = re.sub('(^| )four(-| )star([s.,? ]|$)', r'\g<1>4 star\3', text)
        text = re.sub('(^| )five(-| )star([s.,? ]|$)', r'\g<1>5 star\3', text)
        text = re.sub('archaelogy', 'archaeology', text)  # Systematic typo
        text = re.sub('guesthouse', 'guest house', text)  # Normalization
        text = re.sub('(^| )b ?& ?b([.,? ]|$)', r'\1bed and breakfast\2',
                      text)  # Normalization
        text = re.sub('bed & breakfast', 'bed and breakfast',
                      text)  # Normalization
        return text

    # Loads the dialogue_acts.json and returns a list
    # of slot-value pairs.
    def load_acts(self, input_file):
        with open(input_file, encoding='utf-8') as f:
            acts = json.load(f)
        s_dict = {}
        for d in acts:
            for t in acts[d]:
                if int(t) % 2 == 0:
                    continue
                # Only process, if turn has annotation
                if isinstance(acts[d][t]['dialog_act'], dict):
                    for a in acts[d][t]['dialog_act']:
                        aa = a.lower().split('-')
                        if aa[1] == 'inform' or aa[1] == 'recommend' \
                                or aa[1] == 'select' or aa[1] == 'book':
                            for i in acts[d][t]['dialog_act'][a]:
                                s = i[0].lower()
                                v = i[1].lower().strip()
                                if s == 'none' or v == '?' or v == 'none':
                                    continue
                                slot = aa[0] + '-' + s
                                if slot in self.ACTS_DICT:
                                    slot = self.ACTS_DICT[slot]
                                key = d, str(int(t) // 2 + 1), slot
                                # In case of multiple mentioned values...
                                # ... Option 1: Keep first informed value
                                if key not in s_dict:
                                    s_dict[key] = list([v])
                                # ... Option 2: Keep last informed value
                                # s_dict[key] = list([v])
        return s_dict

    # This should only contain label normalizations. All other mappings should
    # be defined in LABEL_MAPS.
    def normalize_label(self, slot, value_label):
        # Normalization of empty slots
        if value_label == '' or value_label == 'not mentioned':
            return 'none'

        # Normalization of time slots
        if 'leaveAt' in slot or 'arriveBy' in slot or slot == 'restaurant-book_time':
            return self.normalize_time(value_label)

        # Normalization
        if 'type' in slot or 'name' in slot or 'destination' in slot or 'departure' in slot:
            value_label = re.sub('guesthouse', 'guest house', value_label)

        # Map to boolean slots
        if slot == 'hotel-parking' or slot == 'hotel-internet':
            if value_label == 'yes' or value_label == 'free':
                return 'true'
            if value_label == 'no':
                return 'false'
        if slot == 'hotel-type':
            if value_label == 'hotel':
                return 'true'
            if value_label == 'guest house':
                return 'false'

        return value_label

    def tokenize(self, utt):
        utt_lower = convert_to_unicode(utt).lower()
        utt_lower = self.normalize_text(utt_lower)
        utt_tok = [
            tok for tok in map(str.strip, re.split(r'(\W+)', utt_lower))
            if len(tok) > 0
        ]
        return utt_tok

    def delex_utt(self, utt, values, unk_token='[UNK]'):
        utt_norm = self.tokenize(utt)
        for s, vals in values.items():
            for v in vals:
                if v != 'none':
                    v_norm = self.tokenize(v)
                    v_len = len(v_norm)
                    for i in range(len(utt_norm) + 1 - v_len):
                        if utt_norm[i:i + v_len] == v_norm:
                            utt_norm[i:i + v_len] = [unk_token] * v_len
        return utt_norm

    def get_token_pos(self, tok_list, value_label):
        find_pos = []
        found = False
        label_list = [
            item for item in map(str.strip, re.split(r'(\W+)', value_label))
            if len(item) > 0
        ]
        len_label = len(label_list)
        for i in range(len(tok_list) + 1 - len_label):
            if tok_list[i:i + len_label] == label_list:
                find_pos.append((i, i + len_label))  # start, exclusive_end
                found = True
        return found, find_pos

    def check_label_existence(self, value_label, usr_utt_tok):
        in_usr, usr_pos = self.get_token_pos(usr_utt_tok, value_label)
        # If no hit even though there should be one, check for value label variants
        if not in_usr and value_label in self.LABEL_MAPS:
            for value_label_variant in self.LABEL_MAPS[value_label]:
                in_usr, usr_pos = self.get_token_pos(usr_utt_tok,
                                                     value_label_variant)
                if in_usr:
                    break
        return in_usr, usr_pos

    def check_slot_referral(self, value_label, slot, seen_slots):
        referred_slot = 'none'
        if slot == 'hotel-stars' or slot == 'hotel-internet' or slot == 'hotel-parking':
            return referred_slot
        for s in seen_slots:
            # Avoid matches for slots that share values with different meaning.
            # hotel-internet and -parking are handled separately as Boolean slots.
            if s == 'hotel-stars' or s == 'hotel-internet' or s == 'hotel-parking':
                continue
            if re.match('(hotel|restaurant)-book_people',
                        s) and slot == 'hotel-book_stay':
                continue
            if re.match('(hotel|restaurant)-book_people',
                        slot) and s == 'hotel-book_stay':
                continue
            if slot != s and (slot not in seen_slots
                              or seen_slots[slot] != value_label):
                if seen_slots[s] == value_label:
                    referred_slot = s
                    break
                elif value_label in self.LABEL_MAPS:
                    for value_label_variant in self.LABEL_MAPS[value_label]:
                        if seen_slots[s] == value_label_variant:
                            referred_slot = s
                            break
        return referred_slot

    def is_in_list(self, tok, value):
        found = False
        tok_list = [
            item for item in map(str.strip, re.split(r'(\W+)', tok))
            if len(item) > 0
        ]
        value_list = [
            item for item in map(str.strip, re.split(r'(\W+)', value))
            if len(item) > 0
        ]
        tok_len = len(tok_list)
        value_len = len(value_list)
        for i in range(tok_len + 1 - value_len):
            if tok_list[i:i + value_len] == value_list:
                found = True
                break
        return found

    # Fuzzy matching to label informed slot values
    def check_slot_inform(self, value_label, inform_label):
        result = False
        informed_value = 'none'
        vl = ' '.join(self.tokenize(value_label))
        for il in inform_label:
            if vl == il:
                result = True
            elif self.is_in_list(il, vl):
                result = True
            elif self.is_in_list(vl, il):
                result = True
            elif il in self.LABEL_MAPS:
                for il_variant in self.LABEL_MAPS[il]:
                    if vl == il_variant:
                        result = True
                        break
                    elif self.is_in_list(il_variant, vl):
                        result = True
                        break
                    elif self.is_in_list(vl, il_variant):
                        result = True
                        break
            elif vl in self.LABEL_MAPS:
                for value_label_variant in self.LABEL_MAPS[vl]:
                    if value_label_variant == il:
                        result = True
                        break
                    elif self.is_in_list(il, value_label_variant):
                        result = True
                        break
                    elif self.is_in_list(value_label_variant, il):
                        result = True
                        break
            if result:
                informed_value = il
                break
        return result, informed_value

    def get_turn_label(self, value_label, inform_label, sys_utt_tok,
                       usr_utt_tok, slot, seen_slots, slot_last_occurrence):
        usr_utt_tok_label = [0 for _ in usr_utt_tok]
        informed_value = 'none'
        referred_slot = 'none'
        if value_label == 'none' or value_label == 'dontcare' or value_label == 'true' or value_label == 'false':
            class_type = value_label
        else:
            in_usr, usr_pos = self.check_label_existence(
                value_label, usr_utt_tok)
            is_informed, informed_value = self.check_slot_inform(
                value_label, inform_label)
            if in_usr:
                class_type = 'copy_value'
                if slot_last_occurrence:
                    (s, e) = usr_pos[-1]
                    for i in range(s, e):
                        usr_utt_tok_label[i] = 1
                else:
                    for (s, e) in usr_pos:
                        for i in range(s, e):
                            usr_utt_tok_label[i] = 1
            elif is_informed:
                class_type = 'inform'
            else:
                referred_slot = self.check_slot_referral(
                    value_label, slot, seen_slots)
                if referred_slot != 'none':
                    class_type = 'refer'
                else:
                    class_type = 'unpointable'
        return informed_value, referred_slot, usr_utt_tok_label, class_type

    def _create_example(self,
                        utterances,
                        sys_inform_dict,
                        set_type,
                        slot_list,
                        label_maps={},
                        append_history=False,
                        use_history_labels=False,
                        swap_utterances=False,
                        label_value_repetitions=False,
                        delexicalize_sys_utts=False,
                        unk_token='[UNK]',
                        analyze=False,
                        dialog_id='example.json'):

        # Collects all slot changes throughout the dialog
        # cumulative_labels = {slot: 'none' for slot in slot_list}

        # First system utterance is empty, since multiwoz starts with user input
        utt_tok_list = [[]]
        mod_slots_list = []

        # Collect all utterances and their metadata
        usr_sys_switch = True
        turn_itr = 0

        inform_dict = {slot: 'none' for slot in slot_list}
        for utt in utterances:
            # Assert that system and user utterances alternate
            is_sys_utt = utt['metadata'] != {}
            if usr_sys_switch == is_sys_utt:
                print(
                    'WARN: Wrong order of system and user utterances. Skipping rest of the dialog %s'
                    % (dialog_id))
                break
            usr_sys_switch = is_sys_utt

            if is_sys_utt:
                turn_itr += 1

            # Delexicalize sys utterance
            if delexicalize_sys_utts and is_sys_utt:
                inform_dict = {slot: 'none' for slot in slot_list}
                for slot in slot_list:
                    if (str(dialog_id), str(turn_itr),
                            slot) in sys_inform_dict:
                        inform_dict[slot] = sys_inform_dict[(str(dialog_id),
                                                             str(turn_itr),
                                                             slot)]
                utt_tok_list.append(
                    self.delex_utt(utt['text'], inform_dict,
                                   unk_token))  # normalize utterances
            else:
                utt_tok_list.append(self.tokenize(
                    utt['text']))  # normalize utterances

        # Form proper (usr, sys) turns
        turn_itr = 0
        diag_seen_slots_dict = {}
        diag_seen_slots_value_dict = {slot: 'none' for slot in slot_list}
        diag_state = {slot: 'none' for slot in slot_list}
        sys_utt_tok = []
        usr_utt_tok = []
        hst_utt_tok = []
        hst_utt_tok_label_dict = {slot: [] for slot in slot_list}
        new_hst_utt_tok_label_dict = hst_utt_tok_label_dict.copy()
        new_diag_state = diag_state.copy()

        ######
        mod_slots_list = []
        #####

        for i in range(0, len(utt_tok_list) - 1, 2):
            sys_utt_tok_label_dict = {}
            usr_utt_tok_label_dict = {}
            value_dict = {}
            # inform_dict = {}
            inform_slot_dict = {}
            referral_dict = {}
            class_type_dict = {}

            # Collect turn data
            if append_history:
                if swap_utterances:
                    hst_utt_tok = usr_utt_tok + sys_utt_tok + hst_utt_tok
                else:
                    hst_utt_tok = sys_utt_tok + usr_utt_tok + hst_utt_tok
            sys_utt_tok = utt_tok_list[i]
            usr_utt_tok = utt_tok_list[i + 1]
            turn_slots = mod_slots_list[
                i + 1] if len(mod_slots_list) > 1 else {}

            guid = '%s-%s-%s' % (set_type, str(dialog_id), str(turn_itr))

            if analyze:
                print('%15s %2s %s ||| %s' %
                      (dialog_id, turn_itr, ' '.join(sys_utt_tok),
                       ' '.join(usr_utt_tok)))
                print('%15s %2s [' % (dialog_id, turn_itr), end='')

            new_hst_utt_tok_label_dict = hst_utt_tok_label_dict.copy()
            new_diag_state = diag_state.copy()
            for slot in slot_list:
                value_label = 'none'
                if slot in turn_slots:
                    value_label = turn_slots[slot]
                    # We keep the original labels so as to not
                    # overlook unpointable values, as well as to not
                    # modify any of the original labels for test sets,
                    # since this would make comparison difficult.
                    value_dict[slot] = value_label
                elif label_value_repetitions and slot in diag_seen_slots_dict:
                    value_label = diag_seen_slots_value_dict[slot]

                # Get dialog act annotations
                inform_label = list(['none'])
                inform_slot_dict[slot] = 0
                if (str(dialog_id), str(turn_itr), slot) in sys_inform_dict:
                    inform_label = list([
                        self.normalize_label(slot, i)
                        for i in sys_inform_dict[(str(dialog_id),
                                                  str(turn_itr), slot)]
                    ])
                    inform_slot_dict[slot] = 1
                elif (str(dialog_id), str(turn_itr),
                      'booking-' + slot.split('-')[1]) in sys_inform_dict:
                    inform_label = list([
                        self.normalize_label(slot, i)
                        for i in sys_inform_dict[(str(dialog_id),
                                                  str(turn_itr), 'booking-'
                                                  + slot.split('-')[1])]
                    ])
                    inform_slot_dict[slot] = 1

                (informed_value, referred_slot, usr_utt_tok_label,
                 class_type) = self.get_turn_label(
                     value_label,
                     inform_label,
                     sys_utt_tok,
                     usr_utt_tok,
                     slot,
                     diag_seen_slots_value_dict,
                     slot_last_occurrence=True)

                # inform_dict[slot] = informed_value

                # Generally don't use span prediction on sys utterance (but inform prediction instead).
                sys_utt_tok_label = [0 for _ in sys_utt_tok]

                # Determine what to do with value repetitions.
                # If value is unique in seen slots, then tag it, otherwise not,
                # since correct slot assignment can not be guaranteed anymore.
                if label_value_repetitions and slot in diag_seen_slots_dict:
                    if class_type == 'copy_value' and list(
                            diag_seen_slots_value_dict.values()).count(
                                value_label) > 1:
                        class_type = 'none'
                        usr_utt_tok_label = [0 for _ in usr_utt_tok_label]

                sys_utt_tok_label_dict[slot] = sys_utt_tok_label
                usr_utt_tok_label_dict[slot] = usr_utt_tok_label

                if append_history:
                    if use_history_labels:
                        if swap_utterances:
                            new_hst_utt_tok_label_dict[
                                slot] = usr_utt_tok_label + sys_utt_tok_label + new_hst_utt_tok_label_dict[
                                    slot]
                        else:
                            new_hst_utt_tok_label_dict[
                                slot] = sys_utt_tok_label + usr_utt_tok_label + new_hst_utt_tok_label_dict[
                                    slot]
                    else:
                        new_hst_utt_tok_label_dict[slot] = [
                            0 for _ in sys_utt_tok_label + usr_utt_tok_label
                            + new_hst_utt_tok_label_dict[slot]
                        ]

                # For now, we map all occurences of unpointable slot values
                # to none. However, since the labels will still suggest
                # a presence of unpointable slot values, the task of the
                # DST is still to find those values. It is just not
                # possible to do that via span prediction on the current input.
                if class_type == 'unpointable':
                    class_type_dict[slot] = 'none'
                    referral_dict[slot] = 'none'
                    if analyze:
                        if slot not in diag_seen_slots_dict or value_label != diag_seen_slots_value_dict[
                                slot]:
                            print('(%s): %s, ' % (slot, value_label), end='')
                elif slot in diag_seen_slots_dict and class_type == diag_seen_slots_dict[slot] \
                        and class_type != 'copy_value' and class_type != 'inform':
                    # If slot has seen before and its class type did not change, label this slot a not present,
                    # assuming that the slot has not actually been mentioned in this turn.
                    # Exceptions are copy_value and inform. If a seen slot has been tagged as copy_value or inform,
                    # this must mean there is evidence in the original labels, therefore consider
                    # them as mentioned again.
                    class_type_dict[slot] = 'none'
                    referral_dict[slot] = 'none'
                else:
                    class_type_dict[slot] = class_type
                    referral_dict[slot] = referred_slot
                # Remember that this slot was mentioned during this dialog already.
                if class_type != 'none':
                    diag_seen_slots_dict[slot] = class_type
                    diag_seen_slots_value_dict[slot] = value_label
                    new_diag_state[slot] = class_type
                    # Unpointable is not a valid class, therefore replace with
                    # some valid class for now...
                    if class_type == 'unpointable':
                        new_diag_state[slot] = 'copy_value'

            if analyze:
                print(']')

            if swap_utterances:
                txt_a = usr_utt_tok
                txt_b = sys_utt_tok
                txt_a_lbl = usr_utt_tok_label_dict
                txt_b_lbl = sys_utt_tok_label_dict
            else:
                txt_a = sys_utt_tok
                txt_b = usr_utt_tok
                txt_a_lbl = sys_utt_tok_label_dict
                txt_b_lbl = usr_utt_tok_label_dict
            """
            text_a: dialog text
            text_b: dialog text
            history: dialog text
            text_a_label: label，ignore during inference，turns to start/end pos
            text_b_label: label，ignore during inference，turns to start/end pos
            history_label: label，ignore during inference，turns to start/end pos
            values: ignore during inference
            inform_label: ignore during inference
            inform_slot_label: input, system dialog action
            refer_label: label，ignore during inference，turns to start/end pos refer_id
            diag_state: input, history dialog state
            class_label: label，ignore during inference，turns to start/end pos class_label_id
            """
            example = DSTExample(
                guid=guid,
                text_a=txt_a,
                text_b=txt_b,
                history=hst_utt_tok,
                text_a_label=txt_a_lbl,
                text_b_label=txt_b_lbl,
                history_label=hst_utt_tok_label_dict,
                values=diag_seen_slots_value_dict.copy(),
                inform_label=inform_dict,
                inform_slot_label=inform_slot_dict,
                refer_label=referral_dict,
                diag_state=diag_state,
                class_label=class_type_dict)
            # Update some variables.
            hst_utt_tok_label_dict = new_hst_utt_tok_label_dict.copy()
            diag_state = new_diag_state.copy()

            turn_itr += 1
        return example

    def create_example(self,
                       inputs,
                       history_states,
                       set_type,
                       slot_list,
                       label_maps={},
                       append_history=False,
                       use_history_labels=False,
                       swap_utterances=False,
                       label_value_repetitions=False,
                       delexicalize_sys_utts=False,
                       unk_token='[UNK]',
                       analyze=False,
                       dialog_id='0'):
        utterances = self._convert_inputs_to_utterances(inputs, history_states)
        sys_inform_dict = self._load_acts(inputs)
        self.LABEL_MAPS = label_maps
        example = self._create_example(utterances, sys_inform_dict, set_type,
                                       slot_list, label_maps, append_history,
                                       use_history_labels, swap_utterances,
                                       label_value_repetitions,
                                       delexicalize_sys_utts, unk_token,
                                       analyze)

        return example

    def create_examples(self,
                        input_file,
                        acts_file,
                        set_type,
                        slot_list,
                        label_maps={},
                        append_history=False,
                        use_history_labels=False,
                        swap_utterances=False,
                        label_value_repetitions=False,
                        delexicalize_sys_utts=False,
                        unk_token='[UNK]',
                        analyze=False):
        """Read a DST json file into a list of DSTExample."""

        sys_inform_dict = self.load_acts(acts_file)

        with open(input_file, 'r', encoding='utf-8') as reader:
            input_data = json.load(reader)

        self.LABEL_MAPS = label_maps

        examples = []
        for dialog_id in tqdm(input_data):
            entry = input_data[dialog_id]
            utterances = entry['log']

            example = self._create_example(
                utterances, sys_inform_dict, set_type, slot_list, label_maps,
                append_history, use_history_labels, swap_utterances,
                label_value_repetitions, delexicalize_sys_utts, unk_token,
                analyze)
            examples.append(example)

        return examples


class DSTExample(object):
    """
    A single training/test example for the DST dataset.
    """

    def __init__(self,
                 guid,
                 text_a,
                 text_b,
                 history,
                 text_a_label=None,
                 text_b_label=None,
                 history_label=None,
                 values=None,
                 inform_label=None,
                 inform_slot_label=None,
                 refer_label=None,
                 diag_state=None,
                 class_label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.history = history
        self.text_a_label = text_a_label
        self.text_b_label = text_b_label
        self.history_label = history_label
        self.values = values
        self.inform_label = inform_label
        self.inform_slot_label = inform_slot_label
        self.refer_label = refer_label
        self.diag_state = diag_state
        self.class_label = class_label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s_dict = dict()
        s_dict['guid'] = self.guid
        s_dict['text_a'] = self.text_a
        s_dict['text_b'] = self.text_b
        s_dict['history'] = self.history
        if self.text_a_label:
            s_dict['text_a_label'] = self.text_a_label
        if self.text_b_label:
            s_dict['text_b_label'] = self.text_b_label
        if self.history_label:
            s_dict['history_label'] = self.history_label
        if self.values:
            s_dict['values'] = self.values
        if self.inform_label:
            s_dict['inform_label'] = self.inform_label
        if self.inform_slot_label:
            s_dict['inform_slot_label'] = self.inform_slot_label
        if self.refer_label:
            s_dict['refer_label'] = self.refer_label
        if self.diag_state:
            s_dict['diag_state'] = self.diag_state
        if self.class_label:
            s_dict['class_label'] = self.class_label

        s = json.dumps(s_dict)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_ids_unmasked,
                 input_mask,
                 segment_ids,
                 start_pos=None,
                 end_pos=None,
                 values=None,
                 inform=None,
                 inform_slot=None,
                 refer_id=None,
                 diag_state=None,
                 class_label_id=None,
                 guid='NONE'):
        self.guid = guid
        self.input_ids = input_ids
        self.input_ids_unmasked = input_ids_unmasked
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.values = values
        self.inform = inform
        self.inform_slot = inform_slot
        self.refer_id = refer_id
        self.diag_state = diag_state
        self.class_label_id = class_label_id


def convert_examples_to_features(examples,
                                 slot_list,
                                 class_types,
                                 model_type,
                                 tokenizer,
                                 max_seq_length,
                                 slot_value_dropout=0.0):
    """Loads a data file into a list of `InputBatch`s."""

    if model_type == 'bert':
        model_specs = {
            'MODEL_TYPE': 'bert',
            'CLS_TOKEN': '[CLS]',
            'UNK_TOKEN': '[UNK]',
            'SEP_TOKEN': '[SEP]',
            'TOKEN_CORRECTION': 4
        }
    else:
        logger.error('Unknown model type (%s). Aborting.' % (model_type))
        exit(1)

    def _tokenize_text_and_label(text, text_label_dict, slot, tokenizer,
                                 model_specs, slot_value_dropout):
        joint_text_label = [0 for _ in text_label_dict[slot]
                            ]  # joint all slots' label
        for slot_text_label in text_label_dict.values():
            for idx, label in enumerate(slot_text_label):
                if label == 1:
                    joint_text_label[idx] = 1

        text_label = text_label_dict[slot]
        tokens = []
        tokens_unmasked = []
        token_labels = []
        for token, token_label, joint_label in zip(text, text_label,
                                                   joint_text_label):
            token = convert_to_unicode(token)
            sub_tokens = tokenizer.tokenize(token)  # Most time intensive step
            tokens_unmasked.extend(sub_tokens)
            if slot_value_dropout == 0.0 or joint_label == 0:
                tokens.extend(sub_tokens)
            else:
                rn_list = np.random.random_sample((len(sub_tokens), ))
                for rn, sub_token in zip(rn_list, sub_tokens):
                    if rn > slot_value_dropout:
                        tokens.append(sub_token)
                    else:
                        tokens.append(model_specs['UNK_TOKEN'])
            token_labels.extend([token_label for _ in sub_tokens])
        assert len(tokens) == len(token_labels)
        assert len(tokens_unmasked) == len(token_labels)
        return tokens, tokens_unmasked, token_labels

    def _truncate_seq_pair(tokens_a, tokens_b, history, max_length):
        """Truncates a sequence pair in place to the maximum length.
        Copied from bert/run_classifier.py
        """
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b) + len(history)
            if total_length <= max_length:
                break
            if len(history) > 0:
                history.pop()
            elif len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _truncate_length_and_warn(tokens_a, tokens_b, history, max_seq_length,
                                  model_specs, guid):
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP], [SEP] with "- 4" (BERT)
        if len(tokens_a) + len(tokens_b) + len(
                history) > max_seq_length - model_specs['TOKEN_CORRECTION']:
            # logger.info('Truncate Example %s. Total len=%d.' %
            #             (guid, len(tokens_a) + len(tokens_b) + len(history)))
            input_text_too_long = True
        else:
            input_text_too_long = False
        _truncate_seq_pair(tokens_a, tokens_b, history,
                           max_seq_length - model_specs['TOKEN_CORRECTION'])
        return input_text_too_long

    def _get_token_label_ids(token_labels_a, token_labels_b,
                             token_labels_history, max_seq_length,
                             model_specs):
        token_label_ids = []
        token_label_ids.append(0)  # [CLS]
        for token_label in token_labels_a:
            token_label_ids.append(token_label)
        token_label_ids.append(0)  # [SEP]
        for token_label in token_labels_b:
            token_label_ids.append(token_label)
        token_label_ids.append(0)  # [SEP]
        for token_label in token_labels_history:
            token_label_ids.append(token_label)
        token_label_ids.append(0)  # [SEP]
        while len(token_label_ids) < max_seq_length:
            token_label_ids.append(0)  # padding
        assert len(token_label_ids) == max_seq_length
        return token_label_ids

    def _get_start_end_pos(class_type, token_label_ids, max_seq_length):
        if class_type == 'copy_value' and 1 not in token_label_ids:
            class_type = 'none'
        start_pos = 0
        end_pos = 0
        if 1 in token_label_ids:
            start_pos = token_label_ids.index(1)
            # Parsing is supposed to find only first location of wanted value
            if 0 not in token_label_ids[start_pos:]:
                end_pos = len(token_label_ids[start_pos:]) + start_pos - 1
            else:
                end_pos = token_label_ids[start_pos:].index(0) + start_pos - 1
            for i in range(max_seq_length):
                if i >= start_pos and i <= end_pos:
                    assert token_label_ids[i] == 1
        return class_type, start_pos, end_pos

    def _get_transformer_input(tokens_a, tokens_b, history, max_seq_length,
                               tokenizer, model_specs):
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append(model_specs['CLS_TOKEN'])
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append(model_specs['SEP_TOKEN'])
        segment_ids.append(0)
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append(model_specs['SEP_TOKEN'])
        segment_ids.append(1)
        for token in history:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append(model_specs['SEP_TOKEN'])
        segment_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        return tokens, input_ids, input_mask, segment_ids

    total_cnt = 0
    too_long_cnt = 0

    refer_list = ['none'] + slot_list

    features = []
    # Convert single example
    for (example_index, example) in enumerate(examples):

        total_cnt += 1

        value_dict = {}
        inform_dict = {}
        inform_slot_dict = {}
        refer_id_dict = {}
        diag_state_dict = {}
        class_label_id_dict = {}
        start_pos_dict = {}
        end_pos_dict = {}
        for slot in slot_list:
            tokens_a, tokens_a_unmasked, token_labels_a = _tokenize_text_and_label(
                example.text_a, example.text_a_label, slot, tokenizer,
                model_specs, slot_value_dropout)
            tokens_b, tokens_b_unmasked, token_labels_b = _tokenize_text_and_label(
                example.text_b, example.text_b_label, slot, tokenizer,
                model_specs, slot_value_dropout)
            tokens_history, tokens_history_unmasked, token_labels_history = _tokenize_text_and_label(
                example.history, example.history_label, slot, tokenizer,
                model_specs, slot_value_dropout)

            input_text_too_long = _truncate_length_and_warn(
                tokens_a, tokens_b, tokens_history, max_seq_length,
                model_specs, example.guid)

            if input_text_too_long:

                token_labels_a = token_labels_a[:len(tokens_a)]
                token_labels_b = token_labels_b[:len(tokens_b)]
                token_labels_history = token_labels_history[:len(tokens_history
                                                                 )]
                tokens_a_unmasked = tokens_a_unmasked[:len(tokens_a)]
                tokens_b_unmasked = tokens_b_unmasked[:len(tokens_b)]
                tokens_history_unmasked = tokens_history_unmasked[:len(
                    tokens_history)]

            assert len(token_labels_a) == len(tokens_a)
            assert len(token_labels_b) == len(tokens_b)
            assert len(token_labels_history) == len(tokens_history)
            assert len(token_labels_a) == len(tokens_a_unmasked)
            assert len(token_labels_b) == len(tokens_b_unmasked)
            assert len(token_labels_history) == len(tokens_history_unmasked)
            token_label_ids = _get_token_label_ids(token_labels_a,
                                                   token_labels_b,
                                                   token_labels_history,
                                                   max_seq_length, model_specs)

            value_dict[slot] = example.values[slot]
            inform_dict[slot] = example.inform_label[slot]

            class_label_mod, start_pos_dict[slot], end_pos_dict[
                slot] = _get_start_end_pos(example.class_label[slot],
                                           token_label_ids, max_seq_length)
            if class_label_mod != example.class_label[slot]:
                example.class_label[slot] = class_label_mod
            inform_slot_dict[slot] = example.inform_slot_label[slot]
            refer_id_dict[slot] = refer_list.index(example.refer_label[slot])
            diag_state_dict[slot] = class_types.index(example.diag_state[slot])
            class_label_id_dict[slot] = class_types.index(
                example.class_label[slot])

        if input_text_too_long:
            too_long_cnt += 1

        tokens, input_ids, input_mask, segment_ids = _get_transformer_input(
            tokens_a, tokens_b, tokens_history, max_seq_length, tokenizer,
            model_specs)
        if slot_value_dropout > 0.0:
            _, input_ids_unmasked, _, _ = _get_transformer_input(
                tokens_a_unmasked, tokens_b_unmasked, tokens_history_unmasked,
                max_seq_length, tokenizer, model_specs)
        else:
            input_ids_unmasked = input_ids

        assert (len(input_ids) == len(input_ids_unmasked))

        features.append(
            InputFeatures(
                guid=example.guid,
                input_ids=input_ids,
                input_ids_unmasked=input_ids_unmasked,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_pos=start_pos_dict,
                end_pos=end_pos_dict,
                values=value_dict,
                inform=inform_dict,
                inform_slot=inform_slot_dict,
                refer_id=refer_id_dict,
                diag_state=diag_state_dict,
                class_label_id=class_label_id_dict))

    return features


# From bert.tokenization (TF code)
def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode('utf-8', 'ignore')
        else:
            raise ValueError('Unsupported string type: %s' % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode('utf-8', 'ignore')
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError('Unsupported string type: %s' % (type(text)))
    else:
        raise ValueError('Not running on Python2 or Python 3?')


if __name__ == '__main__':
    processor = multiwoz22Processor()
    set_type = 'test'
    slot_list = [
        'taxi-leaveAt', 'taxi-destination', 'taxi-departure', 'taxi-arriveBy',
        'restaurant-book_people', 'restaurant-book_day',
        'restaurant-book_time', 'restaurant-food', 'restaurant-pricerange',
        'restaurant-name', 'restaurant-area', 'hotel-book_people',
        'hotel-book_day', 'hotel-book_stay', 'hotel-name', 'hotel-area',
        'hotel-parking', 'hotel-pricerange', 'hotel-stars', 'hotel-internet',
        'hotel-type', 'attraction-type', 'attraction-name', 'attraction-area',
        'train-book_people', 'train-leaveAt', 'train-destination', 'train-day',
        'train-arriveBy', 'train-departure'
    ]
    append_history = True
    use_history_labels = True
    swap_utterances = True
    label_value_repetitions = True
    delexicalize_sys_utts = True,
    unk_token = '[UNK]'
    analyze = False

    utter1 = {
        'User-1':
        'am looking for a place to to stay that has cheap price range it should be in a type of hotel'
    }
    history_states1 = [
        {},
    ]
    utter2 = {
        'User-1':
        'am looking for a place to to stay that has cheap price range it should be in a type of hotel',
        'System-1':
        'Okay, do you have a specific area you want to stay in?',
        'Dialog_Act-1': {
            'Hotel-Request': [['Area', '?']]
        },
        'User-2':
        'no, i just need to make sure it\'s cheap. oh, and i need parking',
    }

    history_states2 = [{}, {
        'taxi': {
            'book': {
                'booked': []
            },
            'semi': {
                'leaveAt': '',
                'destination': '',
                'departure': '',
                'arriveBy': ''
            }
        },
        'police': {
            'book': {
                'booked': []
            },
            'semi': {}
        },
        'restaurant': {
            'book': {
                'booked': [],
                'people': '',
                'day': '',
                'time': ''
            },
            'semi': {
                'food': '',
                'pricerange': '',
                'name': '',
                'area': ''
            }
        },
        'hospital': {
            'book': {
                'booked': []
            },
            'semi': {
                'department': ''
            }
        },
        'hotel': {
            'book': {
                'booked': [],
                'people': '',
                'day': '',
                'stay': ''
            },
            'semi': {
                'name': 'not mentioned',
                'area': 'not mentioned',
                'parking': 'not mentioned',
                'pricerange': 'cheap',
                'stars': 'not mentioned',
                'internet': 'not mentioned',
                'type': 'hotel'
            }
        },
        'attraction': {
            'book': {
                'booked': []
            },
            'semi': {
                'type': '',
                'name': '',
                'area': ''
            }
        },
        'train': {
            'book': {
                'booked': [],
                'people': ''
            },
            'semi': {
                'leaveAt': '',
                'destination': '',
                'day': '',
                'arriveBy': '',
                'departure': ''
            }
        }
    }, {}]

    utter3 = {
        'User-1':
        'am looking for a place to to stay that has cheap price range it should be in a type of hotel',
        'System-1': 'Okay, do you have a specific area you want to stay in?',
        'Dialog_Act-1': {
            'Hotel-Request': [['Area', '?']]
        },
        'User-2':
        'no, i just need to make sure it\'s cheap. oh, and i need parking',
        'System-2':
        'I found 1 cheap hotel for you that includes parking. Do you like me to book it?',
        'Dialog_Act-2': {
            'Booking-Inform': [['none', 'none']],
            'Hotel-Inform': [['Price', 'cheap'], ['Choice', '1'],
                             ['Parking', 'none']]
        },
        'User-3': 'Yes, please. 6 people 3 nights starting on tuesday.'
    }

    history_states3 = [{}, {
        'taxi': {
            'book': {
                'booked': []
            },
            'semi': {
                'leaveAt': '',
                'destination': '',
                'departure': '',
                'arriveBy': ''
            }
        },
        'police': {
            'book': {
                'booked': []
            },
            'semi': {}
        },
        'restaurant': {
            'book': {
                'booked': [],
                'people': '',
                'day': '',
                'time': ''
            },
            'semi': {
                'food': '',
                'pricerange': '',
                'name': '',
                'area': ''
            }
        },
        'hospital': {
            'book': {
                'booked': []
            },
            'semi': {
                'department': ''
            }
        },
        'hotel': {
            'book': {
                'booked': [],
                'people': '',
                'day': '',
                'stay': ''
            },
            'semi': {
                'name': 'not mentioned',
                'area': 'not mentioned',
                'parking': 'not mentioned',
                'pricerange': 'cheap',
                'stars': 'not mentioned',
                'internet': 'not mentioned',
                'type': 'hotel'
            }
        },
        'attraction': {
            'book': {
                'booked': []
            },
            'semi': {
                'type': '',
                'name': '',
                'area': ''
            }
        },
        'train': {
            'book': {
                'booked': [],
                'people': ''
            },
            'semi': {
                'leaveAt': '',
                'destination': '',
                'day': '',
                'arriveBy': '',
                'departure': ''
            }
        }
    }, {}, {
        'taxi': {
            'book': {
                'booked': []
            },
            'semi': {
                'leaveAt': '',
                'destination': '',
                'departure': '',
                'arriveBy': ''
            }
        },
        'police': {
            'book': {
                'booked': []
            },
            'semi': {}
        },
        'restaurant': {
            'book': {
                'booked': [],
                'people': '',
                'day': '',
                'time': ''
            },
            'semi': {
                'food': '',
                'pricerange': '',
                'name': '',
                'area': ''
            }
        },
        'hospital': {
            'book': {
                'booked': []
            },
            'semi': {
                'department': ''
            }
        },
        'hotel': {
            'book': {
                'booked': [],
                'people': '',
                'day': '',
                'stay': ''
            },
            'semi': {
                'name': 'not mentioned',
                'area': 'not mentioned',
                'parking': 'yes',
                'pricerange': 'cheap',
                'stars': 'not mentioned',
                'internet': 'not mentioned',
                'type': 'hotel'
            }
        },
        'attraction': {
            'book': {
                'booked': []
            },
            'semi': {
                'type': '',
                'name': '',
                'area': ''
            }
        },
        'train': {
            'book': {
                'booked': [],
                'people': ''
            },
            'semi': {
                'leaveAt': '',
                'destination': '',
                'day': '',
                'arriveBy': '',
                'departure': ''
            }
        }
    }, {}]

    example = processor.create_example(utter2, history_states2, set_type,
                                       slot_list, {}, append_history,
                                       use_history_labels, swap_utterances,
                                       label_value_repetitions,
                                       delexicalize_sys_utts, unk_token,
                                       analyze)
    print(f'utterances is {example}')
