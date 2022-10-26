# Copyright (c) Alibaba, Inc. and its affiliates.
import re

from .struct import TypeInfo


class SchemaLinker:

    def __init__(self):
        pass

    def find_in_list(self, comlist, words):
        result = False
        for com in comlist:
            if words in com:
                result = True
                break
        return result

    def get_continue_score(self, pstr, tstr):
        comlist = []
        minlen = min(len(pstr), len(tstr))
        for slen in range(minlen, 1, -1):
            for ts in range(0, len(tstr), 1):
                if ts + slen > len(tstr):
                    continue
                words = tstr[ts:ts + slen]
                if words in pstr and not self.find_in_list(comlist, words):
                    comlist.append(words)

        comlen = 0
        for com in comlist:
            comlen += len(com) * len(com)
        weight = comlen / (len(tstr) * len(tstr) + 0.001)
        if weight > 1.0:
            weight = 1.0

        return weight

    def get_match_score(self, ptokens, ttokens):
        pset = set(ptokens)
        tset = set(ttokens)
        comset = pset & tset
        allset = pset | tset
        weight2 = len(comset) / (len(allset) + 0.001)
        weight3 = self.get_continue_score(''.join(ptokens), ''.join(ttokens))
        return 0.4 * weight2 + 0.6 * weight3

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    def get_match_phrase(self, query, target):
        if target in query:
            return target, 1.0

        qtokens = []
        for i in range(0, len(query), 1):
            qtokens.append(query[i:i + 1])
        ttokens = []
        for i in range(0, len(target), 1):
            ttokens.append(target[i:i + 1])
        ttok_set = set(ttokens)

        phrase = ''
        score = 0.0
        for qidx, qword in enumerate(qtokens):
            if qword not in ttok_set:
                continue

            eidx = (qidx + 2 * len(ttokens)) if (
                len(qtokens) > qidx + 2 * len(ttokens)) else len(qtokens)
            while eidx > qidx:
                ptokens = qtokens[qidx:eidx]
                weight = self.get_match_score(ptokens, ttokens)
                if weight + 0.001 > score:
                    score = weight
                    phrase = ''.join(ptokens)
                eidx -= 1

        if self.is_number(target) and phrase != target:
            score = 0.0
        if len(phrase) > 1 and phrase in target:
            score *= (1.0 + 0.05 * len(phrase))

        return phrase, score

    def allfindpairidx(self, que_tok, value_tok, weight):
        idxs = []
        for i in range(0, len(que_tok) - len(value_tok) + 1, 1):
            s = i
            e = i
            matched = True
            for j in range(0, len(value_tok), 1):
                if value_tok[j].lower() == que_tok[i + j].lower():
                    e = i + j
                else:
                    matched = False
                    break
            if matched:
                idxs.append([s, e, weight])

        return idxs

    def findnear(self, ps1, pe1, ps2, pe2):
        if abs(ps1 - pe2) <= 2 or abs(pe1 - ps2) <= 2:
            return True
        return False

    def get_column_type(self, col_idx, table):
        colType = table['header_types'][col_idx]
        if 'number' in colType or 'duration' in colType or 'real' in colType:
            colType = 'real'
        elif 'date' in colType:
            colType = 'date'
        elif 'bool' in colType:
            colType = 'bool'
        else:
            colType = 'text'

        return colType

    def add_type_all(self, typeinfos, index, idxs, label, linktype, value,
                     orgvalue):
        for idx in idxs:
            info = TypeInfo(label, index, linktype, value, orgvalue, idx[0],
                            idx[1], idx[2])
            flag = True
            for i, typeinfo in enumerate(typeinfos):
                if info.pstart < typeinfo.pstart:
                    typeinfos.insert(i, info)
                    flag = False
                    break

            if flag:
                typeinfos.append(info)

        return typeinfos

    def save_info(self, tinfo, sinfo):
        flag = True
        if tinfo.pstart > sinfo.pend or tinfo.pend < sinfo.pstart:
            pass
        elif tinfo.pstart >= sinfo.pstart and \
                tinfo.pend <= sinfo.pend and tinfo.index == -1:
            flag = False
        elif tinfo.pstart == sinfo.pstart and sinfo.pend == tinfo.pend and \
                abs(tinfo.weight - sinfo.weight) < 0.01:
            pass
        else:
            if sinfo.label == 'col' or sinfo.label == 'val':
                if tinfo.label == 'col' or tinfo.label == 'val':
                    if (sinfo.pend
                            - sinfo.pstart) > (tinfo.pend - tinfo.pstart) or (
                                sinfo.weight > tinfo.weight
                                and sinfo.index != -1):
                        flag = False
                else:
                    flag = False
            else:
                if (tinfo.label == 'op' or tinfo.label == 'agg'):
                    if (sinfo.pend - sinfo.pstart) > (
                            tinfo.pend
                            - tinfo.pstart) or sinfo.weight > tinfo.weight:
                        flag = False

        return flag

    def normal_type_infos(self, infos):
        typeinfos = []
        for info in infos:
            typeinfos = [x for x in typeinfos if self.save_info(x, info)]
            flag = True
            for i, typeinfo in enumerate(typeinfos):
                if not self.save_info(info, typeinfo):
                    flag = False
                    break
                if info.pstart < typeinfo.pstart:
                    typeinfos.insert(i, info)
                    flag = False
                    break
            if flag:
                typeinfos.append(info)
        return typeinfos

    def findnear_typeinfo(self, info1, info2):
        return self.findnear(info1.pstart, info1.pend, info2.pstart,
                             info2.pend)

    def find_real_column(self, infos, table):
        for i, vinfo in enumerate(infos):
            if vinfo.index != -1 or vinfo.label != 'val':
                continue
            eoidx = -1
            for j, oinfo in enumerate(infos):
                if oinfo.label != 'op':
                    continue
                if self.findnear_typeinfo(vinfo, oinfo):
                    eoidx = j
                    break
            for j, cinfo in enumerate(infos):
                if cinfo.label != 'col' or table['header_types'][
                        cinfo.index] != 'real':
                    continue
                if self.findnear_typeinfo(cinfo, vinfo) or (
                        eoidx != -1
                        and self.findnear_typeinfo(cinfo, infos[eoidx])):
                    infos[i].index = cinfo.index
                    break

        return infos

    def filter_column_infos(self, infos):
        delid = []
        for i, info in enumerate(infos):
            if info.label != 'col':
                continue
            for j in range(i + 1, len(infos), 1):
                if infos[j].label == 'col' and \
                        info.pstart == infos[j].pstart and \
                        info.pend == infos[j].pend:
                    delid.append(i)
                    delid.append(j)
                    break

        typeinfos = []
        for idx, info in enumerate(infos):
            if idx in set(delid):
                continue
            typeinfos.append(info)

        return typeinfos

    def filter_type_infos(self, infos, table):
        infos = self.filter_column_infos(infos)
        infos = self.find_real_column(infos, table)

        colvalMp = {}
        for info in infos:
            if info.label == 'col':
                colvalMp[info.index] = []
        for info in infos:
            if info.label == 'val' and info.index in colvalMp:
                colvalMp[info.index].append(info)

        delid = []
        for idx, info in enumerate(infos):
            if info.label != 'val' or info.index in colvalMp:
                continue
            for index in colvalMp.keys():
                valinfos = colvalMp[index]
                for valinfo in valinfos:
                    if valinfo.pstart <= info.pstart and \
                            valinfo.pend >= info.pend:
                        delid.append(idx)
                        break

        typeinfos = []
        for idx, info in enumerate(infos):
            if idx in set(delid):
                continue
            typeinfos.append(info)

        return typeinfos

    def get_table_match_score(self, nlu_t, schema_link):
        match_len = 0
        for info in schema_link:
            scale = 0.6
            if info['question_len'] > 0 and info['column_index'] != -1:
                scale = 1.0
            else:
                scale = 0.5
            match_len += scale * info['question_len'] * info['weight']

        return match_len / (len(nlu_t) + 0.1)

    def get_entity_linking(self,
                           tokenizer,
                           nlu,
                           nlu_t,
                           tables,
                           col_syn_dict,
                           table_id=None,
                           history_sql=None):
        """
        get linking between question and schema column
        """
        typeinfos = []
        numbers = re.findall(r'[-]?\d*\.\d+|[-]?\d+|\d+', nlu)

        if table_id is not None and table_id in tables:
            tables = {table_id: tables[table_id]}

        # search schema link in every table
        search_result_list = []
        for tablename in tables:
            table = tables[tablename]
            trie_set = None
            if 'value_trie' in table:
                trie_set = table['value_trie']

            typeinfos = []
            for ii, column in enumerate(table['header_name']):
                column = column.lower()
                column_new = column
                cphrase, cscore = self.get_match_phrase(
                    nlu.lower(), column_new)
                if cscore > 0.3 and cphrase.strip() != '':
                    phrase_tok = tokenizer.tokenize(cphrase)
                    cidxs = self.allfindpairidx(nlu_t, phrase_tok, cscore)
                    typeinfos = self.add_type_all(typeinfos, ii, cidxs, 'col',
                                                  'column', cphrase, column)
                if cscore < 0.8 and column_new in col_syn_dict:
                    columns = list(set(col_syn_dict[column_new]))
                    for syn_col in columns:
                        if syn_col not in nlu.lower() or syn_col == '':
                            continue
                        phrase_tok = tokenizer.tokenize(syn_col)
                        cidxs = self.allfindpairidx(nlu_t, phrase_tok, 1.0)
                        typeinfos = self.add_type_all(typeinfos, ii, cidxs,
                                                      'col', 'column', syn_col,
                                                      column)

            for ii, trie in enumerate(trie_set):
                ans = trie.match(nlu.lower())
                for cell in ans.keys():
                    vphrase = cell
                    vscore = 1.0
                    phrase_tok = tokenizer.tokenize(vphrase)
                    if len(phrase_tok) == 0 or len(vphrase) < 2:
                        continue
                    vidxs = self.allfindpairidx(nlu_t, phrase_tok, vscore)
                    linktype = self.get_column_type(ii, table)
                    typeinfos = self.add_type_all(typeinfos, ii, vidxs, 'val',
                                                  linktype, vphrase, ans[cell])

            for number in set(numbers):
                number_tok = tokenizer.tokenize(number.lower())
                if len(number_tok) == 0:
                    continue
                nidxs = self.allfindpairidx(nlu_t, number_tok, 1.0)
                typeinfos = self.add_type_all(typeinfos, -1, nidxs, 'val',
                                              'real', number, number)

            newtypeinfos = self.normal_type_infos(typeinfos)

            newtypeinfos = self.filter_type_infos(newtypeinfos, table)

            final_question = [0] * len(nlu_t)
            final_header = [0] * len(table['header_name'])
            for typeinfo in newtypeinfos:
                pstart = typeinfo.pstart
                pend = typeinfo.pend + 1
                if typeinfo.label == 'op' or typeinfo.label == 'agg':
                    score = int(typeinfo.linktype[-1])
                    if typeinfo.label == 'op':
                        score += 6
                    else:
                        score += 11
                    for i in range(pstart, pend, 1):
                        final_question[i] = score

                elif typeinfo.label == 'col':
                    for i in range(pstart, pend, 1):
                        final_question[i] = 4
                    if final_header[typeinfo.index] % 2 == 0:
                        final_header[typeinfo.index] += 1

                elif typeinfo.label == 'val':
                    if typeinfo.index == -1:
                        for i in range(pstart, pend, 1):
                            final_question[i] = 5
                    else:
                        for i in range(pstart, pend, 1):
                            final_question[i] = 2
                        final_question[pstart] = 1
                        final_question[pend - 1] = 3
                        if final_header[typeinfo.index] < 2:
                            final_header[typeinfo.index] += 2

            # collect schema_link
            schema_link = []
            for sl in newtypeinfos:
                if sl.label in ['val', 'col']:
                    schema_link.append({
                        'question_len':
                        max(0, sl.pend - sl.pstart + 1),
                        'question_index': [sl.pstart, sl.pend],
                        'question_span':
                        ''.join(nlu_t[sl.pstart:sl.pend + 1]),
                        'column_index':
                        sl.index,
                        'column_span':
                        table['header_name'][sl.index]
                        if sl.index != -1 else '空列',
                        'label':
                        sl.label,
                        'weight':
                        round(sl.weight, 4)
                    })

            # get the match score of each table
            match_score = self.get_table_match_score(nlu_t, schema_link)

            # cal table_score
            if history_sql is not None and 'from' in history_sql:
                table_score = int(table['table_id'] == history_sql['from'][0])
            else:
                table_score = 0

            search_result = {
                'table_id': table['table_id'],
                'question_knowledge': final_question,
                'header_knowledge': final_header,
                'schema_link': schema_link,
                'match_score': match_score,
                'table_score': table_score
            }
            search_result_list.append(search_result)

        search_result_list = sorted(
            search_result_list,
            key=lambda x: (x['match_score'], x['table_score']),
            reverse=True)[0:1]

        return search_result_list
