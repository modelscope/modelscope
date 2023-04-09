# Copyright (c) Alibaba, Inc. and its affiliates.

import codecs
import re
import sys


class TextClean(object):

    def __init__(self):
        spu = [
            0xA0, 0x1680, 0x202f, 0x205F, 0x3000, 0xFEFF, 8203, 8206, 8207,
            8298, 8300, 65279
        ]
        spu.extend(range(0xE000, 0xF8FF + 1))
        spu.extend(range(0x2000, 0x200A + 1))
        spu.extend(range(0x7F, 0xA0 + 1))

        self.spaces = set([chr(i) for i in spu])

        self.space_pat = re.compile(r'\s+', re.UNICODE)

        self.replace_char = {
            u'`': u"'",
            u'’': u"'",
            u'´': u"'",
            u'‘': u"'",
            u'º': u'°',
            u'–': u'-',
            u'—': u'-'
        }

    def sbc2dbc(self, ch):
        n = ord(ch)
        if 0xFF00 < n < 0xFF5F:
            n -= 0xFEE0
        elif n == 0x3000:
            n = 0x20
        else:
            return ch
        return chr(n)

    def clean(self, s):
        try:
            line = list(s.strip())
            size = len(line)
            i = 0
            while i < size:
                if line[i] < u' ' or line[i] in self.spaces:
                    line[i] = u' '
                else:
                    line[i] = self.replace_char.get(line[i], line[i])
                    line[i] = self.sbc2dbc(line[i])

                i += 1
            line = ''.join(line)

            line = self.space_pat.sub(' ', line).strip()
            return line
        except Exception:
            return ''


if __name__ == '__main__':

    tc = TextClean()

    for line in sys.stdin:
        res = tc.clean(line)
        print(res)
