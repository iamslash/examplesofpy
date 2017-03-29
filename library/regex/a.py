# -*- coding: utf-8 -*-

import re

def test():
    p = re.compile(r'(?P<word>\b\w*\b)')
    m = p.search('(((( Lots of punctuation )))')
    print(m.group('word'))
    print(m.group(0))
    print(m.group(1))

if __name__ == "__main__":
    test()
