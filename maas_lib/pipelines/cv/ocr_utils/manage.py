import os
from os.path import join, exists, abspath
import sys
import json
import glob

SRC_DIR = abspath('./')
SHARED_LIBRARY_NAME = 'libseglink.so'

def build_op():
    build_dir = join(SRC_DIR, 'cpp/build')
    if not exists(build_dir):
        os.mkdir(build_dir)
    os.chdir(build_dir)
    if not exists('Makefile'):
        os.system('cmake -DCMAKE_BUILD_TYPE=Release ..')
    os.system('make -j16')
    os.system('cp %s %s' % (SHARED_LIBRARY_NAME, join(SRC_DIR, SHARED_LIBRARY_NAME)))
    print('Building complete')


def clean_op():
    build_dir = join(SRC_DIR, 'cpp/build')
    print('Deleting recursively: %s' % build_dir)
    os.system('rm -rI %s' % build_dir)
    os.system('rm %s' % join(SRC_DIR, SHARED_LIBRARY_NAME))
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 manage.py <function-name>')
    else:
        fn_name = sys.argv[1]
        eval(fn_name + "()")