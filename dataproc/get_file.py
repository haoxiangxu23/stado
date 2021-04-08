import os
import argparse

def get_files(path):
    out = {}

    for root, dirs, files in os.walk(path):
        if root and not dirs:
            out[root] = files
        
    return out


def get_catagory_and_files(path):
    cats = os.listdir(path)
    cat_files = {}
    for cat in cats:
        if not os.path.isfile(os.path.join(path,cat)):
            cat_files[cat] = os.listdir(os.path.join(path,cat))

    return cat_files

