import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

cs_path = osp.join(this_dir, '..', 'data', 'cityscapesScripts')
add_path(cs_path)
