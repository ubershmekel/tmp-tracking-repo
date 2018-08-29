"""
Add .metadata.json files to a `flow_from_directory` folder structure
"""


import argparse
import json
import os


parser = argparse.ArgumentParser(description='Create metadata.json files.')

parser.add_argument(
    '--root',
    type=str,
    required=True,
    help='folder to add metadat files to')

args = parser.parse_args()
print("Adding metadata to: '{}".format(args.root))

for dirpath, dirnames, filenames in os.walk(args.root):
    for fname in filenames:
        if fname.lower().endswith('.jpg'):
            fpath = os.path.join(dirpath, fname)
            parts = fpath.split(os.path.sep)
            cls = parts[-2]
            metadata = {
                "original_phase": parts[-3],
                "class": cls,
                "name": parts[-1],
                "multi": "multiple" in cls,
                "gtdee": False if cls.lower() < "d" else True,
            }
            metadata_fname = fpath + ".metadata.json"
            with open(metadata_fname, 'w') as metadata_fhand:
                json.dump(metadata, metadata_fhand)


    



