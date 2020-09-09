from lxml import etree

import os
import argparse

parser = argparse.ArgumentParser(description='Make class names file')
parser.add_argument('--path', default='./dataset/train', type=str, help='annotation directory')
args = parser.parse_args()
# python make_names.py --path ./dataset/train
def main():
    labels_dict = {}

    anno_list = os.listdir(os.path.join(args.path,'annotation'))


    for anno_file in anno_list:
        if (anno_file == '.DSStore' or anno_file == '.DS_Store' or anno_file == '._.DS_Store'):
            continue

        p = os.path.join(args.path + '/annotation', anno_file)
        # Get annotation.
        root = etree.parse(p).getroot()
        names = root.xpath('//object/name')

        for n in names:
            labels_dict[n.text] = 0

    labels = list(labels_dict.keys())
    labels.sort()

    with open(os.path.join(args.path,'data.names'), 'w') as f:
        for l in labels:
            f.writelines(l + '\n')


if __name__ == "__main__":
    main()
