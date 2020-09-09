from lxml import etree

import os
import argparse

parser = argparse.ArgumentParser(description='Make image.txt')
parser.add_argument('--path', default='./dataset/train', type=str, help='Save directory')
args = parser.parse_args()
# python make_image_txt.py --path ./dataset/train
def main():
    f = open(os.path.join(args.path ,'image.txt'), 'w')

    image_files = os.listdir(os.path.join(args.path ,'image'))
    image_files = sorted(image_files)

    for file in image_files:
        if (file == '.DSStore' or file == '.DS_Store' or file == '._.DS_Store'):
            continue
        f.write(file + '\n')

    f.close()

if __name__ == "__main__":
    main()