import os
import shutil
import argparse
import tqdm

from lxml import etree

parser = argparse.ArgumentParser(description='Move xml & jpg')
parser.add_argument('--path', default='./dataset/train/', type=str, help='Save directory')
parser.add_argument('--file_name', default=1, type=int, help = '1 : train, 2 : val, 3 : test')
args = parser.parse_args()
# python voc_convert.py --path ./dataset/train --file_name 1
def main():
    # set file name
    if args.file_name == 1:
        file_name = './dataset/train/train.txt'
        image_name = './dataset/train/image.txt'
    elif args.file_name == 2:
        file_name = './dataset/val/val.txt'
        image_name = './dataset/val/image.txt'
    else:
        file_name = './dataset/test/test.txt'
        image_name = './dataset/test/image.txt'

    class_names = [c.strip() for c in open(os.path.join(args.path, 'data.names')).readlines()]
    ANNO_EXT = '.xml'

    image_filenames = os.listdir(os.path.join(args.path, 'image'))
    xml_filenames = os.listdir(os.path.join(args.path, 'annotation'))
    img_names = []
    for ith, image in tqdm.tqdm(enumerate(image_filenames)):
        img_names.append(image)

    xml_names = []
    for ith, image in tqdm.tqdm(enumerate(xml_filenames)):
        xml_name = image.split('.')[0]
        xml_names.append(xml_name)

    with open(image_name, 'r') as f, open(file_name, 'w') as wf:
        while True:
            line = f.readline().strip()
            if line is None or not line:
                break
            # im_p = os.path.join(image_dir, line + IMAGE_EXT)
            # an_p = os.path.join(anno_dir, line + ANNO_EXT)
            _line = line.split('.')[0]
            if _line in xml_names:
                im_p = os.path.join(args.path + '/image', line)
                an_p = os.path.join(args.path + '/annotation', _line + ANNO_EXT)

                # Get annotation.
                root = etree.parse(an_p).getroot()
                bboxes = root.xpath('//object/bndbox')
                names = root.xpath('//object/name')

                box_annotations = []
                for b, n in zip(bboxes, names):
                    name = n.text
                    class_idx = class_names.index(name)

                    xmin = b.find('xmin').text
                    ymin = b.find('ymin').text
                    xmax = b.find('xmax').text
                    ymax = b.find('ymax').text
                    box_annotations.append(','.join([str(xmin), str(ymin), str(xmax), str(ymax), str(class_idx)]))

                annotation = os.path.abspath(im_p) + ' ' + ' '.join(box_annotations) + '\n'

                wf.write(annotation)

if __name__ == "__main__":
    main()



