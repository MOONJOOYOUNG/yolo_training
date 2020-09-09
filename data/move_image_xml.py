import os
import shutil
import argparse

parser = argpartParser(description='Move xml & jpg')
parser.add_argument('--path', default='./dataset/train', type=str, help='Files directory')
args = parser.parse_args()
# python move_image_xml.py --path ./dataset/train
def check_directory(path):
    if type(path) != str: return

    if not os.path.exists(path):
        os.makedirs(path)

def main():
    annotation_path = os.path.join(args.path,'annotation',)
    image_path = os.path.join(args.path,'image')
    # check directory
    check_directory(annotation_path)
    check_directory(image_path)

    # read files
    files = os.listdir(args.path)
    for file in files:
        if (file == '.DSStore' or file == '.DS_Store' or file == '._.DS_Store'):
            continue

        split_file = file.split('.')
        if len(split_file) <= 1: continue
        extension = split_file[1]

        if extension == 'xml' or extension == 'txt':
            save_path = annotation_path
        else:
            save_path = image_path

        # move(src, dir)
        shutil.move(os.path.join(args.path, file), os.path.join(save_path, file))

if __name__ == "__main__":
    main()