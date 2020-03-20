import os
import zipfile

from hparams import train_folder, valid_folder, test_a_folder, test_b_folder


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def extract(folder):
    filename = '{}.zip'.format(folder)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


if __name__ == '__main__':
    # parameters
    ensure_folder('data')

    # if not os.path.isdir(train_image_folder):
    extract(train_folder)

    # if not os.path.isdir(valid_image_folder):
    extract(valid_folder)

    # if not os.path.isdir(test_a_image_folder):
    extract(test_a_folder)

    # if not os.path.isdir(test_b_image_folder):
    extract(test_b_folder)
