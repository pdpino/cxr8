"""Split images into train and validation.

Note that the test set remains the same
"""

import csv
import os
import random
import argparse

import utils


def generate_txts(dataset_dir, split_size=1/8):
    """Split train_val list into train and validation lists randomly.
    
    args:
      split_size: minimum size for the validation dataset
    """

    with open(os.path.join(dataset_dir, 'train_val_list.txt')) as f:
        image_names = f.read().split('\n')

    images_by_patient = dict()
    for image_name in image_names:
        patient_id_str = image_name[:8]
        if patient_id_str not in images_by_patient:
            images_by_patient[patient_id_str] = []

        images_by_patient[patient_id_str].append(image_name)

    # shuffle
    patients = list(images_by_patient)
    random.shuffle(patients)

    # split them into train and val
    train_list = []
    val_list = []
    min_val_amount = len(image_names) * split_size

    for patient in patients:
        patient_images = images_by_patient[patient]

        if len(val_list) < min_val_amount:
            val_list += patient_images
        else:
            train_list += patient_images

    train_list.sort()
    val_list.sort()

    # write files
    utils.write_to_txt(train_list, os.path.join(dataset_dir, 'train_list.txt'))
    utils.write_to_txt(val_list, os.path.join(dataset_dir, 'val_list.txt'))

    train_samples = len(train_list)
    val_samples = len(val_list)
    total_samples = train_samples + val_samples
    
    print('Training and validation lists generated')
    print('\tTotal images: {}'.format(total_samples))
    print('\tTrain samples: {} ({:.2f} %)'.format(train_samples, train_samples/total_samples*100))
    print('\tValidation samples: {} ({:.2f} %)'.format(val_samples, val_samples/total_samples*100))

    
def generate_csvs(dataset_dir, dataset_types=['train', 'val']):
    """Generate label index csv file for train and validation."""
    
    data_entry_file = open(os.path.join(dataset_dir, 'Data_Entry_2017.csv'))

    image_names_by_type = {}
    for dataset_type in dataset_types:
        with open(os.path.join(dataset_dir, dataset_type + '_list.txt')) as f:
            images = [line.strip() for line in f.read().split('\n') if line]
            image_names_by_type[dataset_type] = set(images)
    
    csv_files = {
        t: open(os.path.join(dataset_dir, t + '_label.csv'), 'w+', newline='')
        for t in dataset_types
    }
    csv_writers = {
        t: csv.writer(csv_files[t])
        for t in dataset_types
    }

    # write header
    csv_header = ['FileName'] + list(utils.ALL_DISEASES)
    for dataset_type in dataset_types:
        csv_writers[dataset_type].writerow(csv_header)
    
    # read
    data_entry_lines = data_entry_file.read().splitlines()
    data_entry_file.close()

    #parse the file
    for data_entry_line in data_entry_lines[1:]: # Skip header
        file_name, label_string, *rest = data_entry_line.split(',')

        diseases_vector = [0 for _ in range(14)]
        if label_string != "No Finding":
            for label in label_string.split('|'):
                diseases_vector[utils.DISEASE_INDEX[label]] = 1

        output = [file_name] + diseases_vector
        
        # write to train, val, or test
        for dataset_type in dataset_types:
            image_names = image_names_by_type[dataset_type]
            if file_name in image_names:
                csv_writers[dataset_type].writerow(output)

    for dataset_type in dataset_types:
        csv_files[dataset_type].close()

    print('CSVs generated')

    
def parse_args():
    parser = argparse.ArgumentParser(description="Split train and validation data")
    
    parser.add_argument("--base-dir", default=".", type=str, help="Base directory to load the dataset from")
    # base_dir = '/mnt/data/chest-x-ray-8'
    parser.add_argument("--val-size", default=1/8, type=float, help="Minimum fraction for the validation set")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset_dir = os.path.join(args.base_dir, "dataset")

    generate_txts(dataset_dir, split_size=args.val_size)
    generate_csvs(dataset_dir)
    
