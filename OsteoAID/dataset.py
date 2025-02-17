
import os
import random
import shutil


def copy_files(src_dir, dst_dir, filenames, extension):
    os.makedirs(dst_dir, exist_ok=True)
    missing_files = 0
    for filename in filenames:
        src_path = os.path.join(src_dir, filename + extension)
        dst_path = os.path.join(dst_dir, filename + extension)


        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: File not found for {filename}")
            missing_files += 1

    return missing_files


def split_and_copy_dataset(image_dir, label_dir, output_dir, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):

    image_filenames = [os.path.splitext(f)[0] for f in os.listdir(image_dir)]

    # Randomly shuffle the list of filenames
    random.shuffle(image_filenames)

    # Calculate the number of training sets, verification sets, and test sets
    total_count = len(image_filenames)
    train_count = int(total_count * train_ratio)
    valid_count = int(total_count * valid_ratio)
    test_count = total_count - train_count - valid_count

    # Define the output folder pat-
    train_image_dir = os.path.join(output_dir, 'train', 'images')
    train_label_dir = os.path.join(output_dir, 'train', 'labels')
    valid_image_dir = os.path.join(output_dir, 'valid', 'images')
    valid_label_dir = os.path.join(output_dir, 'valid', 'labels')
    test_image_dir = os.path.join(output_dir, 'test', 'images')
    test_label_dir = os.path.join(output_dir, 'test', 'labels')


    train_missing_files = copy_files(image_dir, train_image_dir, image_filenames[:train_count], '.jpg')
    train_missing_files += copy_files(label_dir, train_label_dir, image_filenames[:train_count], '.txt')

    valid_missing_files = copy_files(image_dir, valid_image_dir, image_filenames[train_count:train_count + valid_count],
                                     '.jpg')
    valid_missing_files += copy_files(label_dir, valid_label_dir,
                                      image_filenames[train_count:train_count + valid_count], '.txt')

    test_missing_files = copy_files(image_dir, test_image_dir, image_filenames[train_count + valid_count:], '.jpg')
    test_missing_files += copy_files(label_dir, test_label_dir, image_filenames[train_count + valid_count:], '.txt')

    # Print the count of each dataset
    print(f"Train dataset count: {train_count}, Missing files: {train_missing_files}")
    print(f"Validation dataset count: {valid_count}, Missing files: {valid_missing_files}")
    print(f"Test dataset count: {test_count}, Missing files: {test_missing_files}")



image_dir = 'image'
label_dir = 'label'
output_dir = './my_dataset'

split_and_copy_dataset(image_dir, label_dir, output_dir)

