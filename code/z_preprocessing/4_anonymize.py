import os
import random
from os import listdir
from os.path import isfile, join, isdir


def get_image_paths(folder):
    image_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) and '.DS_Store' not in f]
    return image_paths


def get_alpha_name(number):
    alphas = "abcdefghijklmnopqrstuvwxyz"
    first = int(number / 26 / 26)
    second = int(number / 26) % 26
    third = number % 26
    return alphas[first] + alphas[second] + alphas[third]


# create an output folder if it does not already exist
def confirm_output_folder(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


# parameters
input_folder = 'wsi_all_3x'
output_folder = 'wsi_3x'
train_split = 0.5
val_split = 0.15
test_split = 0.35

# get all the subfolders
all_input_folders = [join(input_folder, f) for f in listdir(input_folder) if isdir(join(input_folder, f))]

# get list of all images
all_image_paths = []
for folder in all_input_folders:
    all_image_paths += get_image_paths(folder)

# important dictionaries
image_path_to_pid = {}
pid_to_image_paths = {}

# fill the dictionaries
for image_path in all_image_paths:

    image_name = image_path.split('/')[-1]

    # parse the image_name for the patient id
    start = image_name.index('S')
    end = image_name.index('.')
    while image_name[end].lower() not in 'qwertyuiopasdfghjklzxcvbnm':
        end -= 1
    assert end > start
    pid = image_name[start:end]

    # add to dictionaries
    image_path_to_pid[image_path] = pid
    if pid in pid_to_image_paths:
        pid_to_image_paths[pid].append(image_path)
    else:
        pid_to_image_paths[pid] = [image_path]

print(len(image_path_to_pid), 'images from', len(pid_to_image_paths), 'patients')
patient_list = list(pid_to_image_paths.keys())
random.seed(0)
random.shuffle(patient_list)
num_patients = len(patient_list)

# partition data
train_patients = patient_list[:int(train_split*num_patients)]
val_patients = patient_list[int(train_split*num_patients):int((train_split+val_split)*num_patients)]
test_patients = patient_list[int((train_split+val_split)*num_patients):]

train_images = []
for pid in train_patients:
    train_images += pid_to_image_paths[pid]

val_images = []
for pid in val_patients:
    val_images += pid_to_image_paths[pid]

test_images = []
for pid in test_patients:
    test_images += pid_to_image_paths[pid]

print('train_patients:', len(train_patients), 'with', len(train_images), 'images')
print('val_patients:', len(val_patients), 'with', len(val_images), 'images')
print('test_patients:', len(test_patients), 'with', len(test_images), 'images')

# encode image names
all_image_paths_ordered = train_images + val_images + test_images
image_path_to_code = {}
for i, image_path in enumerate(all_image_paths_ordered):
    image_path_to_code[image_path] = get_alpha_name(i)

folder_to_count = {}

writer = open(join(input_folder, 'image_to_pid.tsv'), 'w')
# copy all the images into the new locations!
for image_set, set_type in zip([train_images, val_images, test_images], ['train', 'val', 'test']):
    for image_path in image_set:
        label = image_path.split('/')[-2]

        folder = '/'.join([output_folder, set_type, label])

        if folder in folder_to_count:
            folder_to_count[folder] += 1
        else:
            folder_to_count[folder] = 1

        new_path = '/'.join([output_folder, set_type, label, image_path_to_code[image_path]+'.jpg'])

        write_line = new_path + '\t' + image_path
        writer.write(write_line + '\n')

print(folder_to_count)

test_img_file = open('wsi_all_3x/new_test_images.txt', 'r').readlines()
test_imgs = [x[:-1] for x in test_img_file]
test_patients = []

img_to_pid = {image_path_to_code[img_path]: image_path_to_pid[img_path] for img_path in image_path_to_pid}

for test_img in test_imgs:
    test_patients.append(img_to_pid[test_img[:3]])

print(len(set(test_patients)))
