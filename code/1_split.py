"""
DeepSlide
Jason Wei, Behnaz Abdollahi, Saeed Hassanpour

This splits the train, val, and test data.
"""

import config
from os.path import basename, join, dirname
from utils import (
    confirm_output_folder,
    get_subfolder_paths,
    get_image_paths,
)


# main function
# note that we want the validation and test sets to be balanced
def split(all_wsi, train_folder, val_folder, test_folder, val_split,
          test_split, keep_orig_copy, labels_train, labels_val, labels_test):

    head = 'cp' if keep_orig_copy else 'mv'  # based on whether we want to move or keep the files

    # create folders
    for folder in [train_folder, val_folder, test_folder]:
        subfolders = [join(folder, _class) for _class in config.classes]
        for subfolder in subfolders:
            confirm_output_folder(subfolder)

    train_img_to_label = {}
    val_img_to_label = {}
    test_img_to_label = {}

    def move_set(folder, image_files, ops):
        """
        Return:
            a dictionary where
                key is (str)image_file_name and
                value is (str)image_class
        """
        def remove_topdir(filepath):
            """filepath should be a relative path
            ex) a/b/c.jpg -> b/c.jpg
            """
            first_delimiter_idx = filepath.find('/')
            return filepath[first_delimiter_idx + 1:]

        img_to_label = {}
        for image_file in image_files:
            output_path = join(folder, remove_topdir(image_file))
            os.system(f'{ops} {image_file} {output_path}')
            img_name = basename(image_file)
            img_class = basename(dirname(image_file))
            img_to_label[img_name] = img_class
        return img_to_label

    # sort the images and move/copy them appropriately
    subfolder_paths = get_subfolder_paths(all_wsi)
    for subfolder in subfolder_paths:

        image_paths = get_image_paths(subfolder)
        assert len(image_paths) > val_split + test_split
        # make sure we have enough slides in each class

        # assign training, test, and val images
        test_idx = len(image_paths) - test_split
        val_idx = test_idx - val_split
        train_images = image_paths[:val_idx]
        val_images = image_paths[val_idx:test_idx]
        test_images = image_paths[test_idx:]
        print('class {}:'.format(basename(subfolder)),
              '#train={}'.format(len(train_images)),
              '#val={} '.format(len(val_images)),
              '#test={}'.format(len(test_images)))

        # move train
        tmp_train_img_to_label = move_set(folder=train_folder,
                                          image_files=train_images,
                                          ops=head)
        train_img_to_label.update(tmp_train_img_to_label)

        # move val
        tmp_val_img_to_label = move_set(folder=val_folder,
                                        image_files=val_images,
                                        ops=head)
        val_img_to_label.update(tmp_train_img_to_label)

        # move test
        tmp_test_img_to_label = move_set(folder=test_folder,
                                         image_files=test_images,
                                         ops=head)

    # for making the csv files
    def write_to_csv(dest_filename, image_lable_dict):
        with open(dest_filename, 'w') as writer:
            writer.write('img,gt\n')
            for img in sorted(image_lable_dict.keys()):
                writer.write(img + ',' + image_lable_dict[img] + '\n')

    write_to_csv(dest_filename=labels_train,
                 image_lable_dict=train_img_to_label)
    write_to_csv(dest_filename=labels_val,
                 image_lable_dict=val_img_to_label)
    write_to_csv(dest_filename=labels_test,
                 image_lable_dict=test_img_to_label)


if __name__ == '__main__':

    split(all_wsi=config.all_wsi,
          train_folder=config.wsi_train,
          val_folder=config.wsi_val,
          test_folder=config.wsi_test,
          val_split=config.val_wsi_per_class,
          test_split=config.test_wsi_per_class,
          keep_orig_copy=config.keep_orig_copy,
          labels_train=config.labels_train,
          labels_val=config.labels_val,
          labels_test=config.labels_test)
