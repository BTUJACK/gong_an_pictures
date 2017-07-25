# -*- coding: utf-8 -*-
__author__ = 'oukohou'
__time__ = '2017/7/24 10:14'

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.

import os
import shutil


def copy_file_to_path(origianl_file_path, new_file_path, begin_number_of_img, end_number_of_img):
    for path, subdirs, files in os.walk(origianl_file_path):
        for filename in files:
            if not filename.lower().endswith("jpg"):
                print("got unexpected file: %s which is not image, pass..." % filename)
                continue
            file_number = int(filename.split(".")[0])
            if begin_number_of_img <= file_number < end_number_of_img:
                old_path = os.path.join(path, filename)
                label = path.split(os.sep)[-1]
                if label.startswith("F"):
                    new_path = os.path.join(new_file_path, '0', label[1:] + filename)
                elif label.startswith("M"):
                    new_path = os.path.join(new_file_path, '1', label[1:] + filename)
                else:
                    print("got unexpected file: %s which does not contain number." % filename)
                shutil.copyfile(old_path, new_path)


if __name__ == "__main__":
    original_file = "E:\gong_an_data\data"
    new_file = "E:\gong_an_pictures0612\gong_an_pictures\data\\train\gender_for_age_from_93_to_2000"
    copy_file_to_path(original_file, new_file, 2000, 2500)
