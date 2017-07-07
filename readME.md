# a project.
just upload to keep in version with other members.



# directory:
* --data  :     data to train and test.

    |--test_data : too big to upload;

    |--tf-records : too big to upload;

    |--train : too big to upload;
* --test  : to test trained model.

* --train : to train model.

* --train_log : tensorflow generated checkpoints, to save and restore trained model.


# to run:
### in normal way:
1. run ./train/CNN_for_tfrecords.py to train model.

2. run ./test/test_CNN_no_refactor.py to test model.

3. done.

### however :
as checkpoints data is saved in directory ./train_log, no bother to train again. Just run the second step, and you will see.

## plus:
To inference a single image as female or male and ite respective probabilities, run ./inference/inference_single_image.py with code like:

``` python
python inference_single_image.py image_path train_log_path

# for example:

## this tells the os the python project's path.
export PYTHONPATH=/media/oukohou/E/gong_an_pictures0612/gong_an_pictures/
## this runs the script.
python inference_single_image.py /home/oukohou/study/jingjingZhang/avatarMe.jpg /media/oukohou/E/gong_an_pictures0612/gong_an_pictures/train_log
```


### note:
        ./test/test_CNN_with_checkpoints.py will raise error,
        and is now not suppoorted to run.