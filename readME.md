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

### note:
        ./test/test_CNN_with_checkpoints.py will raise error,
        and is now not suppoorted to run.