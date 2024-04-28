import os
import time
from sys import getsizeof as getsize


list_log_msg = []


def log(msg:str):
    print(msg)
    global list_log_msg
    list_log_msg.append(msg)


def save_training_log(args, file_name='training_log.txt'):
    global list_log_msg

    pth = '{}{}'.format(args.path_save_keypoints, args.test_name)
    if not os.path.exists(pth):
        os.mkdir(pth)

    with open('{}/{}'.format(pth, file_name), 'a') as f:
        now_time = time.ctime()
        f.write(str(now_time)+'\n')

        for msg in list_log_msg:
            f.write(msg+'\n')
        f.close()

    list_log_msg.clear()


# get var size (KB/MB)
def binary_conversion(var):
    size = getsize(var)
    assert isinstance(size, int)
    if size <= 1024:
        return f'{size/1024} KB'
    else:
        return f'{size/1024**2} MB'