import os
import datetime

logs_folder = str
RUNNING_ID = str


def set_vars(logs_folder_, running_id):
    global logs_folder, RUNNING_ID
    logs_folder = logs_folder_
    RUNNING_ID = running_id


def log_function_start():
    message = "Function %s has started." % sys._getframe().f_back.f_code.co_name
    file_handler = open(os.path.join(logs_folder, '%s_log.txt' % RUNNING_ID), 'a')
    file_handler.write("\n" + "=" * 30 + "\n\n" + "{} - {}".format(datetime.datetime.now(), message))
    file_handler.close()


def log_function_end():
    message = "Function %s has ended." % sys._getframe().f_back.f_code.co_name
    file_handler = open(os.path.join(logs_folder, '%s_log.txt' % RUNNING_ID), 'a')
    file_handler.write("\n" + "=" * 30 + "\n\n" + "{} - {}".format(datetime.datetime.now(), message))
    file_handler.close()


def log_message(message):
    print(message)
    file_handler = open(os.path.join(logs_folder, '%s_log.txt' % RUNNING_ID), 'a')
    file_handler.write("\n" + "=" * 30 + "\n\n" + "{} - {}".format(datetime.datetime.now(), message))
    file_handler.close()


def create_dir(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


def check_file_existence(file_path):
    return os.path.exists(file_path)