import os
import json
import requests
import logging
from os import listdir
from os.path import isfile, join


JSON_FILE_PATH = 'results.json'
RESULT_FILES_DIR = 'results'
FOOTBALL_FORK_API_URL = 'http://app.parionsendirect.fr/api/image-result'


def get_json_data(json_file_path):
    with open(json_file_path, 'r') as json_file:
        file_data = []
        try:
            file_data = json.load(json_file)
        except ValueError:
            pass
        return file_data


def send_result(data):
    r = requests.post(FOOTBALL_FORK_API_URL, data=json.dumps(data))
    return r.status_code == 201


def deleteFile(file_path):
    pass


def get_result_files():
    return [
        join(RESULT_FILES_DIR, f)
        for f in listdir(RESULT_FILES_DIR)
        if isfile(join(RESULT_FILES_DIR, f)) and f.endswith('json')
    ]


def send_results_to_football_fork():
    result_files = get_result_files()
    results_len = len(result_files)
    logging.info('New image results: %s' % results_len)
    cnt = 0
    for result_file in result_files:
        cnt += 1
        data = get_json_data(result_file)
        if not isinstance(data, dict):
            return
        try:
            logging.info('Sending %s of %s' % (cnt, results_len))
            r = send_result(data)
            if not r:
                logging.info('Sending error')
                return
            else:
                os.remove(result_file)
        except Exception as e:
            logging.info('Upload failure, %s' % e)
            return
    logging.info('Upload success')
    logging.info('Done')


if __name__ == '__main__':
    send_results_to_football_fork()
