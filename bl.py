import re
import json
import uuid
import cStringIO
import base64
from datetime import datetime
from PIL import Image

from s3 import upload_to_s3
from parions_scanner import getBetsFromImage, MatchesDateNotSupported

re_image_name = re.compile(r'images\/(.+)\.')


def save_image(b64image_string):
    image_string = cStringIO.StringIO(base64.b64decode(b64image_string))  # decode image
    image = Image.open(image_string)
    image_id = str(uuid.uuid4())
    image_path = 'images/' + image_id + '.jpg'  # save to folder
    image.save(image_path)
    return image_path, image_id


def save_image_result_to_json(file_path, result):
    with open(file_path, 'w+') as json_file:
        json.dump(result, json_file)


def save_image_result(image_path, result):
    image_url = upload_to_s3(image_path)
    save_image_result_to_json(
        'results/{}.json'.format(re_image_name.findall(image_path).pop()),
        {
            'image_url': image_url,
            'date': datetime.now().isoformat(),
            'result': result,
        }
    )


def parse_image(image_path, csvfile, csvfileloto, check_date):
    # run image-processing code from atillo
    result = []
    try:
        jobdata = getBetsFromImage(image_path, csvfile, csvfileloto, check_date)
    except MatchesDateNotSupported:
        result.append({
            'status': 'Error',
            'message': 'Matches date not supported'
        })
        return result
    try:
        jobdata['betdata']
    except Exception:
        result.append({
            'status': 'Error',
            'message': jobdata
        })
    else:
        # parse results and send message to client
        date = jobdata['betdata'][3]
        if date:
            date = date.strftime('%d/%m/%Y')
        for i in range(len(jobdata['betdata'][1])):
            if jobdata['betdata'][0][0] == 'loto foot':
                selection = jobdata['betdata'][1][i][1]
                result.append({
                    'status': 'Result',
                    'id': '1/N/2',
                    'match': jobdata['betdata'][1][i][0],
                    'selection': selection[0] if len(selection) else 'x',
                    'reference': jobdata['betdata'][0][1][0],
                    'index': i + 1,
                    'type': 'loto_foot',
                    'first_match_date': date,
                })
            else:
                bet_info = jobdata['betdata'][2][i].split(' ')
                result.append({
                    'status': 'Result',
                    'id': '1/N/2',
                    'match': jobdata['betdata'][1][i][0],
                    'selection': bet_info[1],
                    'reference': bet_info[0],
                    'first_match_date': date,
                })
        result.append({
            'status': 'Finished',
        })
    return result
