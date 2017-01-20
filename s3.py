import boto3
from botocore.client import Config

S3_ACCESS_KEY = 'AKIAJTDPQRZ4WJRW3DEA'
S3_SECRET_KEY = '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'
S3_BUCKET = 'magicboxarchive'
S3_REGION = 'eu-central-1'

s3client = None


def get_s3client():
    global s3client
    if s3client is None:
        s3client = boto3.client(
            's3',
            S3_REGION,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY, config=Config(signature_version='s3v4')
        )
    return s3client


def upload_to_s3(file_path):
    s3client = get_s3client()
    filename = file_path.split('/').pop()
    s3client.upload_file(
        file_path, S3_BUCKET, filename,
        ExtraArgs={
            'GrantRead': 'uri="http://acs.amazonaws.com/groups/global/AllUsers"',
            'ContentType': 'image/jpeg',
        }
    )
    return 'https://s3.{}.amazonaws.com/{}/{}'.format(
        S3_REGION, S3_BUCKET, filename
    )
