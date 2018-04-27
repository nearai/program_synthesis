import time

import boto3

from program_synthesis.tools.output.streams.base import OutputStream


class S3OutputStream(OutputStream):
    def __init__(self, name, bucket, prefix, role_name=None):
        self._name = name
        self._bucket = bucket
        self._prefix = prefix
        self._role_name = role_name
        self._firehose = boto3.client('firehose')
        self._s3 = boto3.client('s3')
        self._is_open = False
        self._firehose_status = False
        self._last_firehose_status_check = None
        self._buffer_interval_in_seconds = 60

    def open(self):
        if self._role_name is None:
            raise Exception('Role name must be specified to open stream')

        role_arn = "arn:aws:iam::561269978991:role/{}".format(self._role_name)
        self._firehose.create_delivery_stream(
            DeliveryStreamName=self._name,
            DeliveryStreamType='DirectPut',
            S3DestinationConfiguration={
                'RoleARN': role_arn,
                'BucketARN': "arn:aws:s3:::{}".format(self._bucket),
                'Prefix': self._prefix,
                'BufferingHints': {
                    'SizeInMBs': 1,
                    'IntervalInSeconds': self._buffer_interval_in_seconds,
                },
            },
        )
        self._is_open = True

    def _check_firehouse_status(self):
        if self._last_firehose_status_check is not None \
                and time.time() - self._last_firehose_status_check < 5:
            return self._firehose_status

        response = self._firehose.describe_delivery_stream(
            DeliveryStreamName=self._name,
        )
        self._last_firehose_status_check = time.time()
        description = response['DeliveryStreamDescription']
        self._firehose_status = description['DeliveryStreamStatus']
        return self._firehose_status

    def _upload_manual_s3_log(self, data):
        # 0_ prefix is necessary for proper ordering in the ListObject API
        # that is used when reading from the stream
        timestamp = round(time.time() * 100)
        key = self._prefix + "0_manual/{}".format(timestamp)
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=str.encode(data),
        )

    def write(self, data):
        if not self._is_open:
            self.open()

        data += '\n'
        if self._firehose_status == 'ACTIVE':
            self._firehose.put_record(
                DeliveryStreamName=self._name,
                Record={'Data': str.encode(data)},
            )
        else:
            self._upload_manual_s3_log(data)
            self._check_firehouse_status()

    def close(self):
        self._firehose.delete_delivery_stream(DeliveryStreamName=self._name)

    def _order_all_listed_objects(self, last_key=None):
        continuation_token = None
        contents = []
        while True:
            kwargs = {}
            if last_key is not None:
                kwargs['StartAfter'] = last_key
            if continuation_token is not None:
                kwargs['ContinuationToken'] = continuation_token
            response = self._s3.list_objects_v2(
                Bucket=self._bucket,
                Prefix=self._prefix,
                **kwargs,
            )
            if 'Contents' in response:
                contents += response['Contents']
            if response['IsTruncated']:
                continuation_token = response['ContinuationToken']
                continue

            break

        return sorted(
            contents,
            key=lambda x: x['LastModified'],
        )

    def read(self):
        try:
            is_readable = self._check_firehouse_status() \
                          in ('CREATING', 'ACTIVE')
        except self._firehose.exceptions.ResourceNotFoundException:
            is_readable = False

        if not is_readable:
            raise Exception('Stream is not active, use aggregate S3 bucket '
                            'if you are sure that data exists.')

        last_key = last_update = final_flush_timestamp = None
        while True:
            list_contents = self._order_all_listed_objects(last_key)
            if len(list_contents) == 0:
                update_limit_minutes = 5
                if final_flush_timestamp is not None:
                    if time.time() > final_flush_timestamp:
                        return
                elif self._check_firehouse_status() == 'DELETING':
                    final_flush_timestamp = (
                            time.time() +
                            self._buffer_interval_in_seconds +
                            30
                    )
                elif last_update is not None \
                        and time.time() - last_update > update_limit_minutes * 60:
                    raise Exception("Active stream has not received update in "
                                    "{} minutes.".format(update_limit_minutes))
                time.sleep(5)
                continue

            last_update = time.time()
            for metadata in list_contents:
                object_response = self._s3.get_object(
                    Bucket=self._bucket,
                    Key=metadata['Key'],
                )
                yield object_response['Body'].read().decode('utf-8')

            last_key = list_contents[-1]['Key']
