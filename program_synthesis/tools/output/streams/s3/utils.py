# This was ripped from https://gist.github.com/jasonrdsouza/f2c77dedb8d80faebcf9
'''
This script performs efficient concatenation of files stored in S3. Given a
folder, output location, and optional suffix, all files with the given suffix
will be concatenated into one file stored in the output location.

Concatenation is performed within S3 when possible, falling back to local
operations when necessary.

Run `python combineS3Files.py -h` for more info.
'''

import logging
import os

import boto3

logging.basicConfig(format='%(asctime)s => %(message)s')


class S3ConcatenateHelper(object):
    def __init__(self, bucket):
        self._bucket = bucket

    def run_concatenation(self, folder_to_concatenate, result_filepath, max_filesize=5e9):
        s3 = self.new_s3_client()
        parts_list = self._list_all_objects_with_size(s3, folder_to_concatenate)
        logging.warning(
            "Found {} parts to concatenate in {}/{}".format(len(parts_list), self._bucket, folder_to_concatenate)
        )
        grouped_parts_list = self.chunk_by_size(parts_list, max_filesize)
        logging.warning("Created {} concatenation groups".format(len(grouped_parts_list)))
        for i, parts in enumerate(grouped_parts_list):
            logging.warning("Concatenating group {}/{}".format(i, len(grouped_parts_list)))
            self.run_single_concatenation(s3, parts, "{}-{}".format(result_filepath, i))

    def run_single_concatenation(self, s3, parts_list, result_filepath):
        if len(parts_list) > 1:
            # perform multi-part upload
            upload_id = self.initiate_concatenation(s3, result_filepath)
            parts_mapping = self.assemble_parts_to_concatenate(s3, result_filepath, upload_id, parts_list)
            self.complete_concatenation(s3, result_filepath, upload_id, parts_mapping)
        elif len(parts_list) == 1:
            # can perform a simple S3 copy since there is just a single file
            resp = s3.copy_object(Bucket=self._bucket, CopySource="{}/{}".format(self._bucket, parts_list[0][0]),
                                  Key=result_filepath)
            logging.warning("Copied single file to {} and got response {}".format(result_filepath, resp))
        else:
            logging.warning("No files to concatenate for {}".format(result_filepath))
            pass

    @staticmethod
    def chunk_by_size(parts_list, max_filesize):
        grouped_list = []
        current_list = []
        current_size = 0
        for p in parts_list:
            current_size += p[1]
            current_list.append(p)
            if current_size > max_filesize:
                grouped_list.append(current_list)
                current_list = []
                current_size = 0

        if len(grouped_list) == 0:
            return [current_list]
        return grouped_list

    @staticmethod
    def new_s3_client():
        # initialize an S3 client with a private session so that multithreading
        # doesn't cause issues with the client's internal state
        session = boto3.session.Session()
        return session.client('s3')

    def _list_all_objects_with_size(self, s3, folder):
        def resp_to_filelist(resp):
            return [(x['Key'], x['Size']) for x in resp['Contents']]

        objects_list = []
        resp = s3.list_objects(Bucket=self._bucket, Prefix=folder)
        objects_list.extend(resp_to_filelist(resp))
        while resp['IsTruncated']:
            # if there are more entries than can be returned in one request, the key
            # of the last entry returned acts as a pagination value for the next request
            logging.warning("Found {} objects so far".format(len(objects_list)))
            last_key = objects_list[-1][0]
            resp = s3.list_objects(Bucket=self._bucket, Prefix=folder, Marker=last_key)
            objects_list.extend(resp_to_filelist(resp))

        return objects_list

    def initiate_concatenation(self, s3, result_filename):
        # performing the concatenation in S3 requires creating a multi-part upload
        # and then referencing the S3 files we wish to concatenate as "parts" of that upload
        resp = s3.create_multipart_upload(Bucket=self._bucket, Key=result_filename)
        logging.warning("Initiated concatenation attempt for {}, and got response: {}".format(result_filename, resp))
        return resp['UploadId']

    def assemble_parts_to_concatenate(self, s3, result_filename, upload_id, parts_list):
        parts_mapping = []
        part_num = 0

        # S3 multi-part upload parts must be larger than 5mb
        min_s3_size = 6000000

        s3_parts = ["{}/{}".format(self._bucket, p[0]) for p in parts_list if p[1] > min_s3_size]
        local_parts = [p[0] for p in parts_list if p[1] <= min_s3_size]

        # assemble parts large enough for direct S3 copy
        for part_num, source_part in enumerate(s3_parts, 1):  # part numbers are 1 indexed
            resp = s3.upload_part_copy(Bucket=self._bucket,
                                       Key=result_filename,
                                       PartNumber=part_num,
                                       UploadId=upload_id,
                                       CopySource=source_part)
            logging.warning(
                "Setup S3 part #{}, with path: {}, and got response: {}".format(part_num, source_part, resp))
            parts_mapping.append({'ETag': resp['CopyPartResult']['ETag'][1:-1], 'PartNumber': part_num})

        # assemble parts too small for direct S3 copy by downloading them locally,
        # combining them, and then reuploading them as the last part of the
        # multi-part upload (which is not constrained to the 5mb limit)
        small_parts = []
        for source_part in local_parts:
            temp_filename = "/tmp/{}".format(source_part.replace("/", "_"))
            s3.download_file(Bucket=self._bucket, Key=source_part, Filename=temp_filename)

            with open(temp_filename, 'rb') as f:
                small_parts.append(f.read().decode('utf-8'))
            os.remove(temp_filename)
            logging.warning("Downloaded and copied small part with path: {}".format(source_part))

        if len(small_parts) > 0:
            last_part_num = part_num + 1
            last_part = ''.join(small_parts)
            resp = s3.upload_part(Bucket=self._bucket, Key=result_filename, PartNumber=last_part_num,
                                  UploadId=upload_id,
                                  Body=last_part)
            logging.warning(
                "Setup local part #{} from {} small files, and got response: {}".format(last_part_num, len(small_parts),
                                                                                        resp))
            parts_mapping.append({'ETag': resp['ETag'][1:-1], 'PartNumber': last_part_num})

        return parts_mapping

    def complete_concatenation(self, s3, result_filename, upload_id, parts_mapping):
        if len(parts_mapping) == 0:
            s3.abort_multipart_upload(Bucket=self._bucket, Key=result_filename, UploadId=upload_id)
            logging.warning(
                "Aborted concatenation for file {}, with upload id #{} due to empty parts mapping".format(
                    result_filename,
                    upload_id))
        else:
            s3.complete_multipart_upload(Bucket=self._bucket, Key=result_filename, UploadId=upload_id,
                                         MultipartUpload={'Parts': parts_mapping})
            logging.warning(
                "Finished concatenation for file {}, with upload id #{}, and parts mapping: {}".format(result_filename,
                                                                                                       upload_id,
                                                                                                       parts_mapping))
