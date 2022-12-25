import gc
import arrow
import json
import math
import mimetypes
import os
import psutil
import re
import tempfile
import time
import uuid
from functools import reduce
from typing import Tuple, Union
from urllib import parse
import boto3
import cv2
import numpy as np
import pytesseract
from boto3.s3.transfer import TransferConfig
from deskew import determine_skew
from pytesseract import Output
from skimage.transform import rotate
from crop import detect_box
import logging
import multiprocessing
from pythonjsonlogger import jsonlogger
import sys
from functools import partial
# import _thread
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import yaml
from os.path import exists as file_exists
import threading
from multiprocessing import Pool
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, CancelledError
from PIL.ExifTags import TAGS
import platform
from Server import run
from tools import default, rotate_cv, crop_doc, bytesto, file_size, set_image_dpi_300, \
    set_image_dpi, cleanImg, canny2, opening, erode, dilate, thresholding, remove_noise, get_grayscale, validateJSON, \
    getDpi, get_size, biggestRectangle
import ray
from botocore.exceptions import ClientError

try:
    from PIL import Image
except ImportError:
    import Image
KB = 1024
MB = KB * KB
s3Client = boto3.client('s3')
sqs = boto3.client('sqs')
textract = boto3.client('textract', region_name='eu-west-1')
config = TransferConfig(
    multipart_threshold=1 * MB,
    max_concurrency=5,
    multipart_chunksize=1 * MB,
    max_io_queue=10000,
    io_chunksize=1 * MB,
    use_threads=True
)
Image.MAX_IMAGE_PIXELS = None
os.environ["OMP_THREAD_LIMIT"] = "1"
num_of_worker_threads = multiprocessing.cpu_count()
is_dev_local = platform.system() == "Darwin"
keyLocal = None
if platform.system() != "Darwin":
    os.environ['TESSDATA_PREFIX'] = "/usr/share/tesseract-ocr/tessdata/"
else:
    keyLocal = "files/1.jpg"

sys.setrecursionlimit(3000000)
num_cpus = psutil.cpu_count(logical=True)
ray.init(num_cpus=num_cpus,
         log_to_driver=False)
allowedDateFormats = [
    'DD/MM/YYYY',
    'D/M/YYYY',
    'DD/MM/YY',
    'D/M/YY',
    'DD/MM/YYYY',
    'DD.MM.YYYY',
    'DD-MM-YYYY',
    'D/M/YYYY',
    'D.M.YYYY',
    'D-M-YYYY',
    'D/M/YY',
    'D.M.YY',
    'D-M-YY',
    'DD/MM/YY',
    'DD.MM.YY',
    'DD-MM-YY',
    'DD/MMM/YYYY',
    'DD.MMM.YYYY',
    'DD-MMM-YYYY',
    'DD,MMM,YYYY',
    'D/MMM/YY',
    'D.MMM.YY',
    'D-MMM-YY',
    'D,MMM,YY',
    'DD MMMM YYYY',
    'DD MMMM, YYYY',
    'DD בMMMM YYYY',
    'DD לMMMM YYYY',
    'DD בMMMM, YYYY',
    'DD לMMMM, YYYY',
    'D MMMM, YYYY',
    'D בMMMM YYYY',
    'D לMMMM YYYY',
    'D בMMMM, YYYY',
    'D לMMMM, YYYY',
    'D MMM, YYYY',
    'D בMMM YYYY',
    'D לMMM YYYY',
    'D בMMM, YYYY',
    'D לMMM, YYYY',
    'D MMM, YY',
    'D בMMM YY',
    'D לMMM YY',
    'D בMMM, YY',
    'D לMMM, YY',
    'DD MMM YYYY',
    'DD MMM YY',
    'DD/MMMM/YYYY',
    'DD.MMMM.YYYY',
    'DD-MMMM-YYYY',
    'DD,MMMM,YYYY',
    'D/MMMM/YY YYYY/MM/DD',
    'D.MMMM.YY YYYY.MM.DD',
    'D-MMMM-YY YYYY-MM-DD',
    'D,MMMM,YY YYYY,MM,DD',
    'YYYY/M/D',
    'YYYY.M.D',
    'YYYY-M-D',
    'YY/MM/DD',
    'YY.MM.DD',
    'YY-MM-DD',
    'YY/MM/D',
    'YY.MM.D',
    'YY-MM-D',
    'YYYY/DD/MMM\'',
    'YYYY.DD.MMM\'',
    'YYYY-DD-MMM\'',
    'YYYY/MMM/DD\'',
    'YYYY.MMM.DD\'',
    'YYYY-MMM-DD\'',
    'YYYY,MMM,DD\'',
    'YYYY/D/MMM\'',
    'YYYY.D.MMM\'',
    'YYYY-D-MMM\'',
    'YYYY,D,MMM\'',
    'YYYY/MMM/D',
    'YYYY.MMM.D',
    'YYYY-MMM-D',
    'YYYY,MMM,D',
    'YYYY/DD/MMMM',
    'YYYY.DD.MMMM',
    'YYYY-DD-MMMM',
    'YYYY,DD,MMMM',
    'YYYY/D/MMMM MM/DD/YYYY',
    'YYYY.D.MMMM MM.DD.YYYY',
    'YYYY-D-MMMM MM-DD-YYYY',
    'YYYY,D,MMMM MM,DD,YYYY',
    'M/D/YYYY',
    'M.D.YYYY',
    'M-D-YYYY',
    'MM/DD/YY',
    'MM.DD.YY',
    'MM-DD-YY',
    'M/D/YY',
    'M.D.YY',
    'M-D-YY',
    'MMM/DD/YYYY',
    'MMM.DD.YYYY',
    'MMM-DD-YYYY',
    'MMM,DD,YYYY',
    'MMM/D/YY',
    'MMM.D.YY',
    'MMM-D-YY',
    'MMM,D,YY',
    'MMMM/DD/YYYY',
    'MMMM.DD.YYYY',
    'MMMM-DD-YYYY',
    'MMMM,DD,YYYY',
    'MMMM/D/YY',
    'MMMM D YY',
    'MMMM D YYYY',
    'MMMM D, YYYY',
    'MMM. DD, YYYY',
    'MMMM DD, YYYY',
    'MMMM DD,YYYY',
    'MMM DD, YYYY',
    'MMM DD,YYYY',
    'MMM.DD,YYYY',
    'MMM. D, YYYY',
    'MMMM D, YYYY',
    'MMMM D,YYYY',
    'MMM D, YYYY',
    'MMM D,YYYY',
    'MMM.D,YYYY',
    'MMMM.D.YY',
    'MMMM-D-YY',
    'MMMM,D,YY'
]


def print_log(fileId='', message='', message2='', message3='', message4=''):
    global logger
    if message2 != '':
        message2 = ' ' + str(message2)
    if message3 != '':
        message3 = ' ' + str(message3)
    if message4 != '':
        message4 = ' ' + str(message4)
    message = str(message) + str(message2) + str(message3) + str(message4)
    logger.info({
        "fileId": fileId,
        "message": message,
        "cpu_count": num_of_worker_threads,
        "cpu_percent": psutil.cpu_percent(),
        "memory_used_percent": psutil.virtual_memory().percent,
        "memory_available_percent": (
                psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
    })


def print_log_pro(message, logger_pro, fileIdKey):
    message = str(message)
    print({
        "fileId": fileIdKey,
        "message": message,
        "cpu_count": num_of_worker_threads,
        "cpu_percent": psutil.cpu_percent(),
        "memory_used_percent": psutil.virtual_memory().percent,
        "memory_available_percent": (
                psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
    })


def uploadingToBucket(path_of_file, targetBucketName, bucketKey, times_of_all_processes, fileId):
    times_of_all_processes['uploadingToBucket'] = time.time()
    try:
        print_log(fileId, 'Result: Succeeded to change angle')
        print_log(fileId, "Check mimetypes of the file")
        mime_type = 'image/jpeg'
        try:
            mime_type = mimetypes.guess_type(path_of_file)[0]
        except Exception as mime_err:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback_details = {
                'filename': exc_traceback.tb_frame.f_code.co_filename,
                'lineno': exc_traceback.tb_lineno,
                'function_name': exc_traceback.tb_frame.f_code.co_name,
                'type': exc_type.__name__,
                'message': str(exc_value)
            }
            del (exc_type, exc_value, exc_traceback)
            print_log(fileId, 'traceback_details: ', str(traceback_details))
            print_log(fileId, str(mime_err))

        print_log(fileId, "The mimetypes is: ", mime_type)
        print_log(fileId, 'Start uploading file to output bucket')
        boto3.client('s3').upload_file(path_of_file, targetBucketName, bucketKey, ExtraArgs={"ContentType": mime_type},
                             Config=config)
        print_log(fileId, 'Finished to upload the file to output bucket')
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_details = {
            'filename': exc_traceback.tb_frame.f_code.co_filename,
            'lineno': exc_traceback.tb_lineno,
            'function_name': exc_traceback.tb_frame.f_code.co_name,
            'type': exc_type.__name__,
            'message': str(exc_value)
        }
        del (exc_type, exc_value, exc_traceback)
        print_log(fileId, 'traceback_details: ', str(traceback_details))
        print_log(fileId, str(e))
        print_log(fileId, 'Error uploading file to output bucket')
    # if os.path.exists(path_of_file):
    #     os.remove(path_of_file)
    #     print_log(fileId,'The download_path removed from the tmp location!')
    # else:
    #     print_log(fileId,"The download_path does not exist in the path: ", path_of_file)
    times_of_all_processes['uploadingToBucket'] = (
        "{0:.0f}s".format(
            time.time() - times_of_all_processes['uploadingToBucket']))


def check_sums(data, fileId):
    seen_titles = set()
    new_list = []
    for obj in data:
        if obj["text"] not in seen_titles:
            new_list.append(obj)
            seen_titles.add(obj["text"])
    arrAllNum = []
    arrNum = []

    for number in new_list:
        # print_log(fileId,'-------')
        num = number['text']
        totalMaam = float(format((num * 0.17), ".2f"))
        originalWithoutMaam = float(format((num / 1.17), ".2f"))
        # from_maam = float(format(((num / 17) * 100), ".2f"))
        # fullMaamMinusWithoutMaam = float(format(num - (num / 1.17), ".2f"))
        halfPSpace_num = 0.5
        # if num is totalIncludedMaam so (num / 1.17) will give the total withpout maam and num - (num / 1.17) =  will give the maam amount
        # if num is total before maam so (num * 0.17) will give the maam amount and num + maam  =  will give the totalIncludedMaam
        # print_log(fileId,'text: ', num)
        # print_log(fileId,'totalMaam', totalMaam)
        # print_log(fileId,'originalWithoutMaam', originalWithoutMaam)
        # print_log(fileId,'-------')
        eq_to_originalWithoutMaam = next((item for item in new_list if item["text"] == originalWithoutMaam), None)
        if eq_to_originalWithoutMaam:
            # print_log(fileId,'eq_to_originalWithoutMaam: ', eq_to_originalWithoutMaam)
            # print_log(fileId,'Found originalWithoutMaam: ', eq_to_originalWithoutMaam["text"], 'the maam amount is: ',
            #           num - originalWithoutMaam)
            eq_to_maam = next((item for item in new_list if item["text"] == (num - originalWithoutMaam)), None)
            if eq_to_maam:
                # print_log(fileId,'eq_to_maam: ', eq_to_maam)
                arrAllNum.append([number, eq_to_originalWithoutMaam, eq_to_maam])
            else:
                eq_to_maam = next((item for item in new_list if (
                        (item["text"] >= ((num - originalWithoutMaam) - halfPSpace_num)) and (
                        item["text"] <= ((num - originalWithoutMaam) + halfPSpace_num)))), None)
                if eq_to_maam:
                    # print_log(fileId,'eq_to_maam: ', eq_to_maam)
                    arrAllNum.append([number, eq_to_originalWithoutMaam, eq_to_maam])
        else:
            eq_to_originalWithoutMaam = next(
                (item for item in new_list if ((item["text"] >= (originalWithoutMaam - halfPSpace_num)) and (
                        item["text"] <= (originalWithoutMaam + halfPSpace_num)))),
                None)
            if eq_to_originalWithoutMaam:
                # print_log(fileId,'eq_to_originalWithoutMaam: ', eq_to_originalWithoutMaam)
                # print_log(fileId,'Found originalWithoutMaam: ', eq_to_originalWithoutMaam["text"], 'the maam amount is: ',
                #           num - originalWithoutMaam)
                eq_to_maam = next((item for item in new_list if item["text"] == (num - originalWithoutMaam)), None)
                if eq_to_maam:
                    # print_log(fileId,'eq_to_maam: ', eq_to_maam)
                    arrAllNum.append([number, eq_to_originalWithoutMaam, eq_to_maam])
                else:
                    eq_to_maam = next((item for item in new_list if (
                            (item["text"] >= ((num - originalWithoutMaam) - halfPSpace_num)) and (
                            item["text"] <= ((num - originalWithoutMaam) + halfPSpace_num)))), None)
                    if eq_to_maam:
                        # print_log(fileId,'eq_to_maam: ', eq_to_maam)
                        arrAllNum.append([number, eq_to_originalWithoutMaam, eq_to_maam])

        eq_to_totalMaam = next((item for item in new_list if item["text"] == totalMaam), None)
        if eq_to_totalMaam:
            # print_log(fileId,'eq_to_totalMaam: ', eq_to_totalMaam)
            # print_log(fileId,'Found the maam amount: ', eq_to_totalMaam["text"], 'the totalIncludedMaam is: ',
            #           num + totalMaam)

            eq_to_totalIncludedMaam = next((item for item in new_list if item["text"] == (num + totalMaam)), None)
            if eq_to_totalIncludedMaam:
                # print_log(fileId,'eq_to_totalIncludedMaam: ', eq_to_totalIncludedMaam)
                arrAllNum.append([eq_to_totalIncludedMaam, number, eq_to_totalMaam])
            else:
                eq_to_totalIncludedMaam = next(
                    (item for item in new_list if (item["text"] >= ((num + totalMaam) - halfPSpace_num)) and (
                            item["text"] <= ((num + totalMaam) + halfPSpace_num))), None)
                if eq_to_totalIncludedMaam:
                    # print_log(fileId,'eq_to_totalIncludedMaam: ', eq_to_totalIncludedMaam)
                    arrAllNum.append([eq_to_totalIncludedMaam, number, eq_to_totalMaam])
        else:
            eq_to_totalMaam = next(
                (item for item in new_list if (item["text"] >= (totalMaam - halfPSpace_num)) and (
                        item["text"] <= (totalMaam + halfPSpace_num))),
                None)
            if eq_to_totalMaam:
                # print_log(fileId,'eq_to_totalMaam: ', eq_to_totalMaam)
                # print_log(fileId,'Found the maam amount: ', eq_to_totalMaam["text"], 'the totalIncludedMaam is: ',
                #           num + totalMaam)
                eq_to_totalIncludedMaam = next((item for item in new_list if item["text"] == (num + totalMaam)),
                                               None)
                if eq_to_totalIncludedMaam:
                    # print_log(fileId,'eq_to_totalIncludedMaam: ', eq_to_totalIncludedMaam)
                    arrAllNum.append([eq_to_totalIncludedMaam, number, eq_to_totalMaam])
                else:
                    eq_to_totalIncludedMaam = next(
                        (item for item in new_list if (item["text"] >= ((num + totalMaam) - halfPSpace_num)) and (
                                item["text"] <= ((num + totalMaam) + halfPSpace_num))), None)
                    if eq_to_totalIncludedMaam:
                        # print_log(fileId,'eq_to_totalIncludedMaam: ', eq_to_totalIncludedMaam)
                        arrAllNum.append([eq_to_totalIncludedMaam, number, eq_to_totalMaam])

    if len(arrAllNum) > 0:
        arrAllNum = sorted(arrAllNum, key=lambda dct: dct[0]['text'], reverse=True)
        arrNum = arrAllNum[0]
        arrNum = sorted(arrNum, key=lambda dct: dct['text'], reverse=True)
        arrNum[0]["type"] = 3
        arrNum[1]["type"] = 1
        arrNum[2]["type"] = 2

    return arrNum


def check_basics_sums(data, fileId):
    is_found_basic_match = []
    is_found_basic_match_exact = []
    all_is_found_basic_match = []
    seen_titles = set()
    new_list = []
    higer_sum = 0
    if len(data) > 0:
        higer_sum = float(data[0]["text"])

    try:
        for obj in data:
            if obj["text"] not in seen_titles:
                new_list.append(obj)
                seen_titles.add(obj["text"])

        for number in new_list:
            is_found = True
            # print_log(fileId,'-------')
            num = number['text']
            totalMaam = float(format((num * 0.17), ".2f"))
            originalWithoutMaam = float(format((num / 1.17), ".2f"))
            from_maam = float(format(((num / 17) * 100), ".2f"))
            fullMaamMinusWithoutMaam = float(format(num - (num / 1.17), ".2f"))
            # print_log(fileId,'text: ', num)
            # print_log(fileId,'totalMaam', totalMaam)
            # print_log(fileId,'originalWithoutMaam', originalWithoutMaam)
            # print_log(fileId,'from_maam', from_maam)
            # print_log(fileId,'fullMaamMinusWithoutMaam', fullMaamMinusWithoutMaam)
            # print_log(fileId,'-------')
            if not is_found:
                eq_to_originalWithoutMaam = next(
                    (item for item in new_list if (item["text"] != 0 and item["text"] == originalWithoutMaam)),
                    None)
                if eq_to_originalWithoutMaam:
                    # print_log(fileId,'1 Found originalWithoutMaam from total_include_maam: ', eq_to_originalWithoutMaam,
                    #           'the maam amount is: ',
                    #           num - originalWithoutMaam)
                    is_found_basic_match.append([number, eq_to_originalWithoutMaam])
                    is_found = True
            if not is_found:
                eq_to_totalMaam = next(
                    (item for item in new_list if (item["text"] != 0 and item["text"] == totalMaam)),
                    None)
                if eq_to_totalMaam:
                    # print_log(fileId,'2 Found the maam amount: ', eq_to_totalMaam, 'the totalIncludedMaam is: ', num + totalMaam)
                    is_found_basic_match.append([number, eq_to_totalMaam])
                    is_found = True
            if not is_found:
                eq_to_from_maam = next(
                    (item for item in new_list if (item["text"] != 0 and item["text"] == from_maam)),
                    None)
                if eq_to_from_maam:
                    # print_log(fileId,'3 Found originalWithoutMaam from maam: ', eq_to_from_maam, 'the maam amount is: ', num)
                    is_found_basic_match.append([number, eq_to_from_maam])
                    is_found = True
            if not is_found:
                eq_to_fullMaamMinusWithoutMaam = next(
                    (item for item in new_list if (item["text"] != 0 and item["text"] == fullMaamMinusWithoutMaam)),
                    None)
                if eq_to_fullMaamMinusWithoutMaam:
                    # print_log(fileId,'4 Found maam sum: ', fullMaamMinusWithoutMaam, ' from including maam amount: ', num)
                    is_found_basic_match.append([number, eq_to_fullMaamMinusWithoutMaam])
                    is_found = True

        if len(is_found_basic_match) > 0:
            is_found_basic_match_exact = True
            print_log(fileId, 'An exact match was found between 2 amounts')

        print_log(fileId, 'Start a rounded match between 2 amounts')
        for number in new_list:
            # print_log(fileId,'-------')
            num = number['text']
            totalMaam = float(format((num * 0.17), ".2f"))
            originalWithoutMaam = float(format((num / 1.17), ".2f"))
            from_maam = float(format(((num / 17) * 100), ".2f"))
            fullMaamMinusWithoutMaam = float(format(num - (num / 1.17), ".2f"))
            halfPSpace_num = 0.5
            # print_log(fileId,'text: ', num)
            # print_log(fileId,'totalMaam', totalMaam)
            # print_log(fileId,'originalWithoutMaam', originalWithoutMaam)
            # print_log(fileId,'from_maam', from_maam)
            # print_log(fileId,'fullMaamMinusWithoutMaam', fullMaamMinusWithoutMaam)
            # print_log(fileId,'-------')
            for x in new_list:
                is_found = False
                if x["text"] != 0:
                    if not is_found and (x["text"] >= (originalWithoutMaam - halfPSpace_num)) and (
                            x["text"] <= (originalWithoutMaam + halfPSpace_num)):
                        # print_log(fileId,'=======-------- 1 Found originalWithoutMaam from total_include_maam: ',
                        #           x["text"],
                        #           'the maam amount is: ',
                        #           num - originalWithoutMaam)
                        is_found_basic_match.append([x, number])
                        is_found = True

                    if not is_found and (x["text"] >= (totalMaam - halfPSpace_num)) and (
                            x["text"] <= (totalMaam + halfPSpace_num)):
                        # print_log(fileId,'=======-------- 2 Found the maam amount: ', x["text"],
                        #           'the totalIncludedMaam is: ',
                        #           num + totalMaam)
                        is_found_basic_match.append([x, number])
                        is_found = True

                    if not is_found and (x["text"] >= (from_maam - halfPSpace_num)) and (
                            x["text"] <= (from_maam + halfPSpace_num)):
                        # print_log(fileId,'=======-------- 3 Found originalWithoutMaam from maam: ', x["text"],
                        #           'the maam amount is: ',
                        #           num)
                        is_found_basic_match.append([x, number])
                        is_found = True

                    if not is_found and (x["text"] >= (fullMaamMinusWithoutMaam - halfPSpace_num)) and (
                            x["text"] <= (fullMaamMinusWithoutMaam + halfPSpace_num)):
                        # print_log(fileId,'=======-------- 4 Found maam sum: ', x["text"], ' from including maam amount: ',
                        #           num)
                        is_found_basic_match.append([x, number])
                        is_found = True

        if not is_found_basic_match_exact and len(is_found_basic_match) > 0:
            print_log(fileId, 'A rounded match was found between 2 amounts')

        if len(is_found_basic_match) > 0:
            for index_two_numbers in range(len(is_found_basic_match)):
                is_found_basic_match[index_two_numbers] = sorted(is_found_basic_match[index_two_numbers],
                                                                 key=lambda dct: dct['text'], reverse=True)
            is_found_basic_match = sorted(is_found_basic_match, key=lambda dct: dct[0]['text'], reverse=True)

            is_found_completed_not_full_maam = False
            if len(is_found_basic_match) > 1:
                unique_data = {str(each[0]['text']) + str(each[1]['text']): each for each in
                               is_found_basic_match}.values()
                is_found_basic_match = list(unique_data)
                # print(is_found_basic_match)
                for x_data in is_found_basic_match:
                    num_0 = float(x_data[0]["text"])
                    num_1 = float(x_data[1]["text"])
                    if abs((float(format((num_0 / 1.17), ".2f")) - float(format(num_1, ".2f"))) / 0.01) <= 1:
                        x_data[0]["type"] = 3
                        x_data[1]["type"] = 1
                    elif abs(((float(format((num_0 * 0.17), ".2f"))) - (float(format(num_1, ".2f")))) / 0.01) <= 1:
                        x_data[0]["type"] = 1
                        x_data[1]["type"] = 2
                    elif abs(((float(format(num_0 - (num_0 / 1.17), ".2f"))) - (
                            float(format(num_1, ".2f")))) / 0.01) <= 1:
                        x_data[0]["type"] = 3
                        x_data[1]["type"] = 2
                    else:
                        x_data[0]["type"] = None
                        x_data[1]["type"] = None
                    if x_data[0]["type"] == 1 and x_data[1]["type"] == 2:
                        # print_log(fileId,'----found case of 2 numbers BeforeMaam and Maam---', x_data[0]["text"],
                        #           x_data[1]["text"])
                        for number_inside in new_list:
                            number_int = number_inside["text"]
                            if float(number_int) > 0:
                                sums_with_maam = float(x_data[0]["text"]) + float(x_data[1]["text"])
                                num = sums_with_maam + float(number_int)
                                eq_to_incluning_maam = next(
                                    (item_inside for item_inside in new_list if (item_inside["text"] != 0 and (
                                            (item_inside["text"] == num) or (
                                            round(item_inside["text"]) == num)))),
                                    None)
                                # print('----found case of 2 numbers BeforeMaam and Maam---', x_data[0]["text"],
                                #       x_data[1]["text"], sums_with_maam, num, number_int, eq_to_incluning_maam)
                                if eq_to_incluning_maam is not None:
                                    is_found_completed_not_full_maam = True
                                    # if 'type' in x_data[0]:
                                    #     del x_data[0]["type"]
                                    # if 'type' in x_data[1]:
                                    #     del x_data[1]["type"]
                                    # if 'type' in eq_to_incluning_maam:
                                    #     del eq_to_incluning_maam["type"]
                                    eq_to_incluning_maam["type"] = 3
                                    x_data[0]["type"] = 1
                                    x_data[1]["type"] = 2
                                    number_inside["type"] = 4
                                    # print('-----is_found_completed_not_full_maam----', eq_to_incluning_maam)
                                    is_found_basic_match_inside = [eq_to_incluning_maam, x_data[0], x_data[1],
                                                                   number_inside]
                                    is_found_basic_match_inside = sorted(is_found_basic_match_inside,
                                                                         key=lambda dct: dct['text'],
                                                                         reverse=True)
                                    all_is_found_basic_match.append(is_found_basic_match_inside)

                if not is_found_completed_not_full_maam:
                    for x_data in is_found_basic_match:
                        num_0 = float(x_data[0]["text"])
                        num_1 = float(x_data[1]["text"])
                        if abs((float(format((num_0 / 1.17), ".2f")) - float(format(num_1, ".2f"))) / 0.5) <= 1:
                            x_data[0]["type"] = 3
                            x_data[1]["type"] = 1
                        elif abs(((float(format((num_0 * 0.17), ".2f"))) - (
                                float(format(num_1, ".2f")))) / 0.5) <= 1:
                            x_data[0]["type"] = 1
                            x_data[1]["type"] = 2
                        elif abs(((float(format(num_0 - (num_0 / 1.17), ".2f"))) - (
                                float(format(num_1, ".2f")))) / 0.5) <= 1:
                            x_data[0]["type"] = 3
                            x_data[1]["type"] = 2
                        else:
                            x_data[0]["type"] = None
                            x_data[1]["type"] = None
                        if x_data[0]["type"] == 1 and x_data[1]["type"] == 2:
                            # print_log(fileId,'----found case of 2 numbers BeforeMaam and Maam---', x_data[0]["text"],
                            #           x_data[1]["text"])
                            for number_inside in new_list:
                                number_int = number_inside["text"]
                                if float(number_int) > 0:
                                    sums_with_maam = float(x_data[0]["text"]) + float(x_data[1]["text"])
                                    num = sums_with_maam + float(number_int)
                                    eq_to_incluning_maam = next(
                                        (item_inside for item_inside in new_list if
                                         (item_inside["text"] != 0 and (
                                                 (item_inside["text"] == num) or (
                                                 round(item_inside["text"]) == round(num))))),
                                        None)
                                    # print('----found case of 2 numbers BeforeMaam and Maam---', x_data[0]["text"],
                                    #       x_data[1]["text"], sums_with_maam, num, number_int, eq_to_incluning_maam)
                                    if eq_to_incluning_maam is not None:
                                        is_found_completed_not_full_maam = True
                                        # if 'type' in x_data[0]:
                                        #     del x_data[0]["type"]
                                        # if 'type' in x_data[1]:
                                        #     del x_data[1]["type"]
                                        # if 'type' in eq_to_incluning_maam:
                                        #     del eq_to_incluning_maam["type"]
                                        eq_to_incluning_maam["type"] = 3
                                        x_data[0]["type"] = 1
                                        x_data[1]["type"] = 2
                                        number_inside["type"] = 4
                                        # print('-----is_found_completed_not_full_maam----', eq_to_incluning_maam)
                                        is_found_basic_match_inside = [eq_to_incluning_maam, x_data[0],
                                                                       x_data[1], number_inside]

                                        is_found_basic_match_inside = sorted(is_found_basic_match_inside,
                                                                             key=lambda dct: dct['text'],
                                                                             reverse=True)
                                        all_is_found_basic_match.append(is_found_basic_match_inside)

                if not is_found_completed_not_full_maam:
                    is_found_basic_match = is_found_basic_match[0]
                    num_0 = float(is_found_basic_match[0]["text"])
                    num_1 = float(is_found_basic_match[1]["text"])
                    if num_0 != num_1:
                        if abs((float(format((num_0 / 1.17), ".2f")) - float(format(num_1, ".2f"))) / 0.5) <= 1:
                            is_found_basic_match[0]["type"] = 3
                            is_found_basic_match[1]["type"] = 1
                        elif abs(((float(format((num_0 * 0.17), ".2f"))) - (float(format(num_1, ".2f")))) / 0.5) <= 1:
                            is_found_basic_match[0]["type"] = 1
                            is_found_basic_match[1]["type"] = 2
                        elif abs(((float(format(num_0 - (num_0 / 1.17), ".2f"))) - (
                                float(format(num_1, ".2f")))) / 0.5) <= 1:
                            is_found_basic_match[0]["type"] = 3
                            is_found_basic_match[1]["type"] = 2
                    else:
                        is_found_basic_match = []
            else:
                is_found_basic_match = is_found_basic_match[0]
                num_0 = float(is_found_basic_match[0]["text"])
                num_1 = float(is_found_basic_match[1]["text"])
                if num_0 != num_1:
                    if abs((float(format((num_0 / 1.17), ".2f")) - float(format(num_1, ".2f"))) / 0.5) <= 1:
                        is_found_basic_match[0]["type"] = 3
                        is_found_basic_match[1]["type"] = 1
                    elif abs(((float(format((num_0 * 0.17), ".2f"))) - (float(format(num_1, ".2f")))) / 0.5) <= 1:
                        is_found_basic_match[0]["type"] = 1
                        is_found_basic_match[1]["type"] = 2
                    elif abs(((float(format(num_0 - (num_0 / 1.17), ".2f"))) - (
                            float(format(num_1, ".2f")))) / 0.5) <= 1:
                        is_found_basic_match[0]["type"] = 3
                        is_found_basic_match[1]["type"] = 2
                else:
                    is_found_basic_match = []
    except Exception as error_check_basics_sums:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_details = {
            'filename': exc_traceback.tb_frame.f_code.co_filename,
            'lineno': exc_traceback.tb_lineno,
            'function_name': exc_traceback.tb_frame.f_code.co_name,
            'type': exc_type.__name__,
            'message': str(exc_value)
        }
        del (exc_type, exc_value, exc_traceback)
        print_log(fileId, 'traceback_details: ', str(traceback_details))
        print_log(fileId, 'Error_ocr: ', str(error_check_basics_sums))

    if len(all_is_found_basic_match) > 0:
        all_is_found_basic_match = sorted(all_is_found_basic_match, key=lambda dct: dct[0]['text'], reverse=True)
        is_found_basic_match = all_is_found_basic_match[0]
        is_found_basic_match[0]["type"] = 3
        is_found_basic_match[1]["type"] = 1
        is_found_basic_match[2]["type"] = 2
        is_found_basic_match[3]["type"] = 4

    return is_found_basic_match


def check_valid_hp(num, fileId):
    check_valid_hp_lambda = lambda idNum: 10 - (reduce(lambda a, b: a + b, list(map(int, idNum[0: -1: 2]))) + reduce(
        lambda a, b: a + (0, 2, 4, 6, 8, 1, 3, 5, 7, 9)[b],
        list(map(int, '0' + idNum[1:: 2])))) % 10 == int(
        idNum[-1])
    return check_valid_hp_lambda(num)


def get_lang_and_angle(image_rotate):
    custom_oem_psm_config = r'--dpi 300 --oem 1 --psm 0 -c tessedit_do_invert=0 -l osd'
    data_main = pytesseract.image_to_osd(image_rotate, config=custom_oem_psm_config)
    # print(data_main)
    lang_main = re.search('(?<=Script: )\w+', data_main).group(0)
    angle_orientation = re.search('(?<=Orientation in degrees: )\d+', data_main).group(0)
    angle_main = re.search('(?<=Rotate: )\d+', data_main).group(0)
    orientation_confidence = re.search('(?<=Orientation confidence: )\d+', data_main).group(0)
    print("The angle_orientation is:  " + str(angle_orientation))
    print("The language is:  " + lang_main)
    res_angle = float(angle_main)
    if res_angle < -45:
        res_angle = -(90 + res_angle)
    else:
        res_angle = -res_angle
    print("The angle (Heb) is:  " + str(res_angle))

    return {
        "orientation_confidence": float(orientation_confidence) > 0.5,
        "angle": angle_orientation,
        "lang": lang_main,
        "rotate": res_angle
    }


@ray.remote
def pytesseract_image_to_data(idx, grayImage, byte_im, len_ac, download_path, logger_pro, fileIdKey):
    try:
        custom_oem_psm_config1 = r'--oem 1 --psm 3'
        # custom_oem_psm_config1 = r'--oem 1 --psm 3 -l eng+heb'
        print({
            "process_id": multiprocessing.current_process().pid,
            "process_name": multiprocessing.current_process().name,
            "processing_active_count": len_ac
        })
        if idx == -1:
            print_log_pro('------------- step -1 textract -----------------', logger_pro, fileIdKey)
            try:
                response = boto3.client('textract', region_name='eu-west-1').detect_document_text(
                    Document={'Bytes': byte_im})
                # print('response textract', response)
                return response
            except ClientError as client_error:
                print_log(fileId, "Couldn't detect aws text.", client_error)

        elif idx == 0:
            print_log_pro('------------- step 1 - 120, 255, cv2.THRESH_BINARY --psm 6: -----------------',
                          logger_pro, fileIdKey)
            image = download_path

            # morph = image
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            # morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
            # morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            # gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)
            # image_channels = np.split(np.asarray(gradient_image), 3, axis=2)
            # channel_height, channel_width, _ = image_channels[0].shape
            # for i in range(0, 3):
            #     _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255,
            #                                          cv2.THRESH_OTSU | cv2.THRESH_BINARY)
            #     image_channels[i] = np.reshape(image_channels[i],
            #                                    newshape=(channel_height, channel_width, 1))
            # opencvImage_to_crop = np.concatenate(
            #     (image_channels[0], image_channels[1], image_channels[2]),
            #     axis=2)

            # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
            image = image[:, :, :3]  # image_without_alpha
            ret, thresh1 = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
            # cv2.imwrite('thresh1.jpg', thresh1)
            image_to_data = pytesseract.image_to_data(thresh1,
                                                      lang='eng+heb',
                                                      output_type=Output.DICT,
                                                      config='--psm 6')
            return image_to_data
        elif idx == 1:
            print_log_pro('------------- step 2 - adaptive_threshold - image_to_data:-----------------', logger_pro,
                          fileIdKey)
            adaptive_threshold = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY, 85,
                                                       11)
            # cv2.imwrite('adaptive_threshold.jpg', adaptive_threshold)
            image_to_data = pytesseract.image_to_data(adaptive_threshold,
                                                      lang='eng+heb',
                                                      output_type=Output.DICT,
                                                      config=custom_oem_psm_config1)
            return image_to_data
        elif idx == 2:
            print_log_pro('------------- step 3 - threshold - Crop - findContours  image_to_data:-----------------',
                          logger_pro, fileIdKey)
            # cv2.imwrite('grayImage2.jpg', grayImage)
            image_to_data = pytesseract.image_to_data(grayImage,
                                                      lang='eng+heb',
                                                      output_type=Output.DICT,
                                                      config='--psm 11 --oem 3')
            return image_to_data
        elif idx == 3:

            print_log_pro('------------- step 4 - threshold - THRESH_TOZERO - image_to_data:-----------------',
                          logger_pro, fileIdKey)
            (thresh, img_rgb) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_TOZERO)
            # cv2.imwrite('threshold3.jpg', img_rgb)
            image_to_data = pytesseract.image_to_data(img_rgb,
                                                      lang='eng+heb',
                                                      output_type=Output.DICT,
                                                      config=custom_oem_psm_config1)
            return image_to_data
        elif idx == 4:
            print_log_pro('------------- step 5 - opening - image_to_data:-----------------', logger_pro, fileIdKey)
            img_rgb = opening(download_path)
            # cv2.imwrite('opening.jpg', img_rgb)
            image_to_data = pytesseract.image_to_data(img_rgb,
                                                      lang='eng+heb',
                                                      output_type=Output.DICT,
                                                      config=custom_oem_psm_config1)
            return image_to_data
        elif idx == 5:
            print_log_pro('------------- step 6 - --dpi 3000 image_to_data:-----------------', logger_pro, fileIdKey)
            (thresh, img_rgb) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_TOZERO)
            # cv2.imwrite('threshold4.jpg', img_rgb)
            image_to_data = pytesseract.image_to_data(img_rgb,
                                                      lang='eng+heb',
                                                      output_type=Output.DICT,
                                                      config=r'--dpi 3000 --oem 1 --psm 3 -l eng+heb')
            return image_to_data
        elif idx == 6:
            print_log_pro(
                '------------- step 7 - threshold cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV image_to_data:-----------------',
                logger_pro, fileIdKey)
            (thresh, img_rgb) = cv2.threshold(grayImage, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            # cv2.imwrite('threshold5.jpg', img_rgb)
            image_to_data = pytesseract.image_to_data(img_rgb,
                                                      lang='eng+heb',
                                                      output_type=Output.DICT,
                                                      config=custom_oem_psm_config1)
            return image_to_data
        elif idx == 7:
            print_log_pro('------------- step 8 - threshold THRESH_BINARY image_to_data:-----------------', logger_pro,
                          fileIdKey)
            (thresh, img_rgb) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
            # cv2.imwrite('img_rgb.jpg', img_rgb)
            image_to_data = pytesseract.image_to_data(img_rgb,
                                                      lang='eng+heb',
                                                      output_type=Output.DICT,
                                                      config=custom_oem_psm_config1)
            return image_to_data
        elif idx == 8:
            print_log_pro('------------- step 9 - digits blackAndWhiteImage - image_to_data:-----------------',
                          logger_pro, fileIdKey)
            (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
            # cv2.imwrite('blackAndWhiteImage.jpg', blackAndWhiteImage)
            image_to_data = pytesseract.image_to_data(blackAndWhiteImage,
                                                      output_type=Output.DICT,
                                                      config='digits')
            return image_to_data
        elif idx == 9:
            print_log_pro('------------- step 10 - canny rgba - image_to_data:-----------------', logger_pro, fileIdKey)
            morph = download_path
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)
            image_channels = np.split(np.asarray(gradient_image), 3, axis=2)
            channel_height, channel_width, _ = image_channels[0].shape
            for i in range(0, 3):
                _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255,
                                                     cv2.THRESH_OTSU | cv2.THRESH_BINARY)
                image_channels[i] = np.reshape(image_channels[i],
                                               newshape=(channel_height, channel_width, 1))
            based_color = np.concatenate((image_channels[0], image_channels[1], image_channels[2]),
                                         axis=2)
            image_to_data = pytesseract.image_to_data(based_color,
                                                      lang='eng+heb',
                                                      output_type=Output.DICT,
                                                      config=custom_oem_psm_config1)
            return image_to_data
        # elif idx == 9:
        #     print_log_pro('------------- step 10 - canny rgba - image_to_data:-----------------', logger_pro, fileIdKey)
        #     img_canny = cv2.Canny(download_path, 50, 100, apertureSize=3)
        #     image_to_data = pytesseract.image_to_data(img_canny,
        #                                               lang='eng+heb',
        #                                               output_type=Output.DICT,
        #                                               config=custom_oem_psm_config1)
        #     return image_to_data
        else:
            return None
    except Exception as error_pytesseract_image_to_data:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_details = {
            'filename': exc_traceback.tb_frame.f_code.co_filename,
            'lineno': exc_traceback.tb_lineno,
            'function_name': exc_traceback.tb_frame.f_code.co_name,
            'type': exc_type.__name__,
            'message': str(exc_value)
        }
        del (exc_type, exc_value, exc_traceback)
        print_log_pro('traceback_details: ' + str(traceback_details), logger_pro, fileIdKey)
        print_log_pro('errorOcr: ' + str(error_pytesseract_image_to_data), logger_pro, fileIdKey)


def get_currency_type(current_word, prev_word, next_word):
    switcher = {
        "₪": '1',
        "שח": '1',
        "ש״ח": '1',
        'ש"ח': '1',
        "USD": '2',
        "usd": '2',
        "$": '2',
        "דולר": '2',
        "EUR": '11',
        "Euro": '11',
        "אירו": '11',
        "EU": '11',
        "יורו": '11'
    }
    switcher_type = None
    for currency_type in switcher:
        if currency_type in current_word:
            switcher_type = switcher[currency_type]
    if prev_word is not None and switcher_type is None:
        for currency_type in switcher:
            if currency_type in prev_word:
                switcher_type = switcher[currency_type]
    if next_word is not None and switcher_type is None:
        for currency_type in switcher:
            if currency_type in next_word:
                switcher_type = switcher[currency_type]
    if switcher_type is None:
        switcher_type = "1"
    return switcher_type


def get_all_dates(image_to_data, fileId):
    # print('get_all_dates')
    arr_build = []
    try:
        s = ' '.join(image_to_data['text'])
        data_full = ''.join(image_to_data['text'])
        res_arr = []
        for x in allowedDateFormats:
            # try:
            #     # print('en: ', arrow.get(normalize_whitespace=True, locale='en').format(x, locale='en'))
            #     print('en: ', arrow.get(normalize_whitespace=True, locale='he').format(x, locale='he'))
            # except Exception as process_ocr_data_err11:
            #     print(process_ocr_data_err11)
            try:
                format_word_found = arrow.get(s, x, normalize_whitespace=True, locale='he').format(x, locale='he')
                # print('format_word_found HEB ', format_word_found)
                res = [(ele.start(), ele.end()) for ele in
                       re.finditer(re.sub(r"\s+", "", format_word_found), data_full)]
                # print("word after cut:  ---- ", data_full[res[0][0]:res[0][1]])
                res_arr = list(set(res + res_arr))
                # print("Word Ranges are : " + str(res))
            # .format(x, locale='en')
            except Exception as process_ocr_data_err:
                pass
                # print(process_ocr_data_err)
            try:
                format_word_found = arrow.get(s, x, normalize_whitespace=True, locale='en').format(x, locale='en')
                # print('format_word_found EN ', format_word_found)
                res = [(ele.start(), ele.end()) for ele in
                       re.finditer(re.sub(r"\s+", "", format_word_found), data_full)]
                # print("word after cut:  ---- ", data_full[res[0][0]:res[0][1]])
                res_arr = list(set(res + res_arr))
                # print("Word Ranges are : " + str(res))
            except Exception as process_ocr_data_err2:
                pass
                # print(process_ocr_data_err2)
        res_arr.sort()
        # print('res_arr---', res_arr)

        index_match_words = 0
        len_txt = 0
        widthReal = 1
        heightReal = 1
        for i in range(0, len(image_to_data["text"])):
            # the current result
            if len(res_arr) > (index_match_words):
                conf = int(image_to_data["conf"][i])
                x = widthReal * image_to_data["left"][i]
                y = heightReal * image_to_data["top"][i]
                w = widthReal * image_to_data["width"][i]
                h = heightReal * image_to_data["height"][i]
                text = image_to_data["text"][i]
                # print(len_txt == res_arr[index_match_words][0])
                # len_txt += len(text)
                # print(text, index_match_words)
                if len_txt == res_arr[index_match_words][0]:
                    # print('שווה לאינדקס של תחילת המשפט', text)
                    arr_build.append({
                        "vertices": [{
                            "x": x,
                            "y": y
                        }, {
                            "x": x + w,
                            "y": y
                        }, {
                            "x": x + w,
                            "y": y + h
                        }, {
                            "x": x,
                            "y": y + h
                        }],
                        "text": text,
                        "conf": conf
                    })
                    len_txt += len(text)
                    if len_txt == res_arr[index_match_words][1]:
                        # print('וגם !!! שווה לאינדקס של סוף המשפט')
                        index_match_words += 1
                else:
                    len_txt += len(text)

                    if len_txt == res_arr[index_match_words][1]:
                        # print('שווה לאינדקס של סוף המשפט', text)

                        if len(arr_build) > 0:
                            arr_build[len(arr_build) - 1]['text'] += ' ' + text
                            arr_build[len(arr_build) - 1]['vertices'][1]['x'] = x + w
                            arr_build[len(arr_build) - 1]['vertices'][2]['x'] = x + w
                        else:
                            arr_build.append({
                                "vertices": [{
                                    "x": x,
                                    "y": y
                                }, {
                                    "x": x + w,
                                    "y": y
                                }, {
                                    "x": x + w,
                                    "y": y + h
                                }, {
                                    "x": x,
                                    "y": y + h
                                }],
                                "text": text,
                                "conf": conf
                            })

                        index_match_words += 1
                    else:
                        if res_arr[index_match_words][1] > len_txt > res_arr[index_match_words][0]:
                            # print('שווה לאינדקס של אמצע המשפט', text)

                            if len(arr_build) > 0:
                                arr_build[len(arr_build) - 1]['text'] += ' ' + text
                                arr_build[len(arr_build) - 1]['vertices'][1]['x'] = x + w
                                arr_build[len(arr_build) - 1]['vertices'][2]['x'] = x + w
                            else:
                                arr_build.append({
                                    "vertices": [{
                                        "x": x,
                                        "y": y
                                    }, {
                                        "x": x + w,
                                        "y": y
                                    }, {
                                        "x": x + w,
                                        "y": y + h
                                    }, {
                                        "x": x,
                                        "y": y + h
                                    }],
                                    "text": text,
                                    "conf": conf
                                })

        if len(arr_build) > 0:
            arr_build = list(filter(lambda x: int(x["conf"]) != -1 and x["text"] != '', arr_build))

    except Exception as get_all_dates_err:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_details = {
            'filename': exc_traceback.tb_frame.f_code.co_filename,
            'lineno': exc_traceback.tb_lineno,
            'function_name': exc_traceback.tb_frame.f_code.co_name,
            'type': exc_type.__name__,
            'message': str(exc_value)
        }
        del (exc_type, exc_value, exc_traceback)
        print_log(fileId, 'traceback_details: ', str(traceback_details))
        print_log(fileId, 'get_all_dates_err: ', str(get_all_dates_err))

    # print(arr_build)
    return arr_build


def get_all_dates_aws(word):
    # print(word)
    is_match_to_date = False
    for x in allowedDateFormats:
        try:
            format_word_found = arrow.get(word, x, normalize_whitespace=True, locale='he').format(x, locale='he')
            if format_word_found:
                is_match_to_date = True
        except Exception as process_ocr_data_err:
            pass
            # print(process_ocr_data_err)
        try:
            format_word_found = arrow.get(word, x, normalize_whitespace=True, locale='en').format(x, locale='en')
            if format_word_found:
                is_match_to_date = True
        except Exception as process_ocr_data_err2:
            pass
            # print(process_ocr_data_err2)
    return is_match_to_date


def process_ocr_data(image_to_data, data, dates, hp_list, arr_numbers, widthReal, heightReal, words, partial_VAT, isAws,
                     height_resized=False, width_resized=False, fileId=''):
    return_obj_ocr = {
        "doc_number": None,
        "dataNum": [],
        "aws_textract": {
            "dataNum": [],
            "words": [],
            "data": [],
            "hp_list": [],
            "partial_VAT": [],
            "dates": [],
            "arr_numbers": []
        }
    }
    words_local = []

    try:
        if not isAws:
            total_len_words = len(image_to_data["text"])
            if total_len_words > 0:
                dates += get_all_dates(image_to_data, fileId)
            else:
                dates += []

            # print(image_to_data)
            index_number_doc = None
            index_words = 0
            is_number_docs_ele = False
            for i in range(0, len(image_to_data["text"])):
                conf = int(image_to_data["conf"][i])
                if conf != -1:
                    # the current result
                    x = widthReal * image_to_data["left"][i]
                    y = heightReal * image_to_data["top"][i]
                    w = widthReal * image_to_data["width"][i]
                    h = heightReal * image_to_data["height"][i]
                    text = image_to_data["text"][i]
                    # print(text)
                    if text.find('חשבונית') != -1:
                        is_number_docs_ele = True
                    if (is_number_docs_ele and text.find('מס') != -1 and index_number_doc is None) or (
                            text.find('Receipt') != -1 and index_number_doc is None):
                        index_number_doc = {
                            "index": index_words,
                            "y": y,
                            "x": x + w,
                            "found": None,
                            "isHeb": text.find('מס') != -1
                        }

                    # if text.find('מספר') != -1 or text.find("מס'") != -1:
                    #     index_number_doc = {
                    #         "index": index_words,
                    #         "y": y,
                    #         "x": x + w,
                    #         "found": None
                    #     }
                    if re.search("\d", text):
                        words_local.append({
                            "index": index_words,
                            "vertices": [{
                                "x": x,
                                "y": y
                            }, {
                                "x": x + w,
                                "y": y
                            }, {
                                "x": x + w,
                                "y": y + h
                            }, {
                                "x": x,
                                "y": y + h
                            }],
                            "text": text,
                            "conf": conf
                        })

                    words.append({
                        "vertices": [{
                            "x": x,
                            "y": y
                        }, {
                            "x": x + w,
                            "y": y
                        }, {
                            "x": x + w,
                            "y": y + h
                        }, {
                            "x": x,
                            "y": y + h
                        }],
                        "text": text,
                        "conf": conf
                    })
                    # "(?<!\S)(?=.)(0|([1-9](\d*|\d{0,2}(,\d{3})*)))(\.\d{0,2})(?!\S)"
                    search_number_with_comma = re.search(
                        r'(?<!\S)(?=.)(0|([1-9](\d*|\d{0,2}(,\d{3})*)))(\.\d{2})(?!\S)',
                        text)
                    if search_number_with_comma:
                        # print_log(fileId,'found number pattern: ', search_number_with_comma.group())
                        numeric_string = re.sub("[^0-9.]", "", search_number_with_comma.group())
                        # print_log(fileId,'numeric_string: ', numeric_string)
                        prev_word = None
                        if i > 0:
                            prev_word = image_to_data["text"][i - 1]
                        next_word = None
                        if i + 1 != total_len_words:
                            next_word = image_to_data["text"][i + 1]

                        obj = {
                            "vertices": [{
                                "x": x,
                                "y": y
                            }, {
                                "x": x + w,
                                "y": y
                            }, {
                                "x": x + w,
                                "y": y + h
                            }, {
                                "x": x,
                                "y": y + h
                            }],
                            "text": float(numeric_string),
                            "conf": conf,
                            "currency_id": get_currency_type(image_to_data["text"][i], prev_word, next_word)
                        }
                        data.append(obj)
                        return_obj_ocr['dataNum'].append(float(numeric_string))
                    else:
                        check_decimal_pattern = re.findall('\\d*\\.\\d+', text)
                        check_decimal_date_pattern = re.findall("\\d*\\.\\d+\\.", text)
                        check_decimal_date_pattern_prev = re.findall("\.\d*\\.\\d+", text)
                        if len(check_decimal_pattern) > 0 and len(check_decimal_date_pattern) == 0 and len(
                                check_decimal_date_pattern_prev) == 0:
                            text = check_decimal_pattern[0]
                            # print_log(fileId,'check_decimal_pattern:', text)
                        else:
                            check_comma_pattern = re.findall("\\d*\\,\\d+", text)
                            if len(check_comma_pattern) > 0:
                                # print_log(fileId,'check_comma_pattern:', check_comma_pattern[0])
                                text = re.sub(",", ".", check_comma_pattern[0])
                                # print_log(fileId,'replace_comma:', text)

                        z = re.match("(?<!\S)(?=.)(0|([1-9](\d*|\d{0,2}(,\d{3})*)))(\.\d{2})(?!\S)", text)
                        if z:
                            numeric_string = re.sub("[^0-9.]", "", text)
                            # print_log(fileId,'numeric_string: ', numeric_string)
                            prev_word = None
                            if i > 0:
                                prev_word = image_to_data["text"][i - 1]
                            next_word = None
                            if i + 1 != total_len_words:
                                next_word = image_to_data["text"][i + 1]

                            obj = {
                                "vertices": [{
                                    "x": x,
                                    "y": y
                                }, {
                                    "x": x + w,
                                    "y": y
                                }, {
                                    "x": x + w,
                                    "y": y + h
                                }, {
                                    "x": x,
                                    "y": y + h
                                }],
                                "text": float(numeric_string),
                                "conf": conf,
                                "currency_id": get_currency_type(image_to_data["text"][i], prev_word, next_word)
                            }
                            data.append(obj)
                            return_obj_ocr['dataNum'].append(float(numeric_string))
                    full_match_hp = re.search(r'(\d{9})', text)
                    if full_match_hp:
                        hp_string = re.sub("[^0-9.]", "", full_match_hp.group())
                        is_hp_valid = check_valid_hp(hp_string, fileId)
                        if is_hp_valid:
                            # print_log(fileId,'found hp pattern: ', hp_string)
                            obj = {
                                "vertices": [{
                                    "x": x,
                                    "y": y
                                }, {
                                    "x": x + w,
                                    "y": y
                                }, {
                                    "x": x + w,
                                    "y": y + h
                                }, {
                                    "x": x,
                                    "y": y + h
                                }],
                                "text": hp_string,
                                "conf": conf
                            }
                            hp_list.append(obj)
                    index_words += 1

            if index_number_doc is not None:
                # print('index_number_doc: ', index_number_doc)
                vertices = []
                for index, item_data in enumerate(words_local):
                    if (index_number_doc['isHeb'] and item_data['vertices'][1]['x'] < index_number_doc['x']) or (
                            not index_number_doc['isHeb'] and item_data['vertices'][1]['x'] > index_number_doc['x']):
                        vertices.append({
                            "text": item_data["text"],
                            "index": index,
                            "distance": abs(index_number_doc["index"] - item_data["index"])
                        })

                if len(vertices) > 0:
                    # print('vertices---', vertices)
                    vertices_min = min(vertices, key=lambda x: x['distance'])
                    # print(vertices_min)
                    item_close = words_local[vertices_min['index']]
                    del item_close["index"]
                    return_obj_ocr["doc_number"] = item_close
                    print('item_close---', item_close)

            len_data = len(data)
            if len_data > 0:
                unique_data = {str(each['vertices']) + str(each['text']): each for each in data}.values()
                data = list(unique_data)
                data = sorted(data, key=lambda dct: dct['text'], reverse=True)
                numbers_maam = check_sums(data, fileId)
                partial_VAT = check_basics_sums(data, fileId)
                arr_numbers = list(numbers_maam)
                len_arr_numbers = len(arr_numbers)
                if len_arr_numbers > 0:
                    # print_log(fileId,'len_arr_numbers > 0')
                    if arr_numbers[0]['text'] == arr_numbers[1]['text']:
                        # print_log(fileId,"arr_numbers[0]['text'] == arr_numbers[1]['text']", str(arr_numbers))
                        arr_numbers = []
        else:
            if image_to_data and image_to_data['Blocks'] and len(image_to_data['Blocks']) > 0:
                # print(image_to_data)
                i = 0
                total_len_words = len(image_to_data["Blocks"])
                for block in image_to_data['Blocks']:
                    blockType = block['BlockType']
                    if blockType == 'WORD':
                        conf = int(block["Confidence"])
                        if conf > 0:
                            text = block["Text"]
                            polygon = block["Geometry"]["Polygon"]
                            for polygonObj in polygon:
                                polygonObj["x"] = (width_resized * polygonObj["X"])
                                polygonObj["y"] = (height_resized * polygonObj["Y"])
                                del polygonObj["X"]
                                del polygonObj["Y"]
                            return_obj_ocr['aws_textract']['words'].append({
                                "vertices": polygon,
                                "text": text,
                                "conf": conf
                            })
                            is_date = get_all_dates_aws(text)
                            # print('is_date: ', is_date)
                            if is_date:
                                return_obj_ocr['aws_textract']['dates'].append({
                                    "vertices": polygon,
                                    "text": text,
                                    "conf": conf
                                })

                            search_number_with_comma = re.search(
                                r'(?<!\S)(?=.)(0|([1-9](\d*|\d{0,2}(,\d{3})*)))(\.\d{2})(?!\S)',
                                text)
                            if search_number_with_comma:
                                # print_log(fileId,'found number pattern: ', search_number_with_comma.group())
                                numeric_string = re.sub("[^0-9.]", "", search_number_with_comma.group())
                                # print_log(fileId,'numeric_string: ', numeric_string)
                                prev_word = None
                                if i > 0:
                                    prev_word = image_to_data["Blocks"][i - 1]["Text"]
                                next_word = None
                                if i + 1 != total_len_words:
                                    next_word = image_to_data["Blocks"][i + 1]["Text"]

                                obj = {
                                    "vertices": polygon,
                                    "text": float(numeric_string),
                                    "conf": conf,
                                    "currency_id": get_currency_type(text, prev_word, next_word)
                                }
                                return_obj_ocr['aws_textract']['data'].append(obj)
                                return_obj_ocr['aws_textract']['dataNum'].append(float(numeric_string))
                            else:
                                check_decimal_pattern = re.findall('\\d*\\.\\d+', text)
                                check_decimal_date_pattern = re.findall("\\d*\\.\\d+\\.", text)
                                check_decimal_date_pattern_prev = re.findall("\.\d*\\.\\d+", text)
                                if len(check_decimal_pattern) > 0 and len(check_decimal_date_pattern) == 0 and len(
                                        check_decimal_date_pattern_prev) == 0:
                                    text = check_decimal_pattern[0]
                                    # print_log(fileId,'check_decimal_pattern:', text)
                                else:
                                    check_comma_pattern = re.findall("\\d*\\,\\d+", text)
                                    if len(check_comma_pattern) > 0:
                                        # print_log(fileId,'check_comma_pattern:', check_comma_pattern[0])
                                        text = re.sub(",", ".", check_comma_pattern[0])
                                        # print_log(fileId,'replace_comma:', text)

                                z = re.match("(?<!\S)(?=.)(0|([1-9](\d*|\d{0,2}(,\d{3})*)))(\.\d{2})(?!\S)", text)
                                if z:
                                    numeric_string = re.sub("[^0-9.]", "", text)
                                    # print_log(fileId,'numeric_string: ', numeric_string)
                                    prev_word = None
                                    if i > 0:
                                        prev_word = image_to_data["Blocks"][i - 1]["Text"]
                                    next_word = None
                                    if i + 1 != total_len_words:
                                        next_word = image_to_data["Blocks"][i + 1]["Text"]

                                    obj = {
                                        "vertices": polygon,
                                        "text": float(numeric_string),
                                        "conf": conf,
                                        "currency_id": get_currency_type(text, prev_word, next_word)
                                    }
                                    return_obj_ocr['aws_textract']['data'].append(obj)
                                    return_obj_ocr['aws_textract']['dataNum'].append(float(numeric_string))
                            full_match_hp = re.search(r'(\d{9})', text)
                            if full_match_hp:
                                hp_string = re.sub("[^0-9.]", "", full_match_hp.group())
                                is_hp_valid = check_valid_hp(hp_string, fileId)
                                if is_hp_valid:
                                    # print_log(fileId,'found hp pattern: ', hp_string)
                                    obj = {
                                        "vertices": polygon,
                                        "text": hp_string,
                                        "conf": conf
                                    }
                                    return_obj_ocr['aws_textract']['hp_list'].append(obj)
                    i = i + 1
                len_data_aws = len(return_obj_ocr['aws_textract']['data'])
                if len_data_aws > 0:
                    unique_data_aws = {str(each['vertices']) + str(each['text']): each for each in
                                       return_obj_ocr['aws_textract']['data']}.values()
                    data_aws = list(unique_data_aws)
                    data_aws = sorted(data_aws, key=lambda dct: dct['text'], reverse=True)
                    numbers_maam_aws = check_sums(data_aws, fileId)
                    partial_VAT_aws = check_basics_sums(data_aws, fileId)
                    arr_numbers_aws = list(numbers_maam_aws)
                    len_arr_numbers_aws = len(arr_numbers_aws)
                    if len_arr_numbers_aws > 0:
                        # print_log(fileId,'len_arr_numbers > 0')
                        if arr_numbers_aws[0]['text'] == arr_numbers_aws[1]['text']:
                            # print_log(fileId,"arr_numbers[0]['text'] == arr_numbers[1]['text']", str(arr_numbers))
                            arr_numbers_aws = []
                    return_obj_ocr['aws_textract']['partial_VAT'] = partial_VAT_aws
                    return_obj_ocr['aws_textract']['arr_numbers'] = arr_numbers_aws
                    return_obj_ocr['aws_textract']['data'] = data_aws
                    del partial_VAT_aws
                    del arr_numbers_aws
                    del data_aws
            print_log(fileId, "Detected aws %s blocks.", len(image_to_data['Blocks']))


    except Exception as process_ocr_data_err:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_details = {
            'filename': exc_traceback.tb_frame.f_code.co_filename,
            'lineno': exc_traceback.tb_lineno,
            'function_name': exc_traceback.tb_frame.f_code.co_name,
            'type': exc_type.__name__,
            'message': str(exc_value)
        }
        del (exc_type, exc_value, exc_traceback)
        print_log(fileId, 'traceback_details: ', str(traceback_details))
        print_log(fileId, 'process_ocr_data_err: ', str(process_ocr_data_err))
    return_obj_ocr['arr_numbers'] = arr_numbers
    return_obj_ocr['data'] = data
    return_obj_ocr['dates'] = dates
    return_obj_ocr['partial_VAT'] = partial_VAT
    return_obj_ocr['words'] = words
    return_obj_ocr['hp_list'] = hp_list
    del arr_numbers
    del dates
    del data
    del partial_VAT
    del words
    del hp_list
    gc.collect()
    return return_obj_ocr


def get_ocr(path_of_img, bucketKey, width_original, height_original, grayImg, download_path, fileId):
    print_log(fileId, 'Start get_ocr function')
    # print_log(fileId,'get_tesseract_version: ', str(pytesseract.get_tesseract_version()))
    grayImage = ''
    img_rgb = ''
    grayImage_base = ''
    new_list_hp = []
    gray = ''
    adaptive_threshold = ''
    img_rgba = ''
    img_canny = ''
    # img_rgb = Image.open(image_rotate)
    print_log(fileId, 'Read img from path')
    custom_oem_psm_config1 = r'--oem 1 --psm 3 -l eng+heb'

    output_obj = {
        'fileId': bucketKey,
        'total_steps': 0,
        'doc_number': None
    }
    try:
        data = []
        dates = []
        partial_VAT = []
        words = []
        img_rgba = ''
        img_rgb = ''
        hp_list = []
        arr_numbers = []
        image_to_data = ''
        grayImage_base = grayImg
        height = grayImage_base.shape[0]
        width = grayImage_base.shape[1]
        # grayImage_base = cv2.imread(download_path, cv2.COLOR_BGR2GRAY)
        # Noise Removal
        # grayImage_base = cv2.fastNlMeansDenoisingColored(grayImage_base, None, 10, 10, 7, 15)
        grayImage = grayImage_base.copy()
        heightReal = height_original / height
        widthReal = width_original / width
        # print_log(fileId,'Image Height       : ', height)
        # print_log(fileId,'Image Width        : ', width)
        # print_log(fileId,'heightReal       : ', height_original)
        # print_log(fileId,'widthReal        : ', width_original)
        print_log(fileId, "------Start pytesseract")

        # try:
        #     # scale_percent = 90
        #     # width_n = int(width_original * scale_percent / 100)
        #     # height_n = int(height_original * scale_percent / 100)
        #     # dim = (width_n, height_n)
        #     # resized = cv2.resize(path_of_img, dim, interpolation=cv2.INTER_AREA)
        #     # height_resized = resized.shape[0]
        #     # width_resized = resized.shape[1]
        #     gray = cv2.cvtColor(path_of_img, cv2.IMREAD_GRAYSCALE)
        #     is_success, im_buf_arr = cv2.imencode(".jpg", gray)
        #     byte_im = im_buf_arr.tobytes()
        #
        #     # response_expense = textract.analyze_expense(Document={'Bytes': byte_im})
        #     # print('response_expense:', response_expense)
        #
        # except ClientError as client_error:
        #     print_log(fileId,"Couldn't detect aws text.", client_error)

        try:
            gray = cv2.cvtColor(path_of_img, cv2.IMREAD_GRAYSCALE)
            is_success, im_buf_arr = cv2.imencode(".jpg", gray)
            byte_im = im_buf_arr.tobytes()

            start_multiprocessing = time.time()
            print_log(fileId, "------Start ProcessPool all")
            from ray.exceptions import TaskCancelledError, GetTimeoutError
            number_allow_threads = math.ceil(num_of_worker_threads / 10)

            try:
                range_to_run = range(-1, 10)
                if number_allow_threads == 1:
                    range_to_run = range(-1, 4)
                results = ray.get(
                    [pytesseract_image_to_data.remote(i, grayImage, byte_im, len(multiprocessing.active_children()),
                                                      path_of_img, logger, fileId) for i in range_to_run], timeout=40)
                indexRemote = 0
                for result in results:
                    indexRemote += 1
                    image_to_data = result

                    if image_to_data and ('Blocks' in image_to_data) and len(image_to_data['Blocks']) > 0 and not (
                            'text' in image_to_data):
                        response_data_obj = process_ocr_data(image_to_data, data, dates, hp_list, arr_numbers,
                                                             widthReal,
                                                             heightReal, words, partial_VAT, True, height_original,
                                                             width_original,
                                                             fileId=fileId)
                        print('dataNum len of aws', response_data_obj['aws_textract']['dataNum'])
                        arr_numbers_aws = response_data_obj['aws_textract']['arr_numbers']
                        partial_VAT_aws = response_data_obj['aws_textract']['data']
                        data_aws = response_data_obj['aws_textract']['data']
                        hp_list_aws = response_data_obj['aws_textract']['hp_list']
                        dates_aws = response_data_obj['aws_textract']['dates']

                        len_arr_numbers = len(arr_numbers_aws)
                        if len_arr_numbers > 0:
                            # print_log(fileId,'last len_arr_numbers > 0')
                            if arr_numbers_aws[0]['text'] == arr_numbers_aws[1]['text']:
                                # print_log(fileId,"arr_numbers[0]['text'] == arr_numbers[1]['text']", arr_numbers)
                                arr_numbers_aws = []

                        partial_VAT_len = len(partial_VAT_aws)
                        if partial_VAT_len == 4 and (
                                (len_arr_numbers == 0) or (
                                len_arr_numbers > 0 and arr_numbers_aws[0]['text'] < partial_VAT_aws[0]['text'])):
                            arr_numbers_aws = partial_VAT_aws
                            # print('partial_VAT --- 4: ', arr_numbers)
                        elif partial_VAT_len == 2 and len_arr_numbers == 0:
                            arr_numbers_aws = partial_VAT_aws
                            # print('partial_VAT --- 2: ', arr_numbers)

                        # print('---data_aws', data_aws)

                        output_obj['aws_textract'] = {}
                        output_obj['aws_textract']['sums'] = data_aws
                        dict_group_by = {}
                        if len(arr_numbers_aws) == 0:
                            for obj in data_aws:
                                if str(obj["text"]) not in dict_group_by:
                                    dict_group_by[str(obj["text"])] = 1
                                else:
                                    dict_group_by[str(obj["text"])] = dict_group_by[str(obj["text"])] + 1
                            if len(dict_group_by) > 0:
                                dict_group_by = {k: v for k, v in
                                                 sorted(dict_group_by.items(), key=lambda item: item[1], reverse=True)}
                                if (list(dict_group_by.values())[0] > 1):
                                    text_bigger_than_one = list(dict_group_by.keys())[0]
                                    get_index_first_match = next(
                                        (i for i, d in enumerate(data_aws) if
                                         str(d['text']) == str(text_bigger_than_one)),
                                        None)
                                    # print_log(fileId,str(text_bigger_than_one) + ' bigger than 1', ' exist the first at index: ',
                                    #           get_index_first_match)
                                    arr_numbers_aws.append(data_aws[get_index_first_match])
                                    arr_numbers_aws.append(data_aws[get_index_first_match + 1])
                                    if (str(0.00) in dict_group_by) and dict_group_by[str(0.00)] > 0:
                                        get_index_first__zero_match = next(
                                            (i for i, d in enumerate(data_aws) if str(d['text']) == str(0.00)),
                                            None)
                                        if get_index_first__zero_match is None:
                                            get_index_first__zero_match = len(data_aws) - 1
                                        # print_log(fileId,'get_index_first__zero_match: ', str(get_index_first__zero_match))
                                        arr_numbers_aws.append(data_aws[get_index_first__zero_match])
                                    else:
                                        obj = {
                                            "vertices": [],
                                            "text": float(0.00),
                                            "conf": 0
                                        }
                                        arr_numbers_aws.append(obj)

                        if len(arr_numbers_aws) > 0:
                            for item_data in arr_numbers_aws:
                                item_data["text"] = format(float(item_data["text"]), ".2f")

                        output_obj['aws_textract']['data'] = arr_numbers_aws
                        output_obj['aws_textract']['dates'] = dates_aws
                        # unique_words = {str(each['vertices']) + str(each['text']): each for each in words}.values()
                        # output_obj['words'] = list(unique_words)
                        # print(list(unique_words))
                        print_log(fileId, 'arr_numbers_aws', arr_numbers_aws)
                        seen_titles_hp = set()
                        new_list_hp = []
                        for obj_hp in hp_list_aws:
                            if obj_hp["text"] not in seen_titles_hp:
                                new_list_hp.append(obj_hp)
                                seen_titles_hp.add(obj_hp["text"])
                        output_obj['aws_textract']['hp_list'] = new_list_hp

                        del response_data_obj
                        gc.collect()
                    else:
                        if image_to_data is not None:
                            response_data_obj = process_ocr_data(image_to_data, data, dates, hp_list, arr_numbers,
                                                                 widthReal,
                                                                 heightReal, words, partial_VAT, False, fileId=fileId)

                            print('dataNum len of ' + str(indexRemote), response_data_obj['dataNum'])
                            arr_numbers = response_data_obj['arr_numbers']
                            if output_obj['doc_number'] is None:
                                output_obj['doc_number'] = response_data_obj['doc_number']
                            else:
                                if response_data_obj['doc_number'] is not None:
                                    if len(str(output_obj['doc_number']['text'])) < len(
                                            str(response_data_obj['doc_number']['text'])):
                                        output_obj['doc_number'] = response_data_obj['doc_number']

                            data = response_data_obj['data']
                            partial_VAT = response_data_obj['partial_VAT']
                            words = response_data_obj['words']
                            hp_list = response_data_obj['hp_list']
                            dates = response_data_obj['dates']
                            output_obj['total_steps'] = output_obj['total_steps'] + 1
                            del image_to_data
                            del response_data_obj
                            gc.collect()

                if number_allow_threads == 1:
                    len_arr_numbers = len(arr_numbers)
                    if len_arr_numbers == 0 or (arr_numbers[0]["text"] < data[0]["text"]):
                        print_log(fileId, "------Start ProcessPool 4-8")
                        results = ray.get([pytesseract_image_to_data.remote(i, grayImage,
                                                                            len(multiprocessing.active_children()),
                                                                            path_of_img,
                                                                            logger, fileId) for i in range(4, 8)],
                                          timeout=30)
                        indexRemote = 4
                        for result in results:
                            indexRemote += 1
                            image_to_data = result
                            if image_to_data is not None:
                                response_data_obj = process_ocr_data(image_to_data, data, dates, hp_list, arr_numbers,
                                                                     widthReal,
                                                                     heightReal, words, partial_VAT, False,
                                                                     fileId=fileId)

                                print('dataNum len of ' + str(indexRemote), response_data_obj['dataNum'])
                                arr_numbers = response_data_obj['arr_numbers']
                                if output_obj['doc_number'] is None:
                                    output_obj['doc_number'] = response_data_obj['doc_number']
                                else:
                                    if response_data_obj['doc_number'] is not None:
                                        if len(str(output_obj['doc_number']['text'])) < len(
                                                str(response_data_obj['doc_number']['text'])):
                                            output_obj['doc_number'] = response_data_obj['doc_number']

                                data = response_data_obj['data']
                                partial_VAT = response_data_obj['partial_VAT']
                                words = response_data_obj['words']
                                hp_list = response_data_obj['hp_list']
                                dates = response_data_obj['dates']
                                output_obj['total_steps'] = output_obj['total_steps'] + 1
                                del image_to_data
                                del response_data_obj
                                gc.collect()

                    len_arr_numbers = len(arr_numbers)
                    if len_arr_numbers == 0 or (arr_numbers[0]["text"] < data[0]["text"]):
                        print_log(fileId, "------Start ProcessPool 8-11")
                        results = ray.get([pytesseract_image_to_data.remote(i, grayImage,
                                                                            len(multiprocessing.active_children()),
                                                                            path_of_img,
                                                                            logger, fileId) for i in range(8, 10)],
                                          timeout=30)
                        indexRemote = 8
                        for result in results:
                            indexRemote += 1
                            image_to_data = result
                            if image_to_data is not None:
                                response_data_obj = process_ocr_data(image_to_data, data, dates, hp_list, arr_numbers,
                                                                     widthReal,
                                                                     heightReal, words, partial_VAT, False,
                                                                     fileId=fileId)

                                print('dataNum len of ' + str(indexRemote), response_data_obj['dataNum'])
                                arr_numbers = response_data_obj['arr_numbers']
                                if output_obj['doc_number'] is None:
                                    output_obj['doc_number'] = response_data_obj['doc_number']
                                else:
                                    if response_data_obj['doc_number'] is not None:
                                        if len(str(output_obj['doc_number']['text'])) < len(
                                                str(response_data_obj['doc_number']['text'])):
                                            output_obj['doc_number'] = response_data_obj['doc_number']

                                data = response_data_obj['data']
                                partial_VAT = response_data_obj['partial_VAT']
                                words = response_data_obj['words']
                                hp_list = response_data_obj['hp_list']
                                dates = response_data_obj['dates']
                                output_obj['total_steps'] = output_obj['total_steps'] + 1
                                del image_to_data
                                del response_data_obj
                                gc.collect()
                                len_arr_numbers = len(arr_numbers)


            except GetTimeoutError:
                print("The cleaning process takes a long time (30 sec), the cleaning process is canceled")
                print("Starts running for ocr processing without cleaning")
                custom_oem_psm_config_global = r'--oem 1 --psm 3'
                image_to_data = pytesseract.image_to_data(grayImage,
                                                          lang='eng+heb',
                                                          output_type=Output.DICT,
                                                          config=custom_oem_psm_config_global)
                if image_to_data is not None:
                    response_data_obj = process_ocr_data(image_to_data, data, dates, hp_list, arr_numbers, widthReal,
                                                         heightReal, words, partial_VAT, False, fileId=fileId)

                    print('dataNum len of ' + str(0), response_data_obj['dataNum'])
                    arr_numbers = response_data_obj['arr_numbers']
                    if output_obj['doc_number'] is None:
                        output_obj['doc_number'] = response_data_obj['doc_number']
                    else:
                        if response_data_obj['doc_number'] is not None:
                            if len(str(output_obj['doc_number']['text'])) < len(
                                    str(response_data_obj['doc_number']['text'])):
                                output_obj['doc_number'] = response_data_obj['doc_number']
                    data = response_data_obj['data']
                    dates = response_data_obj['dates']
                    partial_VAT = response_data_obj['partial_VAT']
                    words = response_data_obj['words']
                    hp_list = response_data_obj['hp_list']
                    output_obj['total_steps'] = output_obj['total_steps'] + 1
                    del image_to_data
                    del response_data_obj
                    gc.collect()
                    len_arr_numbers = len(arr_numbers)

            request_time_multiprocessing = time.time() - start_multiprocessing
            print_log(fileId, ("Step multiprocessing took {0:.0f}ms".format(request_time_multiprocessing)))
        except Exception as error_ocr_proc:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback_details = {
                'filename': exc_traceback.tb_frame.f_code.co_filename,
                'lineno': exc_traceback.tb_lineno,
                'function_name': exc_traceback.tb_frame.f_code.co_name,
                'type': exc_type.__name__,
                'message': str(exc_value)
            }
            del (exc_type, exc_value, exc_traceback)
            print_log(fileId, 'traceback_details: ', str(traceback_details))
            print_log(fileId, 'error_ocr_proc: ', str(error_ocr_proc))

        len_arr_numbers = len(arr_numbers)
        if len_arr_numbers > 0:
            # print_log(fileId,'last len_arr_numbers > 0')
            if arr_numbers[0]['text'] == arr_numbers[1]['text']:
                # print_log(fileId,"arr_numbers[0]['text'] == arr_numbers[1]['text']", arr_numbers)
                arr_numbers = []

        partial_VAT_len = len(partial_VAT)
        if partial_VAT_len == 4 and (
                (len_arr_numbers == 0) or (len_arr_numbers > 0 and arr_numbers[0]['text'] < partial_VAT[0]['text'])):
            arr_numbers = partial_VAT
            # print('partial_VAT --- 4: ', arr_numbers)
        elif partial_VAT_len == 2 and len_arr_numbers == 0:
            arr_numbers = partial_VAT
            # print('partial_VAT --- 2: ', arr_numbers)

        # print_log(fileId,'----- all the numbers that found ------', data)
        # seen_titles_data = set()
        # new_list_data = []
        # for obj_data in data:
        #     if obj_data["text"] not in seen_titles_data:
        #         new_list_data.append(obj_data)
        #         seen_titles_data.add(obj_data["text"])
        # for item_data in new_list_data:
        #     print(item_data['text'])

        # print('----- fileid: ', fileId)
        # for item_data in arr_numbers:
        #     print(item_data['text'])
        # print('----- end ---------')
        output_obj['sums'] = data
        dict_group_by = {}
        if len(arr_numbers) == 0:
            for obj in data:
                if str(obj["text"]) not in dict_group_by:
                    dict_group_by[str(obj["text"])] = 1
                else:
                    dict_group_by[str(obj["text"])] = dict_group_by[str(obj["text"])] + 1
            if len(dict_group_by) > 0:
                dict_group_by = {k: v for k, v in
                                 sorted(dict_group_by.items(), key=lambda item: item[1], reverse=True)}
                if (list(dict_group_by.values())[0] > 1):
                    text_bigger_than_one = list(dict_group_by.keys())[0]
                    get_index_first_match = next(
                        (i for i, d in enumerate(data) if str(d['text']) == str(text_bigger_than_one)),
                        None)
                    # print_log(fileId,str(text_bigger_than_one) + ' bigger than 1', ' exist the first at index: ',
                    #           get_index_first_match)
                    arr_numbers.append(data[get_index_first_match])
                    arr_numbers.append(data[get_index_first_match + 1])
                    if (str(0.00) in dict_group_by) and dict_group_by[str(0.00)] > 0:
                        get_index_first__zero_match = next(
                            (i for i, d in enumerate(data) if str(d['text']) == str(0.00)),
                            None)
                        if get_index_first__zero_match is None:
                            get_index_first__zero_match = len(data) - 1
                        # print_log(fileId,'get_index_first__zero_match: ', str(get_index_first__zero_match))
                        arr_numbers.append(data[get_index_first__zero_match])
                    else:
                        obj = {
                            "vertices": [],
                            "text": float(0.00),
                            "conf": 0
                        }
                        arr_numbers.append(obj)

        if len(arr_numbers) > 0:
            for item_data in arr_numbers:
                item_data["text"] = format(float(item_data["text"]), ".2f")

        output_obj['data'] = arr_numbers
        # print_log(fileId,'arr_numbers', arr_numbers)

        if len(words) > 0:
            unique_words = {str(each['vertices']) + str(each['text']): each for each in words}.values()
            output_obj['words'] = list(unique_words)
            words_sorted = list(unique_words)
            sorted(words_sorted, key=lambda k: [k['vertices'][1]['x'], k['vertices'][1]['y']])
            # print(words_sorted)
            output_obj['words'] = words_sorted

        # for item_data in words_sorted:
        #     print(item_data['text'], item_data['vertices'][1]['x'])

        len_dates = len(dates)
        if len_dates > 0:
            unique_data_dates = {str(each['vertices']) + str(each['text']): each for each in dates}.values()
            data_dates = list(unique_data_dates)
            output_obj['dates'] = data_dates

        seen_titles_hp = set()
        new_list_hp = []
        for obj_hp in hp_list:
            if obj_hp["text"] not in seen_titles_hp:
                new_list_hp.append(obj_hp)
                seen_titles_hp.add(obj_hp["text"])
        output_obj['hp_list'] = new_list_hp
        # print_log(fileId,'----- all the hp_list that found ------', hp_list)

    except Exception as error_ocr:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_details = {
            'filename': exc_traceback.tb_frame.f_code.co_filename,
            'lineno': exc_traceback.tb_lineno,
            'function_name': exc_traceback.tb_frame.f_code.co_name,
            'type': exc_type.__name__,
            'message': str(exc_value)
        }
        del (exc_type, exc_value, exc_traceback)
        print_log(fileId, 'traceback_details: ', str(traceback_details))
        print_log(fileId, 'Error_ocr: ', str(error_ocr))

    res_ocr = output_obj
    # print_log(fileId,'res_ocr: ', res_ocr)
    del output_obj
    del dict_group_by
    del new_list_hp
    del width
    del height
    del heightReal
    del widthReal
    del grayImage_base
    del data
    del words
    del hp_list
    del arr_numbers
    gc.collect()
    return res_ocr


@ray.remote
def ocr_string(opencvImage_to_crop, angle):
    len_str_found = 0
    try:
        image_original_after_rotate_canny = opencvImage_to_crop
        if abs(angle) != 0:
            image_original_after_rotate_canny = rotate_cv(opencvImage_to_crop, angle, (255, 255, 255))

        osd_angle_ocr = pytesseract.image_to_string(image_original_after_rotate_canny,
                                                    lang='eng+heb')
        if len(osd_angle_ocr) > 0:
            # osd_angle_ocr = re.sub("[^0-9]", "", osd_angle_ocr)
            osd_angle_ocr = re.sub("[^0-9A-Za-z\u0590-\u05FF]", "", osd_angle_ocr)
            # osd_angle_ocr = osd_angle_ocr.replace(" ", "").replace("\n", " ").replace("\r", "")
            # osd_angle_ocr = ''.join(osd_angle_ocr.split())
        len_str_found = len(osd_angle_ocr)
        print('osd_angle: ', len_str_found, osd_angle_ocr)
    except Exception as errorAngle:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_details = {
            'filename': exc_traceback.tb_frame.f_code.co_filename,
            'lineno': exc_traceback.tb_lineno,
            'function_name': exc_traceback.tb_frame.f_code.co_name,
            'type': exc_type.__name__,
            'message': str(exc_value)
        }
        del (exc_type, exc_value, exc_traceback)
        print('traceback_details: ', str(traceback_details))
        print(str(errorAngle))
    return len_str_found


@ray.remote
def angle_correction(idx, grayscale):
    if idx == 0:
        determine_skew_angle = 0
        try:
            determine_skew_angle = determine_skew(grayscale)
            if determine_skew_angle is None:
                determine_skew_angle = 0
        except Exception as errorAngle:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback_details = {
                'filename': exc_traceback.tb_frame.f_code.co_filename,
                'lineno': exc_traceback.tb_lineno,
                'function_name': exc_traceback.tb_frame.f_code.co_name,
                'type': exc_type.__name__,
                'message': str(exc_value)
            }
            del (exc_type, exc_value, exc_traceback)
            print('traceback_details: ', str(traceback_details))
            print(str(errorAngle))
        return determine_skew_angle

    if idx == 1:
        image_after_first_rotate = cv2.Canny(grayscale, 100, 200)
        rotate_canny = {
            "angle": 0,
            "good_lang": ""
        }
        try:
            lang_and_angle_after_canny = get_lang_and_angle(image_after_first_rotate)
            rotate_canny["angle"] = int(lang_and_angle_after_canny["rotate"])
            rotate_canny["good_lang"] = str(lang_and_angle_after_canny["lang"])

        except Exception as errorAngle:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback_details = {
                'filename': exc_traceback.tb_frame.f_code.co_filename,
                'lineno': exc_traceback.tb_lineno,
                'function_name': exc_traceback.tb_frame.f_code.co_name,
                'type': exc_type.__name__,
                'message': str(exc_value)
            }
            del (exc_type, exc_value, exc_traceback)
            print('traceback_details: ', str(traceback_details))
            print(str(errorAngle))
        return rotate_canny

@ray.remote
def main_process():
    # print('main_process')
    # print('start ----', keyLocal)
    my_thread = None
    times_of_all_processes = {}
    # fileId = None
    # global times_of_all_processes, my_thread
    # global fileId
    fileId = ''
    path_yaml = "/opt/application.yaml"
    if platform.system() == "Darwin":
        path_yaml = "application.yaml"

    if os.path.exists(path_yaml):
        with open(path_yaml, 'r') as stream:
            try:
                data_loaded = yaml.safe_load(stream)
                sqs_py_result = data_loaded['sqs-py-result']
                sqs_separated = data_loaded['sqs-separated']
                target_bucket = data_loaded['target-bucket']
                # print_log(fileId,'Got params from application.yaml: ', data_loaded)
            except yaml.YAMLError as exc:
                print("Error in configuration file:", exc)
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback_details = {
                    'filename': exc_traceback.tb_frame.f_code.co_filename,
                    'lineno': exc_traceback.tb_lineno,
                    'function_name': exc_traceback.tb_frame.f_code.co_name,
                    'type': exc_type.__name__,
                    'message': str(exc_value)
                }
                del (exc_type, exc_value, exc_traceback)
                print_log(fileId, 'traceback_details: ', str(traceback_details))
                # sqs_py_result = 'https://sqs.eu-west-1.amazonaws.com/136511044100/BB-AWS-Dev-SQS_PY-RESULT'
                # sqs_separated = 'https://sqs.eu-west-1.amazonaws.com/136511044100/BB-AWS-Dev-SQS_SEPARATED'
                # target_bucket = 'bb-aligned-dev'
    else:
        print_log(fileId, 'application.yaml not exists')
        sqs_py_result = None
        sqs_separated = None
        target_bucket = None
    times = ""
    # times = times + "threadingName:" + message + ';'
    start = ''
    result = 0
    request_time = 0
    sub_key_value = False
    is_changed_img = False
    # my_thread = None
    string = ''
    # times_of_all_processes = {}
    jsonMerged = {}
    jsonStringA = json.dumps({})
    request_time_decode = 0
    # print_log(fileId,'Sqs send req to receive_message')
    if not is_dev_local:
        response = boto3.client('sqs').receive_message(
            QueueUrl=sqs_separated,
            AttributeNames=[
                'All'
            ],
            MessageAttributeNames=[
                'All',
            ],
            WaitTimeSeconds=20,
            MaxNumberOfMessages=1
        )
    else:
        response = {
            'Messages': [
                {
                    'MessageId': None,
                    'ReceiptHandle': None,
                    'Body': {
                        "Records": [
                            {
                                "s3": {
                                    "bucket": {"name": "bb-separated-dev"},
                                    "object": {"key": keyLocal},
                                    "is_local_file": True
                                }
                            }
                        ]
                    }
                }
            ]
        }

    try:
        # print(fileId,'receive_message: ', response)
        if ('Messages' in response) and len(response['Messages']) > 0 and ('Body' in response['Messages'][0]):
            data = (response['Messages'][0]['Body'])
            receiptHandle = response['Messages'][0]['ReceiptHandle']
            start = time.time()
            if not is_dev_local:
                data = json.loads(data)
            # print('Records Records Records')

            if ('Records' in data) and len(data['Records']) > 0 and ('s3' in data['Records'][0]):
                print_log(fileId, '--------Start a new process--------')
                data_of_doc = data['Records'][0]['s3']
                event = {
                    "bucket": data_of_doc['bucket']['name'],
                    "targetBucket": target_bucket,
                    "key": data_of_doc['object']['key']
                }
                print_log(fileId, 'params for run: ', str(event))
                print_log(fileId, 'Got path: main path (angle correction)')
                try:
                    if (('key' in event) and (type(event['key']) is str)) and (
                            ('bucket' in event) and (type(event['bucket']) is str)):
                        print_log(fileId, 'good params')
                        try:
                            bucketName = event['bucket']
                            bucketKey = parse.unquote_plus(event['key'], encoding='utf-8')
                            print_log(fileId, 'bucketKey: ', bucketKey)
                            fileId = bucketKey
                            targetBucketName = event['targetBucket']
                            print_log(fileId, 'targetBucketName: ', targetBucketName)
                            if is_dev_local and ('is_local_file' in data_of_doc) and data_of_doc[
                                'is_local_file']:
                                download_path = bucketKey
                            else:
                                start_download_file = time.time()
                                download_path = '/tmp/{}{}'.format(uuid.uuid4(), bucketKey.replace("/", ""))
                                boto3.client('s3').download_file(bucketName, bucketKey, download_path, Config=config)
                                request_time_downloaded = time.time() - start_download_file
                                times = times + "downloadTheFile:" + (
                                    "{0:.0f}s".format(request_time_downloaded)) + ';'
                                times_of_all_processes['download_file'] = (
                                    "{0:.0f}s".format(request_time_downloaded))

                            output_path = '/tmp/{}'.format(bucketKey.replace("/", ""))
                            dpi = getDpi(download_path)
                            print_log(fileId, 'DPI of img is: ', dpi)
                            mime_type = False
                            try:
                                mime_type = mimetypes.guess_type(download_path)[0]
                                print_log(fileId, "The mimetypes is: ", mime_type)
                            except Exception as mime_err:
                                print_log(fileId, 'mime_err: ', mime_err)

                            image_download = Image.open(download_path)
                            get_size_of_file_by_mb = file_size(download_path)
                            print_log(fileId, 'MB of file is: ', get_size_of_file_by_mb)
                            mb_file = float(get_size_of_file_by_mb)
                            quality = 100
                            if mb_file > 1.5:
                                quality = 95
                            basewidth = 2480
                            if float(image_download.size[0]) > float(image_download.size[1]):
                                basewidth = 3508
                            wpercent = (basewidth / float(image_download.size[0]))
                            hsize = int((float(image_download.size[1]) * float(wpercent)))
                            im_resized = image_download.resize((basewidth, hsize), Image.ANTIALIAS)
                            if image_download.mode in ["RGBA", "P"]:
                                im_resized = im_resized.convert("RGB")
                            if dpi and dpi < 500:
                                im_resized.save(download_path,
                                                dpi=(500, 500),
                                                format="JPEG",
                                                optimize=True,
                                                quality=quality)
                            else:
                                im_resized.save(download_path,
                                                format="JPEG",
                                                optimize=True,
                                                quality=quality)

                            if download_path.find('.png') != -1:
                                os.rename(download_path, download_path.replace(".png", ".jpg"))
                                download_path = download_path.replace(".png", ".jpg")
                            if download_path.find('.jpg') == -1 and download_path.find('.jpeg') == -1:
                                os.rename(download_path, download_path + ".jpg")
                                download_path = download_path + ".jpg"

                            times_of_all_processes['angle_correction'] = time.time()
                            is_changed_img_type = False
                            # image_original_after_rotate = False
                            # image_download_for_angle = Image.open(download_path)
                            image_download_for_angle = cv2.imread(download_path)

                            # image_download_for_angle_copy = image_download_for_angle.copy()
                            # width_to_set = 2480
                            # if float(image_download_for_angle.size[0]) > float(image_download_for_angle.size[1]):
                            #     width_to_set = 3508
                            # wpercent = (width_to_set / float(image_download_for_angle.size[0]))
                            # hsize = int((float(image_download_for_angle.size[1]) * float(wpercent)))
                            # im_resized_for_angle = image_download_for_angle.resize((width_to_set, hsize),
                            #                                                        Image.ANTIALIAS)
                            # np.array(image_download_for_angle)

                            # scale_percent = 95
                            # width_n = int(image_download_for_angle.shape[1] * scale_percent / 100)
                            # height_n = int(image_download_for_angle.shape[0] * scale_percent / 100)
                            # dim = (width_n, height_n)
                            # resized = cv2.resize(image_download_for_angle, dim, interpolation=cv2.INTER_AREA)
                            # opencvImage_to_crop = image_download_for_angle
                            grayscale = cv2.cvtColor(image_download_for_angle, cv2.COLOR_BGR2GRAY)

                            # morph = opencvImage_to_crop
                            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                            # morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
                            # morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
                            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                            # gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)
                            # image_channels = np.split(np.asarray(gradient_image), 3, axis=2)
                            # channel_height, channel_width, _ = image_channels[0].shape
                            # for i in range(0, 3):
                            #     _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255,
                            #                                          cv2.THRESH_OTSU | cv2.THRESH_BINARY)
                            #     image_channels[i] = np.reshape(image_channels[i],
                            #                                    newshape=(channel_height, channel_width, 1))
                            # based_color = np.concatenate((image_channels[0], image_channels[1], image_channels[2]),
                            #                              axis=2)

                            # based_color = cv2.cvtColor(opencvImage_to_crop, cv2.COLOR_RGB2BGR)

                            determine_skew_angle = determine_skew(grayscale)
                            print_log(fileId, 'determine_skew_angle', determine_skew_angle)
                            if determine_skew_angle is None:
                                determine_skew_angle = 0
                            if abs(determine_skew_angle) != 0:
                                image_download_for_angle = rotate_cv(image_download_for_angle, determine_skew_angle,
                                                                     (255, 255, 255))
                                grayscale = cv2.cvtColor(image_download_for_angle, cv2.COLOR_BGR2GRAY)

                            rotate_canny = {
                                "angle": 0,
                                "good_lang": ""
                            }
                            angle = 0
                            try:
                                lang_and_angle_after_canny = get_lang_and_angle(grayscale)
                                angle = int(lang_and_angle_after_canny["rotate"])
                                rotate_canny["good_lang"] = str(lang_and_angle_after_canny["lang"])
                                print_log(fileId, 'angle', angle)

                            except Exception as errorAngle:
                                pass

                            if abs(angle) != 0:
                                image_download_for_angle = rotate_cv(image_download_for_angle, angle, (255, 255, 255))

                            if abs(determine_skew_angle) != 0 or abs(angle) != 0:
                                is_changed_img_type = True

                            del grayscale
                            # results = ray.get(
                            #     [angle_correction.remote(0, grayscale), angle_correction.remote(1, grayscale)])
                            # determine_skew_angle = results[0]
                            # rotate_canny = results[1]
                            # print_log(fileId,'----------')
                            # print_log(fileId,'determine_skew_angle', determine_skew_angle)
                            # print_log(fileId,'rotate_canny', rotate_canny["angle"])
                            # print_log(fileId,'good_lang', rotate_canny["good_lang"])
                            #
                            # # print_log(fileId,'is determine_skew_angle close to right angled: ',
                            # #           ((abs(abs(determine_skew_angle) - 0) <= 8) or (
                            # #                   abs(abs(determine_skew_angle) - 90) <= 8) or (
                            # #                    abs(abs(determine_skew_angle) - 180) <= 8) or (
                            # #                    abs(abs(determine_skew_angle) - 270) <= 8) or (
                            # #                    abs(abs(determine_skew_angle) - 360) <= 8)))
                            # print_log(fileId,'----------')
                            #
                            # if determine_skew_angle == int(rotate_canny["angle"]) or rotate_canny[
                            #     "good_lang"] == "Han" or ((str(rotate_canny["good_lang"]) == "Arabic" or str(
                            #     rotate_canny["good_lang"]) == "Hebrew" or str(
                            #     rotate_canny["good_lang"]) == "Malayalam") and (
                            #                                       abs(determine_skew_angle) == 0 or abs(
                            #                                   determine_skew_angle) == 90 or abs(
                            #                                   determine_skew_angle) == 91 or abs(
                            #                                   determine_skew_angle) == 180 or abs(
                            #                                   determine_skew_angle) == 181 or abs(
                            #                                   determine_skew_angle) == 270)):
                            #     angle_match = int(rotate_canny["angle"])
                            # else:
                            #     # scale_percent = 90
                            #     # width_n = int(image_download_for_angle.shape[1] * scale_percent / 100)
                            #     # height_n = int(image_download_for_angle.shape[0] * scale_percent / 100)
                            #     # dim = (width_n, height_n)
                            #     # resized = cv2.resize(image_download_for_angle, dim, interpolation=cv2.INTER_AREA)
                            #     opencvImage_to_crop = image_download_for_angle
                            #     # imgheight = opencvImage_to_crop.shape[0]
                            #     # imgwidth = opencvImage_to_crop.shape[1]
                            #     # if imgheight > imgwidth:
                            #     #     new_width = (imgwidth / 100) * 85
                            #     #     new_height = (imgheight / 100) * 65
                            #     # else:
                            #     #     new_width = (imgwidth / 100) * 65
                            #     #     new_height = (imgheight / 100) * 85
                            #     # start_width = (imgwidth - new_width) / 2
                            #     # end_width = start_width + new_width
                            #     # start_height = (imgheight - new_height) / 2
                            #     # end_height = start_height + new_height
                            #     # opencvImage_to_crop = opencvImage_to_crop[
                            #     #                       int(start_height):int(end_height),
                            #     #                       int(start_width):int(end_width)]
                            #     # cv2.imshow('image', opencvImage_to_crop)
                            #     # cv2.waitKey(0)
                            #     # cv2.destroyAllWindows()
                            #
                            #     print_log(fileId,'Check between two angle - the ocr output')
                            #     # gray = cv2.cvtColor(opencvImage_to_crop, cv2.COLOR_BGR2GRAY)
                            #     # ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
                            #     results_ocr_string = ray.get(
                            #         [ocr_string.remote(grayscale, determine_skew_angle),
                            #          ocr_string.remote(grayscale, int(rotate_canny["angle"]))])
                            #     len_str_after_skew = results_ocr_string[0]
                            #     len_str_after_osd = results_ocr_string[1]
                            #
                            #     print_log(fileId,'The length of ocr output -  len_str_after_skew: ', len_str_after_skew)
                            #     print_log(fileId,'The length of ocr output -  len_str_after_osd: ', len_str_after_osd)
                            #
                            #     if len_str_after_skew > (len_str_after_osd + 1):
                            #         angle_match = determine_skew_angle
                            #         if abs(angle_match) != 0 and abs(angle_match) != 90 and abs(
                            #                 angle_match) != 180 and abs(angle_match) != 270:
                            #             image_original_after_rotate_skew = rotate_cv(opencvImage_to_crop,
                            #                                                          determine_skew_angle,
                            #                                                          (255, 255, 255))
                            #
                            #             grayscale = cv2.cvtColor(image_original_after_rotate_skew, cv2.COLOR_BGR2GRAY)
                            #
                            #             # scale_percent = 95
                            #             # width_n = int(image_original_after_rotate_skew.shape[1] * scale_percent / 100)
                            #             # height_n = int(image_original_after_rotate_skew.shape[0] * scale_percent / 100)
                            #             # dim = (width_n, height_n)
                            #             # resized = cv2.resize(image_original_after_rotate_skew, dim,
                            #             #                      interpolation=cv2.INTER_AREA)
                            #             #
                            #             # morph = resized
                            #             # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                            #             # morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
                            #             # morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
                            #             # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                            #             # gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)
                            #             # image_channels = np.split(np.asarray(gradient_image), 3, axis=2)
                            #             # channel_height, channel_width, _ = image_channels[0].shape
                            #             # for i in range(0, 3):
                            #             #     _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255,
                            #             #                                          cv2.THRESH_OTSU | cv2.THRESH_BINARY)
                            #             #     image_channels[i] = np.reshape(image_channels[i],
                            #             #                                    newshape=(channel_height, channel_width, 1))
                            #             # based_color = np.concatenate(
                            #             #     (image_channels[0], image_channels[1], image_channels[2]),
                            #             #     axis=2)
                            #
                            #             results = ray.get(
                            #                 [angle_correction.remote(1, grayscale)])
                            #             rotate_canny_after_skew = results[0]
                            #
                            #             print_log(fileId,'determine_osd_angle_after_skew',
                            #                       int(rotate_canny_after_skew["angle"]))
                            #             print_log(fileId,'total determine_osd_angle_after_skew',
                            #                       (determine_skew_angle + int(rotate_canny_after_skew["angle"])))
                            #
                            #             # gray = cv2.cvtColor(image_original_after_rotate_skew, cv2.COLOR_BGR2GRAY)
                            #             # ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
                            #             results_ocr_string = ray.get(
                            #                 [ocr_string.remote(image_original_after_rotate_skew,
                            #                                    (int(rotate_canny_after_skew["angle"])))])
                            #             len_str_after_skew_osd = results_ocr_string[0]
                            #             print_log(fileId,'The length of ocr output -  len_str_after_second_osd after skew: ',
                            #                       len_str_after_skew_osd)
                            #             if len_str_after_skew_osd > len_str_after_skew:
                            #                 angle_match = determine_skew_angle + int(rotate_canny_after_skew["angle"])
                            #     else:
                            #         angle_match = int(rotate_canny["angle"])
                            #         if abs(determine_skew_angle) < 3 and rotate_canny == 0:
                            #             angle_match = angle_match + determine_skew_angle
                            #
                            #     # results_ocr_string_second = ray.get(
                            #     #     [ocr_string.remote(opencvImage_to_crop, angle_match + 90), ocr_string.remote(opencvImage_to_crop, angle_match + 180), ocr_string.remote(opencvImage_to_crop, angle_match + 270)])
                            #     # if len_str_after_skew > len_str_after_osd:
                            #     #     results_ocr_string_second.insert(0, len_str_after_skew)
                            #     # else:
                            #     #     results_ocr_string_second.insert(0, len_str_after_osd)
                            #     #
                            #     # print(results_ocr_string_second)
                            #     # max_len_str_ocr = max(results_ocr_string_second)
                            #     # index_the_max = results_ocr_string_second.index(max_len_str_ocr)
                            #     # print_log(fileId,'max_len_str_ocr: ', max_len_str_ocr, 'index_the_max: ', index_the_max)
                            #     #
                            #     # if index_the_max == 0:
                            #     #     angle_match = angle_match
                            #     # elif index_the_max == 1:
                            #     #     angle_match = angle_match + 90
                            #     # elif index_the_max == 2:
                            #     #     angle_match = angle_match + 180
                            #     # elif index_the_max == 3:
                            #     #     angle_match = angle_match + 270
                            #
                            # print_log(fileId,'angle need to be: ', angle_match)
                            #
                            # if abs(angle_match) != 0:
                            #     image_original_after_rotate = rotate_cv(image_download_for_angle_copy,
                            #                                             angle_match, (255, 255, 255))

                            # is_changed_img_type = type(image_original_after_rotate).__module__ == np.__name__
                            print_log(fileId, 'is_changed_img: ', is_changed_img_type)
                            times_of_all_processes['angle_correction'] = (
                                "{0:.0f}s".format(time.time() - times_of_all_processes['angle_correction']))

                            if is_changed_img_type:
                                times_of_all_processes['write_image_after_correction'] = time.time()
                                cv2.imwrite(download_path, image_download_for_angle)
                                del image_download_for_angle
                                image_download = Image.open(download_path)
                                get_size_of_file_by_mb = file_size(download_path)
                                print_log(fileId, 'MB of file is: ', get_size_of_file_by_mb)
                                mb_file = float(get_size_of_file_by_mb)
                                dpi = getDpi(download_path)
                                print_log(fileId, 'DPI of img is: ', dpi)
                                quality = 100
                                if mb_file > 1.5:
                                    quality = 95
                                basewidth = 2480
                                if float(image_download.size[0]) > float(image_download.size[1]):
                                    basewidth = 3508
                                wpercent = (basewidth / float(image_download.size[0]))
                                hsize = int((float(image_download.size[1]) * float(wpercent)))
                                im_resized = image_download.resize((basewidth, hsize), Image.ANTIALIAS)
                                if dpi and dpi < 500:
                                    im_resized.save(download_path,
                                                    dpi=(500, 500),
                                                    format="JPEG",
                                                    optimize=True,
                                                    quality=quality)
                                else:
                                    im_resized.save(download_path,
                                                    format="JPEG",
                                                    optimize=True,
                                                    quality=quality)

                                get_size_of_file_by_mb = file_size(download_path)
                                print_log(fileId, 'MB of file after reduction is: ', get_size_of_file_by_mb)

                                times_of_all_processes['write_image_after_correction'] = (
                                    "{0:.0f}s".format(
                                        time.time() - times_of_all_processes['write_image_after_correction']))
                            else:
                                print_log(fileId, 'Result: Unchanged angle')

                            if not is_dev_local or (
                                    ('is_local_file' in data_of_doc) and not data_of_doc['is_local_file']):
                                my_thread = threading.Thread(target=uploadingToBucket, args=(
                                    download_path, targetBucketName, bucketKey, times_of_all_processes, fileId))
                                my_thread.setDaemon(True)
                                my_thread.start()

                            try:
                                # image_download = Image.open(download_path)
                                # width_original, height_original = image_download.size
                                # basewidth = 1754
                                # wpercent = (basewidth / float(image_download.size[0]))
                                # hsize = int((float(image_download.size[1]) * float(wpercent)))
                                # im_resized = image_download.resize((basewidth, hsize), Image.ANTIALIAS)
                                # im_resized.save(output_path,
                                #                 format="JPEG",
                                #                 optimize=True,
                                #                 quality=100)

                                image_download = cv2.imread(download_path)
                                height_original = image_download.shape[0]
                                width_original = image_download.shape[1]
                                # scale_percent = 80
                                # width_n = int(width_original * scale_percent / 100)
                                # height_n = int(height_original * scale_percent / 100)
                                # dim = (width_n, height_n)
                                # resized = cv2.resize(image_download, dim, interpolation=cv2.INTER_AREA)
                                gray = cv2.cvtColor(image_download, cv2.COLOR_BGR2GRAY)
                                times_of_all_processes['ocr_process'] = time.time()
                                json_string = get_ocr(image_download, bucketKey, width_original, height_original, gray,
                                                      download_path, fileId)
                                times_of_all_processes['ocr_process'] = (
                                    "{0:.0f}s".format(
                                        time.time() - times_of_all_processes['ocr_process']))

                                if not is_dev_local:
                                    # if os.path.exists(output_path):
                                    #     os.remove(output_path)
                                    #     print_log(fileId,'The output_path removed from the tmp location!')
                                    # else:
                                    #     print_log(fileId,"The output_path does not exist in the path: ", output_path)
                                    times_of_all_processes['delete_message'] = time.time()
                                    response_delete_message = boto3.client('sqs').delete_message(
                                        QueueUrl=sqs_separated,
                                        ReceiptHandle=receiptHandle
                                    )
                                    print_log(fileId, 'response_delete_message: ', response_delete_message)
                                    times_of_all_processes['delete_message'] = (
                                        "{0:.0f}s".format(
                                            time.time() - times_of_all_processes['delete_message']))

                                request_time = time.time() - start
                                times_of_all_processes['total_process_time'] = ("{0:.0f}s".format(request_time))
                                print_log(fileId, 'times_of_all_processes: ', str(times_of_all_processes))
                                json_string['total_process_time'] = request_time
                                json_string = json.dumps(json_string, ensure_ascii=False, indent=4)
                                is_validate = validateJSON(json_string)
                                if not is_validate:
                                    json_string = json.dumps({})

                                # sizeof_sums_array = (sys.getsizeof(json_string) / 1024)
                                # print_log(fileId,'sizeof json_string with sums array is :', sizeof_sums_array)
                                # if sizeof_sums_array >= 256:
                                #     json_base = json.loads(json_string)
                                #     if 'sums' in json_base:
                                #         del json_base["sums"]
                                #     json_string = json.dumps(json_base, ensure_ascii=False, indent=4)

                                json_base = json.loads(json_string)
                                times_of_all_processes['send_message'] = time.time()
                                # print_log(fileId,'Data to send: ', str(json.loads(json_string)))

                                if is_dev_local:
                                    json_path = bucketKey + '_json.json'
                                    with open(json_path, 'w') as outfile:
                                        outfile.write(json_string)
                                        print("New json file is created")

                                message_body = {}
                                message_body["fileId"] = json_base["fileId"]
                                # if json_base["doc_number"] is not None:
                                #     message_body["doc_number"] = json_base["doc_number"]
                                message_body["total_steps"] = json_base["total_steps"]
                                message_body["total_process_time"] = json_base["total_process_time"]
                                if not is_dev_local:
                                    times_of_all_processes['uploadingJsonToBucket'] = time.time()
                                    try:
                                        mime_type = 'application/json'

                                        if 'fileId' in json_base:
                                            del json_base["fileId"]
                                        if 'total_steps' in json_string:
                                            del json_base["total_steps"]
                                        if 'total_process_time' in json_string:
                                            del json_base["total_process_time"]

                                        json_base_aws = None
                                        if 'aws_textract' in json_base:
                                            json_base_aws = json_base["aws_textract"]
                                            del json_base["aws_textract"]

                                        json_path = '/tmp/' + bucketKey + '_json.json'
                                        with open(json_path, 'w') as outfile:
                                            json_to_create = json.dumps(json_base, ensure_ascii=False, indent=4)
                                            outfile.write(json_to_create)
                                            print("New json file is created")
                                        print_log(fileId, 'Start uploading JSON file to output bucket')
                                        boto3.client('s3').upload_file(json_path, targetBucketName, bucketKey + '_json',
                                                             ExtraArgs={"ContentType": mime_type},
                                                             Config=config)
                                        message_body["python_results"] = bucketKey + '_json'

                                        amz_json_path = '/tmp/' + bucketKey + '_amz_json.json'
                                        if json_base_aws is not None:
                                            with open(amz_json_path, 'w') as outfile:
                                                json_base_aws_to_create = json.dumps(json_base_aws, ensure_ascii=False,
                                                                                     indent=4)
                                                outfile.write(json_base_aws_to_create)
                                                print("New amz_json file is created")
                                            boto3.client('s3').upload_file(amz_json_path, targetBucketName,
                                                                 bucketKey + '_amz_json',
                                                                 ExtraArgs={"ContentType": mime_type},
                                                                 Config=config)
                                            message_body["amz_results"] = bucketKey + '_amz_json'

                                        print_log(fileId, 'Finished to upload the JSON file to output bucket')

                                        if os.path.exists(json_path):
                                            os.remove(json_path)
                                            print_log(fileId, 'The json_path removed from the tmp location!')
                                        else:
                                            print_log(fileId, "The json_path does not exist in the path: ", json_path)

                                        if os.path.exists(amz_json_path):
                                            os.remove(amz_json_path)
                                            print_log(fileId, 'The amz_json_path removed from the tmp location!')
                                        else:
                                            print_log(fileId, "The amz_json_path does not exist in the path: ",
                                                      amz_json_path)

                                    except Exception as e:
                                        exc_type, exc_value, exc_traceback = sys.exc_info()
                                        traceback_details = {
                                            'filename': exc_traceback.tb_frame.f_code.co_filename,
                                            'lineno': exc_traceback.tb_lineno,
                                            'function_name': exc_traceback.tb_frame.f_code.co_name,
                                            'type': exc_type.__name__,
                                            'message': str(exc_value)
                                        }
                                        del (exc_type, exc_value, exc_traceback)
                                        print_log(fileId, 'traceback_details: ', str(traceback_details))
                                        print_log(fileId, str(e))
                                        print_log(fileId, 'Error uploading JSON file to output bucket')

                                    times_of_all_processes['uploadingJsonToBucket'] = (
                                        "{0:.0f}s".format(
                                            time.time() - times_of_all_processes['uploadingJsonToBucket']))

                                    print('message_body: ', message_body)
                                    response_send_message = boto3.client('sqs').send_message(
                                        QueueUrl=sqs_py_result,
                                        MessageBody=json.dumps(message_body, ensure_ascii=False, indent=4)
                                    )
                                    print_log(fileId, 'response_send_message: ', response_send_message)
                                    times_of_all_processes['send_message'] = (
                                        "{0:.0f}s".format(
                                            time.time() - times_of_all_processes['send_message']))

                            except Exception as errorOcr:
                                exc_type, exc_value, exc_traceback = sys.exc_info()
                                traceback_details = {
                                    'filename': exc_traceback.tb_frame.f_code.co_filename,
                                    'lineno': exc_traceback.tb_lineno,
                                    'function_name': exc_traceback.tb_frame.f_code.co_name,
                                    'type': exc_type.__name__,
                                    'message': str(exc_value)
                                }
                                del (exc_type, exc_value, exc_traceback)
                                print_log(fileId, 'traceback_details: ', str(traceback_details))
                                print_log(fileId, 'errorOcr: ', str(errorOcr))
                                print_log(fileId, 'times_of_all_processes: ', str(times_of_all_processes))
                                # del times_of_all_processes
                                if not is_dev_local:
                                    json_string = json.dumps({
                                        'fileId': bucketKey,
                                        'errorMessage': str(errorOcr)
                                    }, ensure_ascii=False, indent=4)
                                    response_send_message = boto3.client('sqs').send_message(
                                        QueueUrl=sqs_py_result,
                                        MessageBody=json_string
                                    )
                                    print_log(fileId, 'response_send_error_message: ', response_send_message)
                        except Exception as error:
                            print_log(fileId, str(error))
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                            traceback_details = {
                                'filename': exc_traceback.tb_frame.f_code.co_filename,
                                'lineno': exc_traceback.tb_lineno,
                                'function_name': exc_traceback.tb_frame.f_code.co_name,
                                'type': exc_type.__name__,
                                'message': str(exc_value)
                            }
                            del (exc_type, exc_value, exc_traceback)
                            print_log(fileId, 'traceback_details: ', str(traceback_details))
                            if not is_dev_local:
                                json_string = json.dumps({
                                    'fileId': bucketKey,
                                    'errorMessage': str(error)
                                }, ensure_ascii=False, indent=4)
                                response_send_message = boto3.client('sqs').send_message(
                                    QueueUrl=sqs_py_result,
                                    MessageBody=json_string
                                )
                                print_log(fileId, 'response_send_error_message: ', response_send_message)

                    else:
                        print_log(fileId, 'bad req')
                except Exception as errorParams:
                    print_log(fileId, 'bad req: ', str(errorParams))
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback_details = {
                        'filename': exc_traceback.tb_frame.f_code.co_filename,
                        'lineno': exc_traceback.tb_lineno,
                        'function_name': exc_traceback.tb_frame.f_code.co_name,
                        'type': exc_type.__name__,
                        'message': str(exc_value)
                    }
                    del (exc_type, exc_value, exc_traceback)
                    print_log(fileId, 'traceback_details: ', str(traceback_details))
                    print_log(fileId, 'times_of_all_processes: ', str(times_of_all_processes))
                    # del times_of_all_processes
                    if not is_dev_local:
                        json_string = json.dumps({
                            'fileId': bucketKey,
                            'errorMessage': str(errorParams)
                        }, ensure_ascii=False, indent=4)
                        response_send_message = boto3.client('sqs').send_message(
                            QueueUrl=sqs_py_result,
                            MessageBody=json_string
                        )
                        print_log(fileId, 'response_send_error_message: ', response_send_message)

                if not is_dev_local and is_changed_img_type and my_thread is not None:
                    while my_thread.is_alive():
                        pass

                if not is_dev_local:
                    if os.path.exists(download_path):
                        os.remove(download_path)
                        print_log(fileId, 'The download_path removed from the tmp location!')
                    else:
                        print_log(fileId, "The download_path does not exist in the path: ", download_path)

                del event
                del bucketName
                del bucketKey
                if sub_key_value:
                    del sub_key_value
                del download_path
                del output_path
                del is_changed_img
                del string
                del times
                del jsonMerged
                print_log(fileId, ("Total process time was: {0:.0f}s".format(request_time)))
                del request_time
                del request_time_decode
                del jsonStringA
                gc.collect()
                print_log(fileId, 'Removes the variable from the namespace and clear the memory')
                # del times_of_all_processes
                print_log(fileId, '--------End of the process--------')
                if not is_dev_local:
                    return True
                else:
                    return True
            else:
                # print_log(fileId,'Records/s3 not founds')
                # print_log(fileId,'times_of_all_processes: ', str(times_of_all_processes))
                # # del times_of_all_processes
                # print_log(fileId,'--------End of the process--------')
                if not is_dev_local:
                    return True
                else:
                    return True
        else:
            # print_log(fileId,'Messages not founds')
            # print_log(fileId,'times_of_all_processes: ', str(times_of_all_processes))
            # del times_of_all_processes
            # print_log(fileId,'--------End of the process--------')
            if not is_dev_local:
                return True
            else:
                return True
    except Exception as errorResData:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_details = {
            'filename': exc_traceback.tb_frame.f_code.co_filename,
            'lineno': exc_traceback.tb_lineno,
            'function_name': exc_traceback.tb_frame.f_code.co_name,
            'type': exc_type.__name__,
            'message': str(exc_value)
        }
        del (exc_type, exc_value, exc_traceback)
        # print_log(fileId,'times_of_all_processes: ', str(times_of_all_processes))
        # del times_of_all_processes
        # gc.collect()
        print_log(fileId, 'errorResData: ', str(traceback_details))
        # print_log(fileId,'--------End of the process--------')
        if not is_dev_local:
            return True
        else:
            return True


if __name__ == "__main__":
    fileId = ''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logHandler = logging.StreamHandler()
    logHandler.setFormatter(jsonlogger.JsonFormatter('%(timestamp)s  %(message)s ', timestamp=True))
    logger.addHandler(logHandler)
    logging.basicConfig(level=logging.INFO)
    # print(psutil.Process())
    # print(num_of_worker_threads)
    # print('len(multiprocessing.active_children())---', len(multiprocessing.active_children()))
    # print('num_cpus--', num_cpus, list(range(psutil.cpu_count())), psutil.cpu_percent(percpu=True))
    try:
        if not is_dev_local:
            server_thread = threading.Thread(target=run,
                                             args=())
            server_thread.setDaemon(False)
            server_thread.start()
            number_allow_thread = math.ceil(num_of_worker_threads / 10)
            if number_allow_thread == 1:
                while True:
                    obj_ref = main_process.remote()
                    res = ray.get(obj_ref)
                    # main_process()
            else:
                len_thread_to_run = [*range(1, (1 + number_allow_thread), 1)]
                try:
                    while True:
                        results = ray.get(
                            [main_process.remote() for i in len_thread_to_run], timeout=60)

                    # with concurrent.futures.ThreadPoolExecutor(max_workers=(1 + number_allow_thread)) as executor:
                    #     while True:
                    #         print('start new threads ', number_allow_thread)
                    #         future_to_thr = {executor.submit(main_process): thr for thr in len_thread_to_run}
                    #         for future in concurrent.futures.as_completed(future_to_thr):
                    #             thr = future_to_thr[future]
                    #             exception = future.exception()
                    #             # handle exceptional case
                    #             if exception:
                    #                 print(exception)
                    #             else:
                    #                 try:
                    #                     data = future.result()
                    #                     print('complete another thread', data)
                    #                 except Exception as exc:
                    #                     print('%r generated an exception: %s' % (thr, exc))
                except Exception as excs:
                    print('Exception ThreadPoolExecutor', excs)
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback_details = {
                        'filename': exc_traceback.tb_frame.f_code.co_filename,
                        'lineno': exc_traceback.tb_lineno,
                        'function_name': exc_traceback.tb_frame.f_code.co_name,
                        'type': exc_type.__name__,
                        'message': str(exc_value)
                    }
                    del (exc_type, exc_value, exc_traceback)
                    gc.collect()
                    print('traceback_details: ', traceback_details)
                    print("Error: ThreadPoolExecutor")


        else:
            print('Start to run local')
            start_local = time.time()
            obj_ref = main_process.remote()
            res = ray.get(obj_ref)
            # main_process()
            print('Total time run local: ', ("{0:.0f}s".format(time.time() - start_local)))
            # results = ray.get(
            #         [main_process.remote() for i in range(1, 3)], timeout=50)
            # start = time.time()
            # with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
            #     for filename in os.listdir('files'):
            #         filepath = os.path.join('files', filename)
            #         print('filepath:', filepath)
            #         executor.map(main_process)
            #     print('end-----', "{0:.0f}s".format(time.time() - start))

        # ocr_main_thread = threading.Thread(target=ocr,
        #                                    args=())
        # ocr_main_thread.setDaemon(True)
        # ocr_main_thread.start()
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_details = {
            'filename': exc_traceback.tb_frame.f_code.co_filename,
            'lineno': exc_traceback.tb_lineno,
            'function_name': exc_traceback.tb_frame.f_code.co_name,
            'type': exc_type.__name__,
            'message': str(exc_value)
        }
        del (exc_type, exc_value, exc_traceback)
        gc.collect()
        print(fileId, 'traceback_details: ', traceback_details)
        print("Error: unable to start thread")
    # if not is_dev_local:
    #     while 1:
    #         pass
