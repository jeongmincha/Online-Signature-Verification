# coding: utf-8
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys


def remove_image(path):
    filenames = os.listdir(path)
    person_num = len(filenames)
    print(person_num)
    for indir in filenames:
        full_filename = os.path.join(path, indir)
        for infile in (glob.glob(full_filename + '/*_rs_rs.PNG') + glob.glob(full_filename + '/*_rs_rs.png')): 
            os.remove(infile)


def resize_image(path, width=750, height=360):
    size = width, height
    
    for infile in (glob.glob(path + '/*.PNG') + glob.glob(path + '/*.png')):
        outfile = os.path.splitext(infile)[0] + "_rs" + os.path.splitext(infile)[1] 
        if infile != outfile:
            try:
                im = Image.open(infile)
                rs_im = im.resize(size)
                arr = np.asarray(rs_im).copy()
                bi_tx = binarize_array(arr, 220)
                bi_im = Image.fromarray(bi_tx)
                bi_im.save(outfile, "PNG")
            except IOError:
                print ("cannot create thumbnail for '%s'" % infile)


def resize_image_dir(path, width=750, height=360):
    filenames = os.listdir(path)
    person_num = len(filenames)
    print(person_num)
    for indir in filenames:
        full_filename = os.path.join(path, indir)
        resize_image(full_filename, width, height)


def image_load_dir_user(path):
    filenames = os.listdir(path)
    person_num = len(filenames)
    print(person_num)
    
    uid_table = np.zeros((person_num,))
    g_image_d_list = []
    f_image_d_list = []
    for uid, indir in enumerate(filenames):
        full_filename = os.path.join(path, indir)
        
        file_uid = int(indir)
        uid_table[uid] = file_uid
        
        image_list, _, _ = image_load_file(full_filename, uid)
        g_image_d_list.append(image_list[0])
        f_image_d_list.append(image_list[1])
        
    return  g_image_d_list, f_image_d_list, uid_table


def image_load_file(path, uid):
    path_len = len(path)
    Genuine_image_list = []
    Forger_image_list = []
    
    Genuine_uid_list = []
    Forger_uid_list = []
    
    Genuine_isF_list = []
    Forger_isF_list = []
    for filename in (glob.glob(path + '/*_rs.PNG') + glob.glob(path + '/*_rs.png')):
        image = Image.open(filename)
        without_extension = os.path.splitext(filename)[0] #splittext = 확장자(.png or .jpeg ...)를 기준으로 두 부분으로 나눔
        parse_filename = without_extension[path_len:].split('_')
        user_id = uid
        if(len(parse_filename[1]) > 3):
            Forger_image_list.append(image)
            Forger_uid_list.append(int(user_id))
            Forger_isF_list.append(True)
        elif(len(parse_filename[1]) == 3):
            #user_id = parse_filename[1]
            Genuine_image_list.append(image)
            Genuine_uid_list.append(int(user_id))
            Genuine_isF_list.append(False)
        else:
            sys.exit()
            
    return (Genuine_image_list,Forger_image_list), (Genuine_uid_list,Forger_uid_list), (Genuine_isF_list,Forger_isF_list)


def image_load_dir(path):
    filenames = os.listdir(path)
    person_num = len(filenames)
    
    uid_table = np.zeros((person_num,))
    image_d_list = []
    uid_d_list = []
    isF_d_list = []
    
    for uid, indir in enumerate(filenames):
        full_filename = os.path.join(path, indir)
        
        file_uid = int(indir)
        uid_table[uid] = file_uid
        
        image_list, uid_list, isF_list = image_load_file(full_filename, uid)
        image_d_list += (image_list[0] + image_list[1])
        uid_d_list += (uid_list[0] + uid_list[1])
        isF_d_list += (isF_list[0] + isF_list[1])
        
    arri = np.array([np.array(i) for i in image_d_list])
    arru = np.array(uid_d_list)
    arrf = np.array(isF_d_list)
    print (arri.shape)
    return  arri, arru, arrf, uid_table


def image_load(path):
    
    path_len = len(path)
    num_person = 0
    num_sig = 0
    num_person, num_sig = _getNumPersonSig(path)
    dataset = np.zeros((num_person, num_sig, 360, 750, 3),dtype = np.uint8)
    print(dataset.shape)
    
    for filename in (glob.glob(path + '/*_rs.PNG') + glob.glob(path + '/*_rs.png')):
        without_extension = os.path.splitext(filename)[0] #splittext = 확장자(.png or .jpeg ...)를 기준으로 두 부분으로 나눔
        parse_filename = without_extension[path_len:].split('_')
        pre_len = len(parse_filename[0])
        person_id = int(parse_filename[0][pre_len-3:])
        sig_id = int(parse_filename[1][:2])
        #print("%d, %d" %(person_id, sig_id))
        image = Image.open(filename)
        arr_image = np.asarray(image, dtype=np.uint8).copy()
        dataset[person_id-1][sig_id-1] = arr_image
        
    return dataset


def _getNumPersonSig(path):
    
    path_len = len(path)
    num_person = 0
    num_sig = 0
    for filename in (glob.glob(path + '/*_rs.PNG') + glob.glob(path + '/*_rs.png')):
        without_extension = os.path.splitext(filename)[0] #splittext = 확장자(.png or .jpeg ...)를 기준으로 두 부분으로 나눔
        parse_filename = without_extension[path_len:].split('_')
        pre_len = len(parse_filename[0])
        
        person_id = int(parse_filename[0][pre_len-3:])
        sig_id = int(parse_filename[1][:2])
        
        num_person = max(person_id, num_person)
        num_sig = max(sig_id, num_sig)
        
    return num_person, num_sig


def binarize_array(arr, threshold=200):
    tmp_arr = arr.copy()
    tmp_arr[tmp_arr < threshold] = 0
    tmp_arr[tmp_arr >= threshold] = 255
    return tmp_arr


def print_arr(arr):
    plt.imshow(arr)
    plt.show()