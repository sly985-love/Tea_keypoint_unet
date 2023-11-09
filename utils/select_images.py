# -*- coding:utf-8 -*-

"""
# @Time       : 2022/4/19 21:09
# @Author     : GraceKafuu
# @Email      : 
# @File       : select_images.py
# @Software   : PyCharm

Description:
"""

import os
import shutil


if __name__ == '__main__':
    base_path = "data/crops_v2"
    # img_path = base_path + "/tea3"
    # json_path = base_path + "/tea3_annotations"
    img_path = base_path + "/images"
    json_path = base_path + "/jsons"

    save_path = base_path + "/tea3_"
    os.makedirs(save_path, exist_ok=True)

    jsons_list = os.listdir(json_path)
    for js in jsons_list:
        imgPath = img_path + "/{}".format(js.replace(".json", ".jpg"))
        savePath = save_path + "/{}".format(js.replace(".json", ".jpg"))
        shutil.copy(imgPath, savePath)


