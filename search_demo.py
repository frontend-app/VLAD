#!/usr/bin/env python
#-*- coding:utf-8 -*-
import cv2
from VLADlib import Descriptors, VLAD
import VLADtoproto as ckvlad
import feature_extract

vlad_feature_path = '../datafolder/test2/descriptors_dict.vlad'
vlad_des_dict = ckvlad.load_VLAD_from_proto(vlad_feature_path)

km_path = '../datafolder/test2'
est = ckvlad.read_kmean_result(km_path)

image = cv2.imread('../datafolder/demo.jpg')

surf = feature_extract.create_detector('surf')
kp, des = feature_extract.detect( surf, image )

vlad_des = VLAD.VLAD( des, est )


print( vlad_des )
