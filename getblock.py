import cv2
import numpy as np

x = 50
y = 50
w = 16
h = 16

filename = 'forward_00072'
extname = 'jpg'
rgb = cv2.imread('{}.{}'.format(filename, extname))

red = rgb.copy()
red[:,:,0] = 0
red[:,:,1] = 0
cv2.imwrite('{}_red.{}'.format(filename, extname), red)

grn = rgb.copy()
grn[:,:,1] = 0
grn[:,:,2] = 0
cv2.imwrite('{}_blu.{}'.format(filename, extname), grn)

blu = rgb.copy()
blu[:,:,0] = 0
blu[:,:,2] = 0
cv2.imwrite('{}_grn.{}'.format(filename, extname), blu)


rgb_block = rgb[x:x+w, y:y+h]

# 彩色轉灰階
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
gray_block = cv2.cvtColor(rgb_block, cv2.COLOR_RGB2GRAY)
# 輸出灰階影像
cv2.imwrite('{}_gray.{}'.format(filename, extname), gray)
cv2.imwrite('{}_gray_blk.{}'.format(filename, extname), gray_block)
np.savetxt('{}_gray_blk.csv'.format(filename), gray_block, delimiter=",")