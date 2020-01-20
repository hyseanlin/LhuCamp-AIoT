import cv2
import os
import argparse

arg_parser = argparse.ArgumentParser(description='獲取訓練與測試資料。')
arg_parser.add_argument(
    '--data-type',
    help='資料的存放目錄',
    default='test_data',
)
arg_parser.add_argument(
    '--label-name',
    help='標記名稱',
    default='',
)
arg_parser.add_argument(
    '--frame-begin',
    help='畫面檔案的起始編號',
    type=int,
    default=0,
)
arg_parser.add_argument(
    '--frame-end',
    help='畫面檔案的終止編號',
    type=int,
    default=1000,
)
arg_parser.add_argument(
    '--camera-source',
    help='攝影機編號',
    type=int,
    default=0,
)

# 參數設定
args = arg_parser.parse_args()
data_type = args.data_type        # 可設定 train_data 或 test_data 分別代表訓練及測試資料集
label_name = args.label_name      # 資料標記
frame_begin = args.frame_begin    # 起始畫面編號
frame_end = args.frame_end  # 終止畫面編號
camera_src = args.camera_source   # 選定攝影機

# 開啟相機
cap = cv2.VideoCapture(camera_src + cv2.CAP_DSHOW)
if cap.isOpened() is not True:
    print('Camera is not opened.')
    exit()  # 強制結束 python 程式 (因為開啟相機失敗)

window_title = 'Camera: {}'.format(camera_src)
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

# 建立輸出資料的資料夾(如需要的話)
output_folder = os.path.join(data_type, label_name)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

is_recording = False
ret = True
while ret:
    ret, frame = cap.read()
    cv2.imshow(window_title, cv2.flip(frame, 1))
    if is_recording:
        image_pathname = os.path.join(output_folder, '{}_{:05d}.jpg'.format(label_name, frame_begin))
        cv2.imwrite(image_pathname, frame)
        frame_begin = frame_begin + 1
        print(image_pathname)

    key = cv2.waitKey(5) & 0xFF
    if key == 27: # 判斷輸入鍵是否為 ESC
        print('ESC is pressed by user.')
        break
    elif frame_begin >= frame_end:
        print('Max frame count reached.')
        break
    elif key == ord('s'): # 判斷輸入鍵是否為 s
        is_recording = not is_recording

# 終止影像裝置
cap.release()               # 釋放 video capture 資源
cv2.destroyAllWindows()     # 關閉所有視窗