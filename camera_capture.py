import cv2
import os

# 參數設定
data_type = 'train_data'    # 可設定 train_data 或 test_data 分別代表訓練及測試資料集
label_name = 'pepper'       # 資料標記
frame_count = 1             # 起始畫面編號
frame_count_max = 1000      # 終止畫面編號
camera_src = 0              # 選定攝影機

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
        image_pathname = os.path.join(output_folder, '{}_{:05d}.jpg'.format(label_name, frame_count))
        cv2.imwrite(image_pathname, frame)
        frame_count = frame_count + 1
        print(image_pathname)

    key = cv2.waitKey(5) & 0xFF
    if key == 27: # 判斷輸入鍵是否為 ESC
        print('ESC is pressed by user.')
        break
    elif frame_count >= frame_count_max:
        print('Max frame count reached.')
        break
    elif key == ord('s'): # 判斷輸入鍵是否為 s
        is_recording = not is_recording

# 終止影像裝置
cap.release()               # 釋放 video capture 資源
cv2.destroyAllWindows()     # 關閉所有視窗