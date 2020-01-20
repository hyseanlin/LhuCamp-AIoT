#!/usr/bin/env python3
import argparse
import time
import cv2
import os
import numpy as np
from keras.models import model_from_json

# Command-line
# python keras_video.py

def main():
    # 設定程式參數
    arg_parser = argparse.ArgumentParser(description='執行 Keras 模型，辨識影片檔或攝影機影像。')
    arg_parser.add_argument(
        '--model-file',
        help='模型架構檔',
        default='model.json',
    )
    arg_parser.add_argument(
        '--weights-file',
        default='model.h5',
        help='模型參數檔',
    )
    arg_parser.add_argument(
        '--input-width',
        type=int,
        default=48,
        help='模型輸入影像寬度',
    )
    arg_parser.add_argument(
        '--input-height',
        type=int,
        default=48,
        help='模型輸入影像高度',
    )

    # 解讀程式參數
    args = arg_parser.parse_args()
    assert args.input_width > 0 and args.input_height > 0

    # 載入模型架構及權重參數
    with open(args.model_file, 'r') as file_model:
        model_desc = file_model.read()
        model = model_from_json(model_desc)
    model.load_weights(args.weights_file)
    # 載入類別名稱
    with open('classes.txt') as f:
        labels = f.read().splitlines()
    f.close()

    # 參數設定
    camera_src = 0

    cap = cv2.VideoCapture(camera_src)
    if cap.isOpened() is not True:
        print('Camera is not opened.')
        exit()  # 強制結束 python 程式 (因為開啟相機失敗)

    # 主迴圈
    try:
        prev_timestamp = time.time()

        while True:
            ret, orig_image = cap.read()
            curr_time = time.localtime()

            # 檢查串流是否結束
            if ret is None or orig_image is None:
                break

            # 縮放爲模型輸入的維度、調整數字範圍爲 0～1 之間的數值
            resized_image = cv2.resize(
                orig_image,
                (args.input_width, args.input_height),
            ).astype(np.float32)
            # 正規化影像圖素數值
            normalized_image = resized_image / 255.0

            # 執行預測
            batch = normalized_image.reshape(1, args.input_height, args.input_width, 3)
            result_onehot = model.predict(batch)
            scores = result_onehot[0]
            class_id = np.argmax(result_onehot, axis=1)[0]

            class_str = labels[class_id]

            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # 判斷輸入鍵是否為 ESC
                print('ESC is pressed by user.')
                break

            # 計算執行時間
            recent_timestamp = time.time()
            period = recent_timestamp - prev_timestamp
            prev_timestamp = recent_timestamp

            print('時間：%02d:%02d:%02d ' % (curr_time.tm_hour, curr_time.tm_min, curr_time.tm_sec))
            output_str = '輸出：'
            for score in scores:
                output_str = '%s%.2f ' % (output_str, score)
            print(output_str)
            print('類別：%s' % class_str)
            print('費時：%f' % period)
            print()

            # 顯示圖片
            cv2.putText(orig_image, class_str, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2 , (0,0,255), 1)
            cv2.imshow('', orig_image)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        print('使用者中斷')

    # 終止影像裝置
    cap.release()               # 釋放 video capture 資源
    cv2.destroyAllWindows()     # 關閉所有視窗

if __name__ == '__main__':
    main()
