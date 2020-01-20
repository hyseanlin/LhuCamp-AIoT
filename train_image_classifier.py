import os
import glob
import argparse
import datetime
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Dense, Activation, MaxPool2D, Conv2D, Flatten, Dropout, Input, BatchNormalization, Add
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.callbacks import Callback, CSVLogger

# Keras 內建模型
# https://keras.io/applications
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.mobilenetv2 import MobileNetV2


class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('loss'))

def custom_model(input_shape, class_count):
    def conv_block(x, filters):
        x = BatchNormalization()(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        shortcut = x
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = Add()([x, shortcut])
        x = MaxPool2D((2, 2), strides=(2, 2))(x)
        return x

    input_tensor = Input(shape=input_shape)

    x = conv_block(input_tensor, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 512)

    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)

    output_layer = Dense(class_count, activation='softmax')(x)

    inputs = [input_tensor]
    model = Model(inputs, output_layer)

    return model

def choose_model(model_type, input_shape, class_count):
    if model_type == 'VGG16':
        input_tensor = Input(shape=input_shape)
        model = VGG16(
            input_shape=input_shape,
            classes=class_count,
            weights=None,
            input_tensor=input_tensor,
        )
    elif model_type == 'VGG19':
        input_tensor = Input(shape=input_shape)
        model = VGG19(
            input_shape=input_shape,
            classes=class_count,
            weights=None,
            input_tensor=input_tensor,
        )
    elif model_type == 'ResNet50':
        input_tensor = Input(shape=input_shape)
        model = ResNet50(
            input_shape=input_shape,
            classes=class_count,
            weights=None,
            input_tensor=input_tensor,
        )
    elif model_type == 'DenseNet121':
        input_tensor = Input(shape=input_shape)
        model = DenseNet121(
            input_shape=input_shape,
            classes=class_count,
            weights=None,
            input_tensor=input_tensor,
        )
    elif model_type == 'MobileNetV2':
        input_tensor = Input(shape=input_shape)
        model = MobileNetV2(
            input_shape=input_shape,
            classes=class_count,
            weights=None,
            input_tensor=input_tensor,
        )
    elif model_type == 'custom':
        model = custom_model(input_shape, class_count)
    else:
        model = custom_model(input_shape, class_count)
    return model

def main():

    # 定義程式參數
    arg_parser = argparse.ArgumentParser(description='模型訓練範例')
    arg_parser.add_argument(
        '--model-file',
        # required=True,
        help='模型描述檔',
        default='model.json',
    )
    arg_parser.add_argument(
        '--weights-file',
        # required=True,
        help='模型參數檔案',
        default='model.h5',
    )
    arg_parser.add_argument(
        '--train-dir',
        # required=True,
        help='訓練資料目錄',
        default='train_data',
    )
    arg_parser.add_argument(
        '--test-dir',
        # required=True,
        help='測試資料目錄',
        default='test_data',
    )
    arg_parser.add_argument(
        '--model-type',
        choices=('VGG16', 'VGG19', 'ResNet50', 'DenseNet121', 'MobileNetV2', 'custom'),
        default='custom',
        help='選擇模型類別',
    )
    arg_parser.add_argument(
        '--epochs',
        type=int,
        default=32,
        help='訓練回合數',
    )
    arg_parser.add_argument(
        '--input-width',
        type=int,
        default=48,
        help='模型輸入寬度',
    )
    arg_parser.add_argument(
        '--input-height',
        type=int,
        default=48,
        help='模型輸入高度',
    )
    arg_parser.add_argument(
        '--load-weights',
        action='store_true',
        help='從 --weights-file 指定的檔案載入模型參數',
    )
    arg_parser.add_argument(
        '--num-gpu',
        type=int,
        default=1,
        help='使用的GPU數量，預設為1',
    )
    args = arg_parser.parse_args()

    # 資料參數
    input_height = args.input_height    # 影像高度
    input_width = args.input_width      # 影像寬度
    input_channel = 3                   # 影像色彩數
    input_shape = (input_height, input_width, input_channel)

    # 搜尋訓練資料目錄下的所有子目錄，並以子目路名稱作為訓練標記
    train_data_map = {}
    train_data_count = 0
    labels = []
    for subdir in os.listdir(args.train_dir):
        labels.append(subdir)
        match_train_files = os.path.join(args.train_dir, subdir, '*.jpg')
        train_data_map[subdir] = glob.glob(match_train_files)
        train_data_count = train_data_count + len(train_data_map[subdir])
    # 獲取類別數量
    class_count = len(train_data_map)

    # 將所有類別名稱依序輸入至 classes.txt 檔
    class_file = open('classes.txt', 'w')
    for label in labels:
        class_file.write('%s\n' %(label))
    class_file.close()

    # 選取模型
    model = choose_model(args.model_type, input_shape, class_count)

    # 如果有 GPU，轉換模型以獲取 GPU 加速支援
    if args.num_gpu > 1:
        model = multi_gpu_model(model, gpus=args.num_gpu)

    # 選定最佳化演算法、損失函數以利模型編譯
    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['acc'],
    )

    history = AccuracyHistory()
    csv_logger = CSVLogger('training.csv')
    cbks = [history, csv_logger]

    # 搜尋測試資料目錄下的所有的待測試檔案
    match_test = os.path.join(args.test_dir, '*.jpg')
    paths_test = glob.glob(match_test)
    n_test = len(paths_test)

    # 初始化資料集矩陣
    trainset = np.zeros(
        shape=(train_data_count, input_height, input_width, input_channel),
        dtype='float32',
    )
    trainlabels = np.zeros(
        shape=(train_data_count, class_count),
        dtype='float32',
    )
    testset = np.zeros(
        shape=(n_test, input_height, input_width, input_channel),
        dtype='float32',
    )

    # 讀取圖片到各資料集
    paths_train = []
    for train_files in train_data_map.values():
        paths_train = paths_train + train_files

    for ind, path in enumerate(paths_train):
        image = cv2.imread(path)
        resized_image = cv2.resize(image, (input_width, input_height))
        trainset[ind] = resized_image

    for ind, path in enumerate(paths_test):
        image = cv2.imread(path)
        resized_image = cv2.resize(image, (input_width, input_height))
        testset[ind] = resized_image

    # 設定訓練集的標記
    begin_ind = end_ind = 0
    for ind, label in enumerate(labels):
        end_ind = end_ind + len(train_data_map[label])
        trainlabels[begin_ind:end_ind, ind] = 1.0
        begin_ind = end_ind

    # 正規化影像圖素數值
    trainset = trainset / 255.0
    testset = testset / 255.0

    # 載入模型參數
    if args.load_weights:
        model.load_weights(args.weights_file)

    # 開始訓練模型
    if args.epochs > 0:
        time_ticker = datetime.datetime.now()
        model.fit(trainset, trainlabels, epochs=args.epochs, validation_split=0.2, batch_size=64, callbacks=cbks,)
        time_ticker = datetime.datetime.now() - time_ticker
        print('model.fit() takes %s seconds' %(time_ticker.total_seconds()))

    # 儲存模型架構及權重參數
    model_desc = model.to_json()
    with open(args.model_file, 'w') as file_model:
        file_model.write(model_desc)
    model.save_weights(args.weights_file)

    # 針對測試資料進行預測
    if testset.shape[0] != 0:
        result_onehot = model.predict(testset)
        result_sparse = np.argmax(result_onehot, axis=1)
        # 印出預測結果
        print('檔名\t預測類別')
        for path, label_id in zip(paths_test, result_sparse):
            filename = os.path.basename(path)
            label_name = labels[label_id]
            print('%s\t%s' % (filename, label_name))
    else:
        exit()  # 強制結束 python 程式(因為測試資料不全)

if __name__ == '__main__':
    main()
