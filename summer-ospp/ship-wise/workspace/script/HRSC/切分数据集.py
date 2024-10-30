import os
import random
from shutil import copyfile


def split_dataset(img_dir, annotation_dir,
                  train_dir, val_dir, test_dir,
                  train_set_file_path, val_set_file_path, test_set_file_path,
                  train_ratio=0.8, val_ratio=0.1):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    img_files = os.listdir(img_dir)
    annotation_files = os.listdir(annotation_dir)

    # 过滤无标注的图片
    img_files = [img_file for img_file in img_files if img_file.replace('.bmp', '.txt') in annotation_files]

    random.shuffle(img_files)

    num_files = len(img_files)
    num_train_files = int(num_files * train_ratio)
    num_val_files = int(num_files * val_ratio)
    num_test_files = num_files - num_train_files - num_val_files

    # 将划分的图片路径写入文件
    train_set = img_files[:num_train_files]
    val_set = img_files[num_train_files:num_train_files + num_val_files]
    test_set = img_files[num_train_files + num_val_files:]
    for (set_file_path, set_dir, set_files) in zip([train_set_file_path, val_set_file_path, test_set_file_path],
                                                   [train_dir, val_dir, test_dir],
                                                   [train_set, val_set, test_set]):
        with open(set_file_path, 'w') as file:
            for set_file in set_files:
                file.write(os.path.join(set_dir, set_file) + '\n')

    for i, img_file in enumerate(img_files):
        annotation_file = img_file.replace('.bmp', '.txt')
        if i < num_train_files:
            copyfile(os.path.join(img_dir, img_file), os.path.join(train_dir, img_file))
            copyfile(os.path.join(annotation_dir, annotation_file), os.path.join(train_dir, annotation_file))
        elif i < num_train_files + num_val_files:
            copyfile(os.path.join(img_dir, img_file), os.path.join(val_dir, img_file))
            copyfile(os.path.join(annotation_dir, annotation_file), os.path.join(val_dir, annotation_file))
        else:
            copyfile(os.path.join(img_dir, img_file), os.path.join(test_dir, img_file))
            copyfile(os.path.join(annotation_dir, annotation_file), os.path.join(test_dir, annotation_file))


def main():
    configs = {
        'img_dir': r"H:\Library\Datasets\HRSC\HRSC2016_dataset\HRSC2016\FullDataSet\AllImages",
        'annotation_dir': r"H:\Library\Datasets\HRSC\HRSC2016_dataset\HRSC2016\FullDataSet-YOLO\Annotations",
        'train_dir': r"H:\Library\Datasets\HRSC\HRSC2016_dataset\HRSC2016\FullDataSet-YOLO-Split\train",
        'val_dir': r"H:\Library\Datasets\HRSC\HRSC2016_dataset\HRSC2016\FullDataSet-YOLO-Split\validation",
        'test_dir': r"H:\Library\Datasets\HRSC\HRSC2016_dataset\HRSC2016\FullDataSet-YOLO-Split\test",
        'train_set_file_path': r"H:\Library\Datasets\HRSC\HRSC2016_dataset\HRSC2016\FullDataSet-YOLO-Split\train.txt",
        'val_set_file_path': r"H:\Library\Datasets\HRSC\HRSC2016_dataset\HRSC2016\FullDataSet-YOLO-Split\val.txt",
        'test_set_file_path': r"H:\Library\Datasets\HRSC\HRSC2016_dataset\HRSC2016\FullDataSet-YOLO-Split\test.txt",
    }
    split_dataset(**configs)


if __name__ == '__main__':
    main()
