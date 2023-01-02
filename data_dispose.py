
import os
import dlib
import csv
import numpy as np
import logging
import cv2
import  cv2.cv2
# 要读取人脸图像文件的路径
path_images_from_camera = "data/data_from_camera/"

# dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型，提取 128D 的特征矢量
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


# 返回单张图像的 128D 特征
# 输入:   照片路径
# 输出:   128D特征
def return_128d_features(path_img):
    img_rd = cv2.cv2.imread(path_img)
    faces = detector(img_rd, 1)

    logging.info("%-40s %-20s", "检测到人脸的图像:", path_img)

    # 因为有可能截下来的人脸再去检测，检测不出来人脸了, 所以要确保是 检测到人脸的人脸图像拿去算特征
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        logging.warning("没有人脸图片")
    return face_descriptor


# 返回 person 的 128D 特征均值
# 输入:    存放个人人脸数据的文件夹
# 输出:    128D的特征均值
def return_features_mean_person(path_face_person):
    features_list_person = []
    photos_list = os.listdir(path_face_person)
    if photos_list:
        for i in range(len(photos_list)):
            # 调用 return_128d_features() 得到 128D 特征
            logging.info("%-40s %-20s", "正在读的人脸图像:", path_face_person + "/" + photos_list[i])
            features_128d = return_128d_features(path_face_person + "/" + photos_list[i])
            # 遇到没有检测出人脸的图片跳过
            if features_128d == 0:
                i += 1
            else:
                features_list_person.append(features_128d)
    else:
        logging.warning("文件夹%s内图像文件为空", path_face_person)

    # 计算 128D 特征的均值
    # person的N张图像 x 128D
    if features_list_person:
        features_mean_person = np.array(features_list_person, dtype=object).mean(axis=0)
    else:
        features_mean_person = np.zeros(128, dtype=object, order='C')
    return features_mean_person


def main():
    logging.basicConfig(level=logging.INFO)
    # 获取已录入的最后一个人脸序号
    person_list = os.listdir("data/data_from_camera/")
    person_list.sort()

    with open("data/features_all.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for person in person_list:
            # 获取人脸数据的128D均值
            logging.info("%sperson_%s", path_images_from_camera, person)
            features_mean_person = return_features_mean_person(path_images_from_camera + person)

            # 获取用户信息
            person_message = person.split('_', 2,)[-1]

            # person_name = person_message.split('_',1)[0]
            # person_num = person_message.split('_',1)[-1]
            # print(person_num)
            # print("名字：")
            # print(person_name)
            features_mean_person = np.insert(features_mean_person, 0, person_message,   axis=0)
            # 数据会是129D 用户信息,人脸128D数据
            writer.writerow(features_mean_person)
            logging.info('\n')
        logging.info("所有录入人脸数据存入: data/features_all.csv")


if __name__ == '__main__':
    main()

