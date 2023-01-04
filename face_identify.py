

# 利用 OT 人脸追踪, 进行人脸实时识别
# 检测 -> 识别人脸, 新人脸出现 -> 不需要识别, 而是利用质心追踪来判断识别结果
# 人脸进行再识别需要花费大量时间, 这里用 OT 做跟踪
import tkinter

from PIL import ImageTk, Image
from tkinter import font as tkFont
import dlib
import numpy as np
import cv2
import cv2.cv2
import os
import pandas as pd
import time
import logging

# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型, 提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC

        # 计数器
        self.get_csvdata = 0
        # 正确识别的次数
        self.true_times = 0
        # 不正确识别的次数
        self.flase_times =0


        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # cnt for frame
        self.frame_cnt = 0

        # 用来存放所有录入人脸特征的数组
        self.face_features_known_list = []
        # 存储录入人脸信息
        self.face_message_known_list = []

        # 用来存储上一帧和当前帧 ROI 的质心坐标
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # 用来存储上一帧和当前帧检测出目标的名字
        self.last_frame_face_message_list = []
        self.current_frame_face_message_list = []

        # 上一帧和当前帧中人脸数的计数器
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # 用来存放进行识别时候对比的欧氏距离
        self.current_frame_face_X_e_distance_list = []

        # 存储当前摄像头中捕获到的所有人脸的坐标名字
        self.current_frame_face_position_list = []
        # 存储当前摄像头中捕获到的人脸特征
        self.current_frame_face_feature_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        # 控制再识别的后续帧数 / Reclassify after 'reclassify_interval' frames
        # 如果识别出 "unknown" 的脸, 将在 reclassify_interval_cnt 计数到 reclassify_interval 后, 对于人脸进行重新识别
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 5

        # 特殊作用
        self.img_rd = np.ndarray

        '''人脸识别gui'''
        self.gui = tkinter.Tk()
        self.gui.title("人脸识别")
        self.gui.geometry("1400x600")

        # 格式设置
        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')
        self.font_warning = tkFont.Font(family='Helvetica', size=15, weight='bold')

        '''左边的gui部分'''
        self.left_camera_frame = tkinter.Frame(self.gui)
        self.label = tkinter.Label(self.gui)
        self.label.pack(side=tkinter.LEFT)
        self.left_camera_frame.pack()



        '''右上边的gui部分'''
        self.right_up_camera_frame = tkinter.Frame(self.gui)
        self.log_first = tkinter.Label(self.right_up_camera_frame, fg='blue')
        self.log_first['text'] = '正在识别，请稍后...'
        self.fps_frame = tkinter.Label(self.right_up_camera_frame, fg="blue")
        self.fps_frame['text'] ='0'
        self.true_identify = tkinter.Label(self.right_up_camera_frame, fg='blue')
        self.true_identify['text']='0%'

        self.right_up_camera_frame.pack()



        self.photo_path = "data/unknown.png"
        self.label_photo = tkinter.Label(self.gui)
        self.user_photo = np.ndarray


        '''右下边的gui部分'''
        self.right_down_camera_frame =tkinter.Frame(self.gui)
        self.user_name =tkinter.Label(self.right_down_camera_frame,text='')
        self.user_num =tkinter.Label(self.right_down_camera_frame,text='')
        self.right_down_camera_frame.pack()


        self.cap = cv2.cv2.VideoCapture(0)


    def gui_info(self):
        tkinter.Label(self.right_up_camera_frame,
                      text="人脸识别扫描",
                      font=self.font_title).grid(row=0, column=0, columnspan=3, sticky=tkinter.W, padx=5, pady=30)

        self.log_first.grid(row=1, column=0, columnspan=15, sticky=tkinter.W, padx=5, pady=10)

        tkinter.Label(self.right_up_camera_frame,
                      text="单次识别的fps: ").grid(row=2, column=0, columnspan=2, sticky=tkinter.W, padx=5, pady=10)
        self.fps_frame.grid(row=2, column=2, columnspan=3, sticky=tkinter.W, padx=5, pady=10)

        tkinter.Label(self.right_up_camera_frame,
                      text="识别的准确度: ").grid(row=3, column=0, columnspan=2, sticky=tkinter.W, padx=5, pady=10)
        self.true_identify.grid(row=3, column=2, columnspan=3, sticky=tkinter.W, padx=5, pady=10)

        '''图片设置'''
        img = Image.open(self.photo_path)

        img = img.resize((200, 200))
        img_Photoimage = ImageTk.PhotoImage(image = img)
        self.label_photo.img_tk = img_Photoimage
        self.label_photo.configure(image=img_Photoimage)
        self.label_photo.pack()

        tkinter.Label(self.right_down_camera_frame,
                      text="姓名: ").grid(row=0, column=0, columnspan=2, sticky=tkinter.W, padx=5, pady=10)
        self.user_name.grid(row=0, column=2, columnspan=3, sticky=tkinter.W, padx=5, pady=10)

        tkinter.Label(self.right_down_camera_frame,
                      text="工号: ").grid(row=1, column=0, columnspan=2, sticky=tkinter.W,padx=5, pady=10)
        self.user_num.grid(row=1, column=2, columnspan=3, sticky=tkinter.W, padx=5, pady=10)

    # 从 "features_all.csv" 读取录入人脸特征
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                # 存储人物信息
                self.face_message_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("数据集中的人脸照片数： %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("没有找到文件夹 features_all.csv!")
            logging.warning("请先执行 'get_faces.py' "
                            "and 'data_dispose.py'在你执行'face_identify.py'之前")
            return 0

    def update_fps(self):
        now = time.time()
        # 每秒刷新 fps
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
            self.fps_frame['text'] = str(self.fps.__round__(2))
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    # 计算两个128D向量间的欧式距离
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # 使用质心追踪来识别人脸
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            # 对于当前帧中的人脸1, 和上一帧中的 人脸1/2/3/4/.. 进行欧氏距离计算
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_message_list[i] = self.last_frame_face_message_list[last_frame_num]

    # 添加说明文字
    def draw_note(self, img_rd):

        for i in range(len(self.current_frame_face_message_list)):
            img_rd = cv2.cv2.putText(img_rd, "Face" + str(i + 1),
                                     tuple([int(self.current_frame_face_centroid_list[i][0]),
                                     int(self.current_frame_face_centroid_list[i][1])]),
                                     self.font,
                                     0.8, (255, 190, 0),
                                     1,
                                     cv2.cv2.LINE_AA)

    # 获取视频流的当前帧
    def get_frame(self):
        try:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                return ret, cv2.cvtColor(frame, cv2.cv2.COLOR_BGR2RGB)
        except:
            print("没有视频流输入!!!")

    # 处理获取的视频流, 进行人脸识别
    def process(self):
        if self.get_csvdata == 0:

            # 1. 读取存放所有人脸特征的 csv
            self.get_csvdata = self.get_face_database()
        else:
            flag, self.img_rd = self.get_frame()
            self.frame_cnt += 1

            # 2. 检测人脸
            faces = detector(self.img_rd, 0)

            # 3. 更新人脸计数器
            self.last_frame_face_cnt = self.current_frame_face_cnt
            self.current_frame_face_cnt = len(faces)

            # 4. 更新上一帧中的人脸列表
            self.last_frame_face_message_list = self.current_frame_face_message_list[:]

            # 5. 更新上一帧和当前帧的质心列表
            self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
            self.current_frame_face_centroid_list = []

            # 6.1 如果当前帧和上一帧人脸数没有变化
            if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                    self.reclassify_interval_cnt != self.reclassify_interval):
                logging.debug("当前帧和上一帧相比没有发生人脸数变化 !!!")

                self.current_frame_face_position_list = []

                if "unknown" in self.current_frame_face_message_list:
                    logging.debug("有未知人脸, 开始进行 reclassify_interval_cnt 计数")
                    self.reclassify_interval_cnt += 1

                if self.current_frame_face_cnt != 0:
                    for k, d in enumerate(faces):
                        self.current_frame_face_position_list.append(tuple(
                            [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                        self.current_frame_face_centroid_list.append(
                            [int(faces[k].left() + faces[k].right()) / 2,
                             int(faces[k].top() + faces[k].bottom()) / 2])

                        self.img_rd = cv2.cv2.rectangle(self.img_rd,
                                               tuple([d.left(), d.top()]),
                                               tuple([d.right(), d.bottom()]),
                                               (255, 255, 255), 2)

                # 如果当前帧中有多个人脸, 使用质心追踪 / Multi-faces in current frame, use centroid-tracker to track
                if self.current_frame_face_cnt != 1:
                    self.centroid_tracker()

                for i in range(self.current_frame_face_cnt):
                    # 6.2 在感兴趣区域标注名字
                    self.img_rd = cv2.cv2.putText(self.img_rd, self.current_frame_face_message_list[i].split('_',1)[0],
                                                  self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1,
                                                  cv2.cv2.LINE_AA)


                self.draw_note(self.img_rd)

            # 6.2 如果当前帧和上一帧人脸数发生变化
            else:
                logging.debug("scene 2: 当前帧和上一帧相比人脸数发生变化 ")
                self.current_frame_face_position_list = []
                self.current_frame_face_X_e_distance_list = []
                self.current_frame_face_feature_list = []
                self.reclassify_interval_cnt = 0

                # 6.2.1 人脸数减少  1->0 2->1
                if self.current_frame_face_cnt == 0:
                    logging.debug("人脸消失, 当前帧中没有人脸 !!!")
                    self.log_first['text'] = '人脸消失, 当前帧中没有人脸 !!!'

                    self.user_name['text']  = ''
                    self.user_num['text']  = ''
                    self.photo_path = "data/unknown.png"
                    img = Image.open(self.photo_path)

                    img = img.resize((200, 200))
                    img_Photoimage = ImageTk.PhotoImage(image=img)
                    self.label_photo.img_tk = img_Photoimage
                    self.label_photo.configure(image=img_Photoimage)


                    # 清空列表中的人脸数据
                    self.current_frame_face_message_list = []
                # 6.2.2 人脸数增加  0->1, 0->2, ..., 1->2, ...
                else:
                    logging.debug("scene 2.2 出现人脸, 进行人脸识别")
                    self.log_first['text'] = '出现人脸, 进行人脸识别!!!'
                    self.current_frame_face_message_list = []
                    for i in range(len(faces)):
                        shape = predictor(self.img_rd, faces[i])
                        self.current_frame_face_feature_list.append(
                            face_reco_model.compute_face_descriptor(self.img_rd, shape))
                        self.current_frame_face_message_list.append("unknown")

                    # 6.2.2.1 遍历捕获到的图像中所有的人脸
                    for k in range(len(faces)):
                        logging.debug("  For face %d in current frame:", k + 1)
                        self.current_frame_face_centroid_list.append(
                            [int(faces[k].left() + faces[k].right()) / 2,
                             int(faces[k].top() + faces[k].bottom()) / 2])

                        self.current_frame_face_X_e_distance_list = []

                        # 6.2.2.2 每个捕获人脸的名字坐标
                        self.current_frame_face_position_list.append(tuple(
                            [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                        # 6.2.2.3 对于某张人脸, 遍历所有存储的人脸特征
                        for i in range(len(self.face_features_known_list)):
                            # 如果 q 数据不为空
                            if str(self.face_features_known_list[i][0]) != '0.0':
                                e_distance_tmp = self.return_euclidean_distance(
                                    self.current_frame_face_feature_list[k],
                                    self.face_features_known_list[i])
                                logging.debug("with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                            else:
                                # 空数据 person_X
                                self.current_frame_face_X_e_distance_list.append(999999999)

                        # 6.2.2.4 寻找出最小的欧式距离匹配
                        similar_person_num = self.current_frame_face_X_e_distance_list.index(
                            min(self.current_frame_face_X_e_distance_list))

                        if min(self.current_frame_face_X_e_distance_list) < 0.4:
                            self.current_frame_face_message_list[k] = self.face_message_known_list[similar_person_num]
                            #识别出来了
                            logging.info("人脸识别结果: %s",
                                          self.face_message_known_list[similar_person_num].split('_',1)[0])
                            self.log_first['text'] = '人脸识别结果:'+self.face_message_known_list[similar_person_num].split('_',1)[0]
                            self.user_name['text']  = self.face_message_known_list[similar_person_num].split('_',1)[0]
                            self.user_num['text']  = self.face_message_known_list[similar_person_num].split('_', 1)[-1]
                            self.photo_path = "data/data_from_camera/person_1_"\
                                              +self.face_message_known_list[similar_person_num]\
                                              +"/img_face_1.jpg"
                            img = Image.open(self.photo_path)
                            img = img.resize((200, 200))
                            img_Photoimage = ImageTk.PhotoImage(image=img)
                            self.label_photo.img_tk = img_Photoimage
                            self.label_photo.configure(image=img_Photoimage)
                            self.true_times += 1
                            if self.true_times != 0 or self.flase_times != 0:
                                true_identify = self.true_times / (self.true_times + self.flase_times)
                                true_identify = true_identify * 100
                                self.true_identify['text'] =str(true_identify.__round__(2)) + "%"

                        else:
                            # 没有识别出来
                            logging.info("人脸识别结果: Unknown person")
                            self.log_first['text'] = '人脸识别结果:没有该人物'
                            self.flase_times += 1
                            if self.true_times != 0 or self.flase_times != 0:
                                true_identify = self.true_times / (self.true_times + self.flase_times)
                                true_identify = true_identify * 100
                                self.true_identify['text'] = str(true_identify.__round__(2)) + "%"
                    # 7. 生成的窗口添加说明文字
                    self.draw_note(self.img_rd)
                    # cv2.imwrite("debug/debug_" + str(self.frame_cnt) + ".png", img_rd) # Dump current frame image if needed


            self.update_fps()

            img_Image = Image.fromarray(self.img_rd)
            img_Image = img_Image.resize((750,550))
            img_PhotoImage = ImageTk.PhotoImage(img_Image)
            self.label.img_tk = img_PhotoImage
            self.label.configure(image=img_PhotoImage)


        self.gui.after(20,self.process)

    def run(self):
        self.gui_info()
        self.process()
        self.gui.mainloop()
        self.cap.release()
        cv2.cv2.destroyAllWindows()


def main():
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()





if __name__ == '__main__':
        main()
