import os.path
import shutil
import tkinter
from tkinter import font as tkFont


import cv2.cv2
import  dlib


#dlib 正向的人脸检测器
import numpy as np

detector = dlib.get_frontal_face_detector()

class Get_Face:
    def __init__(self):

        # 人脸文件夹计数器
        self.personface_file_num_cnt = 0
        # 单文件夹里人脸照片计数器
        self.personface_photo_num_cnt = 0



        # 人脸识别的GUI
        self.gui = tkinter.Tk()
        self.gui.title("人脸识别")
        self.gui.geometry("1400x600")

        # 左边的gui部分
        self.left_camera_frame = tkinter.Frame(self.gui)
        self.label = tkinter.Label(self.gui)
        self.label.pack(side=tkinter.LEFT)
        self.left_camera_frame.pack()

        # 右边的gui部分
        self.right_camera_frame = tkinter.Frame(self.gui)
        # 文件夹
        self.label_cnt_face_file = tkinter.Label(self.right_camera_frame,text = str(self.personface_file_num_cnt) )
        # 照片数量
        self.label_cnt_face_in_datafile = tkinter.Label(self.right_camera_frame, text = str(self.personface_photo_num_cnt))
        # 个人姓名输入
        self.input_name  = tkinter.Entry(self.right_camera_frame)
        self.input_name_char = ""

        self.label_warning = tkinter.Label(self.right_camera_frame)
        self.lable_face_cnt = tkinter.Label(self.right_camera_frame, text="脸的位置适合录入数据：")
        self.log_all = tkinter.Label(self.right_camera_frame)


        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')
        self.font_warning = tkFont.Font(family='Helvetica', size=15, weight='bold')


        # 存放文件的路径
        self.path_file_from_camera = "data/data_from_camera"
        self.current_face_dir = ""
        self.font = cv2.cv2.FONT_ITALIC

        # 脸部位置判断和感兴趣区域
        self.current_frame = np.ndarray
        self.face_ROI_image = np.ndarray
        self.face_ROI_width_start = 0
        self.face_ROI_height_start = 0
        self.face_ROI_width = 0
        self.face_ROI_height = 0
        self.ww = 0
        self.hh = 0


        self.out_of_range_flag = False
        self.face_folder_created_flag = False


        # 获取视频流
        self.cap = cv2.cv2.VideoCapture(0)




    # 删除保存的人脸数据文件夹
    def gui_clear_datafile(self):
        # 删除保存人脸的文件夹   person_x
        datafile_path = os.listdir(self.path_file_from_camera)
        for i in range(len(datafile_path)):
            shutil.rmtree(self.path_file_from_camera + datafile_path[i])
        if os.path.isfile("data/features_all.csv"):
            os.remove("data/features_all.csv")
        self.label_cnt_face_in_datafile['text'] = "0"
        self.personface_file_num_cnt = 0
        self.personface_photo_num_cnt = 0
        self.log_all['text'] = "人脸照片与数据分析文件已删除"

    # ui界面获取名称
    def gui_get_name(self):
        self.input_name_char = self.input_name.get()
        self.create_face_folder()
        self.label_cnt_face_in_datafile['text'] = str(self.personface_photo_num_cnt)

    #




    # 建一个文件夹，存放录入的人脸照片和csv文件
    def newfile_mkdir(self):
        # 新建文件夹
        if os.path.isdir(self.path_file_from_camera):
            pass
        else:
            os.mkdir(self.path_file_from_camera)
    # 人脸数据存放的位置文件按序号递增
    def personface_num_cnt(self):
        if os.listdir("data/data_from_camera/"):
            # 获取已经录入的最后一个人脸序号
            personface_list = os.listdir("data/data_from_camera/")
            personface_num_list = []

    def create_face_folder(self):
        pass



