import logging
import os.path
import shutil
import tkinter
from tkinter import font as tkFont
import numpy as np
from PIL import ImageTk, Image



import cv2.cv2
import  dlib


#dlib 正向的人脸检测器

detector = dlib.get_frontal_face_detector()

class Get_Face:
    def __init__(self):

        # 人脸文件夹计数器
        self.personface_file_num_cnt = 0
        # 单文件夹里人脸照片计数器
        self.personface_photo_num_cnt = 0
        # 标记当前帧 也没有人脸
        self.current_frame_face = 0



        # 人脸识别的GUI
        self.gui = tkinter.Tk()
        self.gui.title("人脸识别")
        self.gui.geometry("1400x600")

        '''左边的gui部分'''
        self.left_camera_frame = tkinter.Frame(self.gui)
        self.label = tkinter.Label(self.gui)
        self.label.pack(side=tkinter.LEFT)
        self.left_camera_frame.pack()

        '''右边的gui部分'''
        self.right_camera_frame = tkinter.Frame(self.gui)

        # 文件夹数量
        self.label_cnt_face_file = tkinter.Label(self.right_camera_frame,text = str(self.personface_file_num_cnt) )

        # 照片数量
        self.label_cnt_face_in_datafile = tkinter.Label(self.right_camera_frame, text = str(self.personface_photo_num_cnt))

        # 个人姓名输入
        self.input_name  = tkinter.Entry(self.right_camera_frame)
        self.input_name_char = ""
        # 个人工号输入
        self.input_num  = tkinter.Entry(self.right_camera_frame)
        self.input_num_char = ""
        # 警告和提示
        self.label_warning = tkinter.Label(self.right_camera_frame)
        self.label_face_cnt = tkinter.Label(self.right_camera_frame, text="脸的位置适合录入数据：")
        self.log_all = tkinter.Label(self.right_camera_frame)

        # 格式设置
        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')
        self.font_warning = tkFont.Font(family='Helvetica', size=15, weight='bold')

        # 数量提示
        self.label_user_file_num = tkinter.Label(self.right_camera_frame, text="用户已创文件夹的数目:")
        self.label_user_filepthoto_num =tkinter.Label(self.right_camera_frame, text="文件夹已存的人脸数目:")


        '''存放文件的路径'''
        self.path_file_from_camera = "data/data_from_camera/"
        self.current_face_dir = ""
        self.current_face_new_dir = ""
        self.font = cv2.cv2.FONT_ITALIC

        '''脸部位置判断和感兴趣区域'''
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
        self.label_cnt_face_file['text'] = "0"
        self.personface_file_num_cnt = 0
        self.label_cnt_face_in_datafile['text'] = "0"
        self.personface_photo_num_cnt = 0
        self.label_user_filepthoto_num['text'] = "文件夹已保存的照片数量："
        self.log_all['text'] = "人脸照片与数据分析文件已删除"

    # ui界面获取人物名称
    def gui_get_message(self):
        self.input_name_char = self.input_name.get()
        self.input_num_char = self.input_num.get()
        if len(self.input_name_char) == 0 or len(self.input_num_char) == 0:
            self.log_all['text'] = "请确保个人信息已经输入！"
        else:
            self.create_face_folder()
            self.label_cnt_face_file['text'] = str(self.personface_file_num_cnt)
            self.label_user_file_num['text'] = "用户" + self.input_name_char + "已创文件夹的数目:"



    # gui布局
    def gui_info(self):
        # 标题
        tkinter.Label(self.right_camera_frame,
                      text = "人脸数据获取",
                      font = self.font_title).grid(row=0, column=0, columnspan=3, sticky=tkinter.W, padx=2, pady=20)

        self.label_user_file_num.grid(row=1, column=0, columnspan=3, sticky=tkinter.W, padx=5, pady=2)
        self.label_cnt_face_file.grid(row=1, column=3, columnspan=2, sticky=tkinter.W, padx=5, pady=2)

        self.label_user_filepthoto_num.grid(row=2, column=0, columnspan=3, sticky=tkinter.W, padx=5, pady=2)
        self.label_cnt_face_in_datafile.grid(row=2, column=3, columnspan=2, sticky=tkinter.W, padx=5, pady=2)


        self.label_warning.grid(row=4, column=0, columnspan=3, sticky=tkinter.W, padx=5, pady=2)

        '''1、清除旧的文件夹'''
        tkinter.Label(self.right_camera_frame,
                      font = self.font_step_title,
                      text = "1、清除人脸数据").grid(row=5, column=0, columnspan=2, sticky=tkinter.W, padx=5, pady=20)

        tkinter.Button(self.right_camera_frame,
                       text = '清除',
                       command = self.gui_clear_datafile).grid(row=6, column=0, columnspan=1, sticky=tkinter.W, padx=5, pady=2)

        '''2、获取用户名字并创建文件夹'''
        tkinter.Label(self.right_camera_frame,
                 font=self.font_step_title,
                 text="2、输入你的个人信息").grid(row=7, column=0, columnspan=2, sticky=tkinter.W, padx=5, pady=20)

        tkinter.Label(self.right_camera_frame, text="姓名: ").grid(row=8, column=0, columnspan=3, sticky=tkinter.W, padx=5, pady=10)
        self.input_name.grid(row=8, column=1, sticky=tkinter.W, padx=0, pady=2)

        tkinter.Label(self.right_camera_frame, text="工号: ").grid(row=9, column=0, columnspan=3, sticky=tkinter.W,padx=5, pady=10)
        self.input_num.grid(row=9, column=1, sticky=tkinter.W, padx=0, pady=2)


        tkinter.Button(self.right_camera_frame,
                  text='确定输入',
                  command=self.gui_get_message).grid(row=10, column=1, columnspan=3, sticky=tkinter.W)

        '''3、保存人脸当前帧到文件夹'''
        tkinter.Label(self.right_camera_frame,
                 font=self.font_step_title,
                 text="3、保存人脸照片").grid(row=11, column=0, columnspan=2, sticky=tkinter.W, padx=5, pady=20)

        tkinter.Button(self.right_camera_frame,
                  text='确定保存',
                  command=self.save_face).grid(row=12, column=0, columnspan=3, sticky=tkinter.W)

        # Show log in GUI
        self.log_all.grid(row=13, column=0, columnspan=15, sticky=tkinter.W, padx=5, pady=20)

        self.right_camera_frame.pack()

    # 建一个文件夹，存放录入的人脸照片和csv文件
    def newfile_mkdir(self):
        # 新建文件夹
        if os.path.isdir(self.path_file_from_camera):
            pass
        else:
            os.mkdir(self.path_file_from_camera)

    # 更新存储用户数据文件夹的数量
    def update_personface_num_cnt(self):
        # 如果已经有录入过该用户的数据，那么新的数据存放的位置文件按序号递增
        if os.listdir("data/data_from_camera/"):
            # 获取已经录入的最后一个人脸序号
            personface_list = os.listdir("data/data_from_camera/")
            personface_num_list = []

            for person in personface_list:
                person_order = person.split('_')[1].split('_')[0]
                personface_num_list.append(int(person_order))
            self.personface_file_num_cnt = max(personface_num_list)
        # 如果没有该人脸数据，从person_1_name 开始
        else:
            self.personface_file_num_cnt = 0

    # 新建存储用户数据的文件夹
    def create_face_folder(self):
        # 更新数据
        self.personface_file_num_cnt += 1
        # 如果有输出用户名字
        if self.input_name_char:
            self.current_face_new_dir = "person_" \
                                        + str(self.personface_file_num_cnt) \
                                        + "_" \
                                        + self.input_name_char\
                                        + "_" \
                                        + self.input_num_char
        # 没有就默认 unknow
        else:
            self.current_face_new_dir = "person_" \
                                        + str(self.personface_file_num_cnt)\
                                        + "_" \
                                        + "unknowname"\
                                        + "_" \
                                        + "unknownum"

        self.current_face_dir = self.path_file_from_camera + self.current_face_new_dir
        # 创建文件夹
        os.makedirs(self.current_face_dir)
        # 输出提示
        self.log_all["text"] = "\"" + self.current_face_dir + "/\" 已经被创建!"
        logging.info("\n%-40s %s","新建的人脸文件夹 / Create folders:", self.current_face_dir)

        # 人脸计数器
        self.personface_photo_num_cnt = 0

        self.face_folder_created_flag = True

    def save_face(self):
        # 判断是否有文件夹
        if self.face_folder_created_flag:
            # 判断当前帧也没有人脸
            if self.current_frame_face == 1:
                if not self.out_of_range_flag:
                    self.personface_photo_num_cnt += 1
                    self.label_cnt_face_in_datafile['text'] = self.personface_photo_num_cnt
                    self.label_user_filepthoto_num['text'] = "文件夹" + self.current_face_new_dir + "已保存的照片数量："
                     # 生成人脸图象
                    self.face_ROI_image = np.zeros((int(self.face_ROI_height * 2),
                                                    self.face_ROI_width * 2,
                                                    3),
                                                   np.uint8)
                    for ii in range(self.face_ROI_height * 2):
                        for jj in range(self.face_ROI_width * 2):
                            self.face_ROI_image[ii][jj] = self.current_frame[self.face_ROI_height_start - self.hh + ii][
                                self.face_ROI_width_start - self.ww + jj]

                    self.log_all["text"] = "\"" + self.current_face_dir + "/img_face_" + str(
                        self.personface_photo_num_cnt) + ".jpg\"" + " 已经被保存!"

                    self.face_ROI_image = cv2.cvtColor(self.face_ROI_image, cv2.cv2.COLOR_BGR2RGB)

                    cv2.cv2.imwrite(self.current_face_dir + "/img_face_" + str(self.personface_photo_num_cnt) + ".jpg", self.face_ROI_image)
                    logging.info("%-40s %s/img_face_%s.jpg", "写入本地 ：",
                                 str(self.current_face_dir), str(self.personface_photo_num_cnt))
                else:
                    self.log_all["text"] = "人脸超出摄像头检测范围!"
            else:
                self.log_all["text"] = "当前未能检测到人脸!"
        else:
            self.log_all["text"] = "请确保输入你的信息!"



    # 获取视频流的当前帧
    def get_frame(self):
        try:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                return ret, cv2.cvtColor(frame, cv2.cv2.COLOR_BGR2RGB)
        except:
            print("没有视频流输入!!!")


    # 获取人脸
    def process(self):
        ret, self.current_frame = self.get_frame()
        faces = detector(self.current_frame, 0)
        # 获取当前帧
        if ret:
            self.label_face_cnt['text'] = str(len(faces))
            # 如果检测到人脸
            if len(faces) != 0:
                # 用框框画出感兴趣的区域
                for k, d in enumerate(faces):
                    self.face_ROI_width_start = d.left()
                    self.face_ROI_height_start = d.top()
                    # 框框的大小
                    self.face_ROI_width = (d.right() - d.left())
                    self.face_ROI_height = (d.bottom() - d.top())
                    self.hh = int(self.face_ROI_height / 2)
                    self.ww = int(self.face_ROI_width / 2)

                    # 判断框是否超出视频框
                    if(d.right() + self.ww) > 640 or (d.bottom() + self.hh) > 480 or \
                        (d.left() - self.ww) < 0 or (d.top() - self.hh) < 0 :
                        self.label_warning["text"] = "请调整人脸位置，至检测框为白色!!!"
                        self.label_warning['fg'] = 'red'
                        self.out_of_range_flag = True
                        color_rectangle = (255, 0, 0)
                    else:
                        self.out_of_range_flag = False
                        self.label_warning["text"] = ""
                        color_rectangle = (255, 255, 255)
                    self.current_frame = cv2.cv2.rectangle(self.current_frame,
                                                       tuple([d.left() - self.ww, d.top() - self.hh]),
                                                       tuple([d.right() + self.ww, d.bottom() + self.hh]),
                                                       color_rectangle, 2)

            self.current_frame_face = len(faces)

            # 转换图片类型
            global  img_PhotoImage
            img_Image = Image.fromarray(self.current_frame)
            img_PhotoImage = ImageTk.PhotoImage(image=img_Image)
            self.label.img_tk = img_PhotoImage
            self.label.configure(image=img_PhotoImage)

            # 刷新当前帧
        self.gui.after(20, self.process)


    def run(self):
        self.newfile_mkdir()
        self.update_personface_num_cnt()
        self.gui_info()
        self.process()
        self.gui.mainloop()

def main():
    logging.basicConfig(level=logging.INFO)
    get_face = Get_Face()
    get_face.run()


if __name__ == '__main__':
    main()
