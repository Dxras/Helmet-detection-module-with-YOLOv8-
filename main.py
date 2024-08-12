from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import smart_inference_mode
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.checks import check_imshow
from ultralytics.yolo.cfg import get_cfg
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, QPoint, Qt
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
from UIFunctions import *
from collections import defaultdict
from pathlib import Path
from utils.capnums import Camera
from utils.rtsp_win import Window
import numpy as np
import time
import json
import torch
import sys
import cv2
import os


class YoloPredictor(BasePredictor, QObject):
    yolo2main_pre_img = Signal(np.ndarray)   # необработанный сигнал изображения
    yolo2main_res_img = Signal(np.ndarray)   # сигнал о результате теста
    yolo2main_status_msg = Signal(str)       # Сигнал обнаружения/приостановки/остановки/завершения тестирования/сообщения об ошибке
    yolo2main_fps = Signal(str)              # fps
    yolo2main_labels = Signal(dict)          # Выявленные классовые результаты (количество по каждой категории)
    yolo2main_progress = Signal(int)         # Полнота
    yolo2main_class_num = Signal(int)        # Количество обнаруженных категорий
    yolo2main_target_num = Signal(int)       # Обнаруженные объекты

    def __init__(self, cfg=DEFAULT_CFG, overrides=None): 
        super(YoloPredictor, self).__init__() 
        QObject.__init__(self)

        self.args = get_cfg(cfg, overrides)
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = f'{self.args.mode}'
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # GUI аргументы
        self.used_model_name = None      # Название модели распознования объектов
        self.new_model_name = None       # ИЗменение модели
        self.source = ''                 # Вставить источник
        self.stop_dtc = False            # Прерывание
        self.continue_dtc = True         # Пауза
        self.save_res = False            # Сохранить результат
        self.save_txt = False            # Сохранить запись в txt
        self.iou_thres = 0.45            # iou
        self.conf_thres = 0.25           # conf
        self.speed_thres = 10            # delay, ms
        self.labels_dict = {}            # Вывод информации расзпознавания
        self.progress_value = 0          # Прогресс
    

        # Используется полсе завершения настройки
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.annotator = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        callbacks.add_integration_callbacks(self)

    # Обнаружение
    @smart_inference_mode()
    def run(self):
        try:
            if self.args.verbose:
                LOGGER.info('')

            # Выбор модели
            self.yolo2main_status_msg.emit('Загрузка модели')
            if not self.model:
                self.setup_model(self.new_model_name)
                self.used_model_name = self.new_model_name

            # Выбор данных
            self.setup_source(self.source if self.source is not None else self.args.source)

            # Check save path/label
            if self.save_res or self.save_txt:
                (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # warmup model
            if not self.done_warmup:
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None

            # Начало обнаружения
            # Для  личного набора данных:


            count = 0                       # запуск локализации frame
            start_time = time.time()        # расчет частоты кадров
            batch = iter(self.dataset)
            while True:
                # Обнаружение прерывания
                if self.stop_dtc:
                    if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()  # выход последнего видео
                    self.yolo2main_status_msg.emit('Распознавание прервано')
                    break
                
                # Изменить модель во время обработки(НЕ РАБОТАЕТ)
                if self.used_model_name != self.new_model_name:  
                    # self.yolo2main_status_msg.emit('Изменение модели')
                    self.setup_model(self.new_model_name)
                    self.used_model_name = self.new_model_name
                
                # Переключение паузы
                if self.continue_dtc:
                    # time.sleep(0.001)
                    self.yolo2main_status_msg.emit('Распознавание..')
                    batch = next(self.dataset)  # Следуюищй датасет

                    self.batch = batch
                    path, im, im0s, vid_cap, s = batch
                    visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.args.visualize else False

                    # Завершение расчета и частота кадров
                    count += 1              # количество кадров +1
                    if vid_cap:
                        all_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)   # все кадры
                    else:
                        all_count = 1
                    self.progress_value = int(count/all_count*1000)         # шкала прогресса (0/1000)
                    if count % 5 == 0 and count >= 5:                     # Рассчет частоты кадров каждые 5 кадров
                        self.yolo2main_fps.emit(str(int(5/(time.time()-start_time))))
                        start_time = time.time()
                    
                    # Предварительная обработка
                    with self.dt[0]:
                        im = self.preprocess(im)
                        if len(im.shape) == 3:
                            im = im[None]
                    # Вывод
                    with self.dt[1]:
                        preds = self.model(im, augment=self.args.augment, visualize=visualize)
                    # Постпроцесс
                    with self.dt[2]:
                        self.results = self.postprocess(preds, im, im0s)

                    #  визуализация, сохранение, запись результата
                    n = len(im)     # To be improved: support multiple img
                    for i in range(n):
                        self.results[i].speed = {
                            'preprocess': self.dt[0].dt * 1E3 / n,
                            'inference': self.dt[1].dt * 1E3 / n,
                            'postprocess': self.dt[2].dt * 1E3 / n}
                        p, im0 = (path[i], im0s[i].copy()) if self.source_type.webcam or self.source_type.from_img \
                            else (path, im0s.copy())
                        p = Path(p)     # the source dir

                        # s:::   video 1/1 (6/6557) 'path':
                        # must, to get boxs\labels
                        label_str = self.write_results(i, self.results, (p, im, im0))   # labels   /// original :s += 
                        
                        # labels and nums dict
                        class_nums = 0
                        target_nums = 0
                        self.labels_dict = {}
                        if 'no detections' in label_str:
                            pass
                        else:
                            for ii in label_str.split(',')[:-1]:
                                nums, label_name = ii.split('~')
                                self.labels_dict[label_name] = int(nums)
                                target_nums += int(nums)
                                class_nums += 1

                        # Сохранение изображения или видео
                        if self.save_res:
                            self.save_preds(vid_cap, i, str(self.save_dir / p.name))

                        # Отправка результатов тестирования
                        self.yolo2main_res_img.emit(im0) #
                        self.yolo2main_pre_img.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                        # self.yolo2main_labels.emit(self.labels_dict)        # webcam need to change the def write_results
                        self.yolo2main_class_num.emit(class_nums)
                        self.yolo2main_target_num.emit(target_nums)

                        if self.speed_thres != 0:
                            time.sleep(self.speed_thres/1000)   # delay , ms

                    self.yolo2main_progress.emit(self.progress_value)   # шкала прогресса

                # Распознавание завершено
                if count + 1 >= all_count:
                    if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()  # конец
                    self.yolo2main_status_msg.emit('Распознавание завершено')
                    break

        except Exception as e:
            pass
            print(e)
            self.yolo2main_status_msg.emit('%s' % e)


    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):

        preds = ops.non_max_suppression(preds,
                                        self.conf_thres,
                                        self.iou_thres,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        # print(results)
        return results

    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1         # attention
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        # log_string += '%gx%g ' % im.shape[2:]         # !!! don't add img size~
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes  # TODO: make boxes inherit from tensors

        if len(det) == 0:
            return f'{log_string}(no detections), ' # if no, send this~~

        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n}~{self.model.names[int(c)]},"   #   {'s' * (n > 1)}, "   # don't add 's'
        # now log_string is the classes 👆


        # Запись
        for d in reversed(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            if self.save_txt:  # Запись в файл
                line = (cls, *(d.xywhn.view(-1).tolist()), conf) \
                    if self.args.save_conf else (cls, *(d.xywhn.view(-1).tolist()))  # формат записей
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            if self.save_res or self.args.save_crop or self.args.show or True:  # Add bbox to image(must)
                c = int(cls)  # integer class
                name = f'id:{int(d.id.item())} {self.model.names[c]}' if d.id is not None else self.model.names[c]
                label = None if self.args.hide_labels else (name if self.args.hide_conf else f'{name} {conf:.2f}')
                self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if self.args.save_crop:
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string
        


class MainWindow(QMainWindow, Ui_MainWindow):
    main2yolo_begin_sgl = Signal()  # The main window sends an execution signal to the yolo instance
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # basic interface
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)  # rounded transparent
        self.setWindowFlags(Qt.FramelessWindowHint)  # Set window flag: hide window borders
        UIFuncitons.uiDefinitions(self)
        # Show module shadows
        UIFuncitons.shadow_style(self, self.Class_QF, QColor(162,129,247))
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF, QColor(64, 186, 193))
        


        # Модели
        self.pt_list = os.listdir('./models')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))   # сортирование по размеру файла
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.Qtimer_ModelBox = QTimer(self)
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)

        # Yolo-v8 thread
        self.yolo_predict = YoloPredictor()                           # Create a Yolo instance
        self.select_model = self.model_box.currentText()                   # default model
        self.yolo_predict.new_model_name = "./models/%s" % self.select_model  
        self.yolo_thread = QThread()                                  # Create yolo thread
        self.yolo_predict.yolo2main_pre_img.connect(lambda x: self.show_image(x, self.pre_video)) 
        self.yolo_predict.yolo2main_res_img.connect(lambda x: self.show_image(x, self.res_video))
        self.yolo_predict.yolo2main_status_msg.connect(lambda x: self.show_status(x))             
        self.yolo_predict.yolo2main_fps.connect(lambda x: self.fps_label.setText(x))              
        # self.yolo_predict.yolo2main_labels.connect(self.show_labels)                            
        self.yolo_predict.yolo2main_class_num.connect(lambda x:self.Class_num.setText(str(x)))         
        self.yolo_predict.yolo2main_target_num.connect(lambda x:self.Target_num.setText(str(x)))       
        self.yolo_predict.yolo2main_progress.connect(lambda x: self.progress_bar.setValue(x))     
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)     
        self.yolo_predict.moveToThread(self.yolo_thread)              

        # Параметры модели
        self.model_box.currentTextChanged.connect(self.change_model)     
        self.iou_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'iou_spinbox'))    # iou box
        self.iou_slider.valueChanged.connect(lambda x:self.change_val(x, 'iou_slider'))      # iou ползунок
        self.conf_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'conf_spinbox'))  # conf box
        self.conf_slider.valueChanged.connect(lambda x:self.change_val(x, 'conf_slider'))    # conf ползунок
        self.speed_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'speed_spinbox'))# speed box
        self.speed_slider.valueChanged.connect(lambda x:self.change_val(x, 'speed_slider'))  # speed ползунок

        # Панель показателей
        self.Class_num.setText('--') # Класс
        self.Target_num.setText('--') # Объекты
        self.fps_label.setText('--')
        self.Model_name.setText(self.select_model)
        
        # Выбор файла
        self.src_file_button.clicked.connect(self.open_src_file)  # выбор файла
        #self.src_cam_button.clicked.connect(self.show_status("НЕ РЕАЛИЗОВАНА."))#chose_cam
        #self.src_rtsp_button.clicked.connect(self.show_status("НЕ РЕАЛИЗОВАНА."))#chose_rtsp

        # Старт
        self.run_button.clicked.connect(self.run_or_continue)   # старт пауза
        self.stop_button.clicked.connect(self.stop)             # прервать

        # Кнопки
        self.save_res_button.toggled.connect(self.is_save_res)  # Сохранить файл
        self.save_txt_button.toggled.connect(self.is_save_txt)  # Сохранить текст
        self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))   # Левая панель
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))   # Правая панель
        
        # initialization
        self.load_config()

    # Отображение исходного и обработанного изображения
    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # Сохранение исходного соотнашения данных
            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    #  при нажатии старта\паузы
    def run_or_continue(self):
        if self.yolo_predict.source == '':
            self.show_status('Пожалуйста, выберите файл для работы')
            self.run_button.setChecked(False)
        else:
            self.yolo_predict.stop_dtc = False
            if self.run_button.isChecked():
                self.run_button.setChecked(True)    # старт
                self.save_txt_button.setEnabled(False)  # Запрещает сохранять файл сразу после запуска обнаружения
                self.save_res_button.setEnabled(True)
                self.show_status('Распознавание')
                self.yolo_predict.continue_dtc = True   # Контролирует, будет ли модель приостановлена
                if not self.yolo_thread.isRunning():
                    self.yolo_thread.start()
                    self.main2yolo_begin_sgl.emit()

            else:
                self.yolo_predict.continue_dtc = False
                self.show_status("Пауза")
                self.run_button.setChecked(False)    # старт

    # при работе
    def show_status(self, msg):
        self.status_bar.setText(msg)
        if msg == 'Распознавание завершено' or msg == '':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)    
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()         # конец
        elif msg == 'Распознавание прервано' or msg == '':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)    
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()         # конец
            self.pre_video.clear()           # очистка дисплея
            self.res_video.clear()          
            self.Class_num.setText('--')
            self.Target_num.setText('--')
            self.fps_label.setText('--')

    # статус при выборе файла\ функция выбора
    def open_src_file(self):
        config_file = 'config/fold.json'    
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']     
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
        if name:
            self.yolo_predict.source = name
            self.show_status('Загружаемый файл：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)  
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()             

    # Select camera source----  have one bug
    def chose_cam(self):
        try:
            self.stop()
            MessageBox(
                self.close_button, title='Note', text='loading camera...', time=2000, auto=True).exec()
            # get the number of local cameras
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.src_cam_button.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.src_cam_button.mapToGlobal(self.src_cam_button.pos()).x()     
            y = self.src_cam_button.mapToGlobal(self.src_cam_button.pos()).y()     
            y = y + self.src_cam_button.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec(pos)
            if action:
                self.yolo_predict.source = action.text()
                self.show_status('Loading camera：{}'.format(action.text()))

        except Exception as e:
            self.show_status('%s' % e)

    # select network source
    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.2:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))
    
    # load network sources
    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.close_button, title='', text=' rtsp...', time=1000, auto=True).exec()
            self.yolo_predict.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.show_status('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.show_status('%s' % e)

    # При нажатии\отжатии сохранить видео\текст
    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('Файл не сохранен')
            self.yolo_predict.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            self.show_status('Файл сохранен')
            self.yolo_predict.save_res = True
    
    # При нажатии\отжатии сохранить текст
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('Текстовая запись не сохранена.')
            self.yolo_predict.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            self.show_status('Текстовая запись сохранена.')
            self.yolo_predict.save_txt = True

    # Configuration initialization  ~~~wait to change~~~
    def load_config(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33   
            rate = 10
            save_res = 0   
            save_txt = 0    
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "save_res": save_res,
                          "save_txt": save_txt
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                save_res = 0
                save_txt = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                save_res = config['save_res']
                save_txt = config['save_txt']
        self.save_res_button.setCheckState(Qt.CheckState(save_res))
        self.yolo_predict.save_res = (False if save_res==0 else True )
        self.save_txt_button.setCheckState(Qt.CheckState(save_txt)) 
        self.yolo_predict.save_txt = (False if save_txt==0 else True )
        self.run_button.setChecked(False)  
        self.show_status("Начинаем работу~")

    # Terminate button and associated state
    def stop(self):
        if self.yolo_thread.isRunning():
            self.yolo_thread.quit()         # end thread
        self.yolo_predict.stop_dtc = True
        self.run_button.setChecked(False)    # start key recovery
        self.save_res_button.setEnabled(True)   # Ability to use the save button
        self.save_txt_button.setEnabled(True)   # Ability to use the save button
        self.pre_video.clear()           # clear image display
        self.res_video.clear()           # clear image display
        self.progress_bar.setValue(0)
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')

    # Зименение параметров распознавания
    def change_val(self, x, flag):
        if flag == 'iou_spinbox':
            self.iou_slider.setValue(int(x*100))    # Меняется ползунок
        elif flag == 'iou_slider':
            self.iou_spinbox.setValue(x/100)        # TМеняется ползунок
            self.show_status('IOU Threshold: %s' % str(x/100))
            self.yolo_predict.iou_thres = x/100
        elif flag == 'conf_spinbox':
            self.conf_slider.setValue(int(x*100))
        elif flag == 'conf_slider':
            self.conf_spinbox.setValue(x/100)
            self.show_status('Conf Threshold: %s' % str(x/100))
            self.yolo_predict.conf_thres = x/100
        elif flag == 'speed_spinbox':
            self.speed_slider.setValue(x)
        elif flag == 'speed_slider':
            self.speed_spinbox.setValue(x)
            self.show_status('Delay: %s ms' % str(x))
            self.yolo_predict.speed_thres = x  # ms
            
    # Изменеие модели
    def change_model(self,x):
        self.select_model = self.model_box.currentText()
        self.yolo_predict.new_model_name = "./models/%s" % self.select_model
        self.show_status('Change Model：%s' % self.select_model)
        self.Model_name.setText(self.select_model)

    # label result
    # def show_labels(self, labels_dic):
    #     try:
    #         self.result_label.clear()
    #         labels_dic = sorted(labels_dic.items(), key=lambda x: x[1], reverse=True)
    #         labels_dic = [i for i in labels_dic if i[1]>0]
    #         result = [' '+str(i[0]) + '：' + str(i[1]) for i in labels_dic]
    #         self.result_label.addItems(result)
    #     except Exception as e:
    #         self.show_status(e)

    # Cycle monitoring model file changes
    def ModelBoxRefre(self):
        pt_list = os.listdir('./models')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))
        # It must be sorted before comparing, otherwise the list will be refreshed all the time
        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.model_box.clear()
            self.model_box.addItems(self.pt_list)

    # Get the mouse position (used to hold down the title bar and drag the window)
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # Optimize the adjustment when dragging the bottom and right edges of the window size
    def resizeEvent(self, event):
        # Update Size Grips
        UIFuncitons.resize_grips(self)

    # Exit Exit thread, save settings
    def closeEvent(self, event):
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.iou_spinbox.value()
        config['conf'] = self.conf_spinbox.value()
        config['rate'] = self.speed_spinbox.value()
        config['save_res'] = (0 if self.save_res_button.checkState()==Qt.Unchecked else 2)
        config['save_txt'] = (0 if self.save_txt_button.checkState()==Qt.Unchecked else 2)
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        # Exit the process before closing
        if self.yolo_thread.isRunning():
            self.yolo_predict.stop_dtc = True
            self.yolo_thread.quit()
            MessageBox(
                self.close_button, title='Note', text='Закрываем приложение', time=3000, auto=True).exec()
            sys.exit(0)
        else:
            sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())  
