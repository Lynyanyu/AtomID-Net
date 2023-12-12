from torchvision import transforms
import time
import sys
import torch
from utils.data_utils import *
from models.unet import UNet
import pandas as pd

import threading
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

MODEL_PATH = "./models/best_model_0209.pt"

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.trained_model_path = MODEL_PATH
        self._initUI()
        self._initBackend()

    def _initUI(self):
        '''初始化UI'''
        self._setIcon()
        self.statusBar = QStatusBar()
        self.file_name_line_edit = QLineEdit()
        self.file_name_line_edit.setToolTip("Input image path")
        self.file_name_line_edit.textChanged.connect(self.set_process_btn_enabled)
        self.file_browse_btn = self.Button("Browse", "Browse", self.select_file)
        self.file_open_btn = self.Button("Open", "Open", self.open_file)
        self.start_process_btn = self.Button("Process", "Process", self.start_or_stop_process)
        #self.mode_cbox = QCheckBox()
        #self.mode_cbox_caption_txt = self.TextView("real time mode")
        self.map_cbox = QCheckBox()
        self.map_cbox_caption_txt = self.TextView("show map")
        self.image_browser_lb = QLabel()
        self.image_browser_lb.setMinimumSize(512, 512)
        self.image_browser_lb.setStyleSheet("border-width: 1px;border-style: solid;border-color: rgb(25, 25, 25)")
        
        self.save_image_btn = self.Button("Save image", "Save image", self.save_image)
        self.save_points_btn = self.Button("Save points", "Save points", self.save_points)
        self.return_default_btn = self.Button("Default settings","Default settings", self.return_default_settings)
        
        self.mark_transparent_caption_txt = self.TextView("Marks Transparency")
        self.mark_transparent_caption_txt.setContentsMargins(0, 20, 0, 0)
        self.mark_transparent_scroll = QScrollBar()
        self.mark_transparent_scroll.setOrientation(Qt.Horizontal)
        self.mark_transparent_scroll.setMinimumWidth(120)
        self.mark_transparent_scroll.setRange(0, 100)
        self.mark_transparent_txt = self.TextView("1.00")
        self.mark_transparent_scroll.valueChanged.connect(lambda: self.onScrollChange(self.mark_transparent_scroll, self.mark_transparent_txt, "mark_transparent"))
        
        self.filter_param_caption_txt = self.TextView("Filter Lower Bound")
        self.filter_param_caption_txt.setContentsMargins(0, 20, 0, 0)
        self.filter_param_scroll = QScrollBar()
        self.filter_param_scroll.setMinimumWidth(120)
        self.filter_param_scroll.setOrientation(Qt.Horizontal)
        self.filter_param_scroll.setRange(1, 100)
        self.filter_param_txt = self.TextView("0.10")
        self.filter_param_scroll.valueChanged.connect(lambda: self.onScrollChange(self.filter_param_scroll, self.filter_param_txt, "filter_param"))
        
        self.win_size_caption_txt = self.TextView("Filter Window Size")
        self.win_size_caption_txt.setContentsMargins(0, 20, 0, 0)
        self.win_size_scroll = QScrollBar()
        self.win_size_scroll.setMinimumWidth(120)
        self.win_size_scroll.setOrientation(Qt.Horizontal)
        self.win_size_scroll.setRange(1, 5)
        self.win_size_txt = self.TextView("3")
        self.win_size_scroll.valueChanged.connect(lambda: self.onScrollChange(self.win_size_scroll, self.win_size_txt, "win_size"))

        # Pyqt5-Scroll

        # --------------- layout settings ----------------- #
        fileSysLayout = QHBoxLayout()
        fileSysLayout.setContentsMargins(10, 0, 10, 0)
        fileSysLayout.addWidget(self.file_name_line_edit)
        fileSysLayout.addWidget(self.file_browse_btn)
        fileSysLayout.addWidget(self.file_open_btn)
        fileSysLayout.addWidget(self.start_process_btn)
        
        controlSubLayout1 = QHBoxLayout()
        controlSubLayout1.setContentsMargins(10, 0, 10, 30)
        controlSubLayout1.addWidget(self.mark_transparent_scroll)
        controlSubLayout1.addWidget(self.mark_transparent_txt)
        
        controlSubLayout2 = QHBoxLayout()
        controlSubLayout2.setContentsMargins(10, 0, 10, 30)
        controlSubLayout2.addWidget(self.filter_param_scroll)
        controlSubLayout2.addWidget(self.filter_param_txt)

        controlSubLayout3 = QHBoxLayout()
        controlSubLayout3.setContentsMargins(10, 0, 10, 30)
        controlSubLayout3.addWidget(self.win_size_scroll)
        controlSubLayout3.addWidget(self.win_size_txt)
        
        controlSubLayout4 = QHBoxLayout()
        controlSubLayout4.addWidget(self.map_cbox)
        controlSubLayout4.addWidget(self.map_cbox_caption_txt)
        controlSubLayout4.setAlignment(Qt.AlignLeft)
        
        controlLayout = QVBoxLayout()
        controlLayout.setContentsMargins(10, 20, 10, 0)
        controlLayout.addWidget(self.save_image_btn)
        controlLayout.addWidget(self.save_points_btn)
        controlLayout.addWidget(self.return_default_btn)
        controlLayout.addLayout(controlSubLayout4)
        controlLayout.addWidget(self.mark_transparent_caption_txt)
        controlLayout.addLayout(controlSubLayout1)
        controlLayout.addWidget(self.filter_param_caption_txt)
        controlLayout.addLayout(controlSubLayout2)
        controlLayout.addWidget(self.win_size_caption_txt)
        controlLayout.addLayout(controlSubLayout3)
        controlLayout.setAlignment(Qt.AlignTop)
        
        mainFrameLayout = QHBoxLayout()
        mainFrameLayout.setContentsMargins(10, 0, 10, 0)
        mainFrameLayout.addWidget(self.image_browser_lb)
        mainFrameLayout.addLayout(controlLayout)

        layout = QVBoxLayout()
        layout.addLayout(fileSysLayout)
        layout.addLayout(mainFrameLayout)
        
        layout.addSpacerItem(QSpacerItem(0, -1, hPolicy=QSizePolicy.Fixed, vPolicy=QSizePolicy.Fixed))
        layout.addWidget(self.statusBar, alignment=Qt.AlignBottom)
        self.setLayout(layout)
        
        self.show()

        # 初始空窗口将功能性按钮禁用
        # self.mode_cbox.setEnabled(False)
        self.file_open_btn.setEnabled(False)
        self.start_process_btn.setEnabled(False)
        self.save_image_btn.setEnabled(False)
        self.save_points_btn.setEnabled(False)

    def _initBackend(self):

        # Set Device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Get Model
        self.model = UNet(1, 1)
        self.model.load_state_dict(torch.load(self.trained_model_path), strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.toTensor = transforms.ToTensor()
        self.transform = transforms.Normalize([0.5], [0.5])
        
        # Other buffers
        self.selected_file_path = None
        self.image = None
        self.image_displayed = None
        self.mark_image = None
        self.model_buffer = {"inputs":None, "out_diam":None, "out_map":None, "keep_centers":None, "keep_phases":None}
        
        self.processing = False
        self.the_process_thread = None

        # Settings
        self.display_size = 768
        self.timer = time.time()
        self.return_default_settings()

        
    def _setIcon(self):
        '''set icon'''
        # self.setGeometry(500, 500, 500, 500)
        self.move(int(QApplication.desktop().width() * 0.13), int(QApplication.desktop().height() * 0.08))
        self.setWindowTitle('AtomIDNet Demo')
        #self.setWindowIcon(QIcon('icons/icon.ico'))
        
    def set_process_btn_enabled(self):
        if len(self.file_name_line_edit.text())>0:
            self.file_open_btn.setEnabled(True)
            self.start_process_btn.setEnabled(True)
        else:
            self.file_open_btn.setEnabled(False)
            self.start_process_btn.setEnabled(False)
        
        
    def TextView(self, init_text=None, margin=[0, 0, 0, 0]):
        text = QLabel(self)
        text.setText(init_text)
        text.setContentsMargins(*margin)
        return text


    def Button(self, btn_name: str = None, Tips: str = None, clicked_event=None, position: tuple = None):
        '''set a new button'''
        btn = QPushButton(btn_name, self)
        btn.setToolTip(Tips)
        btn.setStatusTip(Tips)
        btn.setFixedHeight(22)
        btn.setIconSize(QSize(32, 32))
        btn.resize(btn.sizeHint())
        btn.setFont(QFont("Noto Sans", 8))
        # btn.setIcon(QIcon.fromTheme("document-open", QIcon("icons/icon.png")))
        if position is not None:
            btn.move(position[0], position[1])
        if clicked_event is not None:
            btn.clicked.connect(clicked_event)
        return btn
    
    def select_file(self):
        selected_file_path, _ = QFileDialog.getOpenFileName(self, "select image", ".", "image (*.png *.jpg *.jpge *.bmp *.tiff)")
        self.file_name_line_edit.setText(selected_file_path)
        if len(selected_file_path)>0:
            self.selected_file_path = selected_file_path
            self.selected_folder_path = None
            self.start_process_btn.setEnabled(True)
            self.open_file()
        
    def select_folder(self):
        selected_folder_path = QFileDialog.getExistingDirectory(self, "select folder", ".")
        self.file_name_line_edit.setText(selected_folder_path)
        if len(selected_folder_path)>0:
            self.selected_folder_path = selected_folder_path
            self.selected_file_path = None
            self.start_process_btn.setEnabled(True)
            
    def open_file(self):
        self.save_points_btn.setEnabled(False)
        try:
            self.image = cv2.imdecode(np.fromfile(self.selected_file_path, dtype=np.uint8),cv2.IMREAD_COLOR)
            self.mark_image = None
            self.model_buffer = {"inputs":None, "out_diam":None, "out_map":None, "keep_centers":None}
            self.display_image(self.image)
            self.image_displayed = self.image.copy()
            self.save_image_btn.setEnabled(True)
        except Exception as e:
            print(e)
            
    def display_image(self, image=None):
        if image is None:
            qimage_displayed = self.image.copy()
        else:
            qimage_displayed = image.copy()
        
        image_shape = qimage_displayed.shape[:2]
        if True: #np.max(image_shape)<self.display_size:
            idx = np.argsort(image_shape)
            image_shape = np.array([round(image_shape[idx[0]]/image_shape[idx[1]] * self.display_size), self.display_size])[idx]
            qimage_displayed = cv2.resize(qimage_displayed, image_shape)
        image_shape = qimage_displayed.shape[:2]
        qimage_displayed = QImage(qimage_displayed, image_shape[1], image_shape[0], QImage.Format_RGB888)
        self.image_browser_lb.setPixmap(QPixmap.fromImage(qimage_displayed))
        self.image_browser_lb.setMinimumSize(*list(image_shape))
        self.image_browser_lb.setMaximumSize(*list(image_shape))
    
    def save_image(self):
        if self.image_displayed is None:
            return
        else:
            defalut_save_path = self.selected_file_path.split('.')[0]
            save_path, save_type = QFileDialog.getSaveFileName(self, "save image", defalut_save_path, "png(*.png)")
            try:
                cv2.imwrite(save_path, self.image_displayed[:,:,[2,1,0]])
                self.statusBar.showMessage("Image saved in {}".format(save_path))
            except Exception as e:
                self.statusBar.showMessage("Path \"{}\" is invalid!".format(save_path))
    
    def save_points(self):
        if (self.image_displayed is None) or (self.model_buffer["keep_centers"] is None):
            return
        else:
            if len(self.model_buffer["keep_centers"])==0:
                self.statusBar.showMessage("Centers not detected!")
                return
            
            keep_centers = self.model_buffer["keep_centers"].squeeze().cpu().numpy().astype('int')
            keep_phases = -np.ones((len(keep_centers), 1)) if self.model_buffer["keep_phases"] is None else self.model_buffer["keep_phases"].squeeze().cpu().numpy().astype('int')
            
            defalut_save_path = self.selected_file_path.split('.')[0]
            save_path, save_type = QFileDialog.getSaveFileName(self, "save points", defalut_save_path, "csv(*.csv)")
            try:
                save_array = np.concatenate((np.arange(len(keep_centers))[:,None]+1,
                                             keep_centers[:,[1,0]],
                                             keep_phases[:]), axis=-1)
                df = pd.DataFrame(data=save_array, columns=["Peak#", "X", "Y", "Phase"])
                df.to_csv(save_path,index=False)
                self.statusBar.showMessage("Csv saved in {}".format(save_path))
            except Exception as e:
                self.statusBar.showMessage("Path \"{}\" is invalid!".format(save_path))
        
    def start_or_stop_process(self):
        if self.processing:
            self.stop()
        else:
            self.start() 
        
    def start(self):
        if self.processing == True:
            return
        else:
            self.processing = True
            self.mark_image = None
            self.model_buffer = {"inputs":None, "out_diam":None, "out_map":None, "keep_centers":None, "keep_phases":None}
            self.start_process_btn.setText("stop")

            self.the_process_thread = threading.Thread(target=self.analyze)
            self.the_process_thread.setDaemon(True)
            self.the_process_thread.start()
    
    def stop(self):
        if self.processing == False:
            return
        else:
            self.processing = False
            self.start_process_btn.setText("process")
            #self.mode_cbox.setEnabled(True)
            
    def onScrollChange(self, scrollbar:QScrollBar, textview:QLabel, name:str):
        if name in ["mark_transparent"]:
            textview.setText(str(scrollbar.value() / max(1, scrollbar.maximum())))
            self.apply_mark()
        elif name in ["filter_param"]:
            textview.setText(str(scrollbar.value() / max(1, scrollbar.maximum())))
            #if self.mode_cbox.isChecked():
            #    self.analyze_nms()
        elif name in ["win_size"]:
            textview.setText(str(2**scrollbar.value()))
            #if self.mode_cbox.isChecked():
            #    self.analyze_nms()
        
        
    def return_default_settings(self):
        #self.mode_cbox.setChecked(False)
        self.map_cbox.setChecked(False)
        
        self.mark_transparent_scroll.setValue(100)
        self.mark_transparent_txt.setText(str(1.0))
        
        self.filter_param_scroll.setValue(10)
        self.filter_param_txt.setText(str(0.1))
        
        self.win_size_scroll.setValue(3)
        self.win_size_txt.setText(str(2**3))
        
        self.analyze_nms()
        
    def analyze(self):
        self.timer = time.time()
        self.analyze_model()
        self.analyze_nms()
        self.stop()
        cost_time = float(time.time() - self.timer)
        self.timer = time.time()
        self.statusBar.showMessage("Cost {:.4f}s".format(cost_time))
    
    def analyze_model(self):
        inputs = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        inputs = self.transform(self.toTensor(inputs)).unsqueeze(0)
        inputs = inputs.type(torch.FloatTensor).to(self.device)
        _, _, h, w = inputs.size()
        upRatio = 1
        inputs = F.interpolate(inputs, size=(round(h/16 * upRatio) * 16, round(w/16 * upRatio) * 16))
        
        out_diam, out_map = self.model(inputs)
        
        self.model_buffer["inputs"] = inputs
        self.model_buffer["out_diam"] = out_diam.detach()
        self.model_buffer["out_map"] = out_map.detach()
        
    
    def analyze_nms(self):
        inputs = self.model_buffer["inputs"]
        out_diam = self.model_buffer["out_diam"]
        out_map = self.model_buffer["out_map"]
        
        if (inputs is None) or (out_diam is None) or (out_map is None):
            return
        
        gray_th = float(self.filter_param_txt.text())
        iou_th = 0.3
        window_size = min(int(self.win_size_txt.text()), min(inputs.size(-2), inputs.size(-1))//16)
        keep_centers = visualize(inputs, out_diam, out_map, None, self.selected_file_path, gray_th, iou_th, window_size, None, self.device)
        diam = torch.floor(out_diam).int().item()
        self.mark_image = np.zeros(self.image.shape)
        for center in keep_centers:
            cv2.circle(self.mark_image, (center[1].item(), center[0].item()), max(1, diam//4), (255, 95, 0), -1)
        
        self.model_buffer["keep_centers"] = keep_centers
        self.save_points_btn.setEnabled(True)
        self.apply_mark()
            
    def apply_mark(self):
        if (self.image is None) or (self.mark_image is None):
            return
        else:
            if not self.map_cbox.isChecked():
                # original + marks
                self.image_displayed = self.image.copy()
                ratio = float(self.mark_transparent_txt.text())
                self.image_displayed[self.mark_image>0] = self.mark_image[self.mark_image>0] * ratio + self.image[self.mark_image>0] * (1-ratio)
            else:
                # map
                image_displayed = self.model_buffer["out_map"].cpu().numpy()
                image_displayed -= image_displayed.min()
                image_displayed /= max(image_displayed.max(), 1e-8)
                image_displayed *= 255
                image_displayed = image_displayed.squeeze()
                self.image_displayed[self.image_displayed<self.image_displayed.mean()-self.image_displayed.std()] = 0
                self.image_displayed = np.around(np.repeat(image_displayed[:,:,None], 3, -1)).astype('uint8')
            
            self.display_image(self.image_displayed)
            

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())