# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\WinPython-64bit-3.5.1.3\UI\MLS\MLS.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(924, 615)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.VideoSlider = QtGui.QSlider(self.centralwidget)
        self.VideoSlider.setGeometry(QtCore.QRect(30, 520, 571, 22))
        self.VideoSlider.setSingleStep(1)
        self.VideoSlider.setProperty("value", 1)
        self.VideoSlider.setOrientation(QtCore.Qt.Horizontal)
        self.VideoSlider.setObjectName(_fromUtf8("VideoSlider"))
        self.progressBar = QtGui.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(590, 560, 118, 23))
        self.progressBar.setProperty("value", 100)
        self.progressBar.setObjectName(_fromUtf8("progressBar"))
        self.VideoButtonBox = QtGui.QFrame(self.centralwidget)
        self.VideoButtonBox.setGeometry(QtCore.QRect(230, 550, 251, 31))
        self.VideoButtonBox.setFrameShape(QtGui.QFrame.StyledPanel)
        self.VideoButtonBox.setFrameShadow(QtGui.QFrame.Raised)
        self.VideoButtonBox.setObjectName(_fromUtf8("VideoButtonBox"))
        self.B_last = QtGui.QToolButton(self.VideoButtonBox)
        self.B_last.setGeometry(QtCore.QRect(70, 0, 37, 31))
        self.B_last.setObjectName(_fromUtf8("B_last"))
        self.B_pause = QtGui.QToolButton(self.VideoButtonBox)
        self.B_pause.setGeometry(QtCore.QRect(110, 0, 37, 31))
        self.B_pause.setObjectName(_fromUtf8("B_pause"))
        self.B_next = QtGui.QToolButton(self.VideoButtonBox)
        self.B_next.setGeometry(QtCore.QRect(150, 0, 37, 31))
        self.B_next.setObjectName(_fromUtf8("B_next"))
        self.TimeShow = QtGui.QFrame(self.centralwidget)
        self.TimeShow.setGeometry(QtCore.QRect(600, 520, 101, 21))
        self.TimeShow.setFrameShape(QtGui.QFrame.StyledPanel)
        self.TimeShow.setFrameShadow(QtGui.QFrame.Raised)
        self.TimeShow.setObjectName(_fromUtf8("TimeShow"))
        self.T_second = QtGui.QLCDNumber(self.TimeShow)
        self.T_second.setGeometry(QtCore.QRect(80, 0, 21, 21))
        self.T_second.setSmallDecimalPoint(False)
        self.T_second.setNumDigits(2)
        self.T_second.setDigitCount(2)
        self.T_second.setMode(QtGui.QLCDNumber.Dec)
        self.T_second.setSegmentStyle(QtGui.QLCDNumber.Flat)
        self.T_second.setProperty("value", 0.0)
        self.T_second.setProperty("intValue", 0)
        self.T_second.setObjectName(_fromUtf8("T_second"))
        self.T_minute = QtGui.QLCDNumber(self.TimeShow)
        self.T_minute.setGeometry(QtCore.QRect(40, 0, 21, 21))
        self.T_minute.setSmallDecimalPoint(False)
        self.T_minute.setNumDigits(2)
        self.T_minute.setDigitCount(2)
        self.T_minute.setMode(QtGui.QLCDNumber.Dec)
        self.T_minute.setSegmentStyle(QtGui.QLCDNumber.Flat)
        self.T_minute.setProperty("value", 0.0)
        self.T_minute.setProperty("intValue", 0)
        self.T_minute.setObjectName(_fromUtf8("T_minute"))
        self.label = QtGui.QLabel(self.TimeShow)
        self.label.setGeometry(QtCore.QRect(70, 0, 16, 21))
        self.label.setScaledContents(False)
        self.label.setObjectName(_fromUtf8("label"))
        self.T_hour = QtGui.QLCDNumber(self.TimeShow)
        self.T_hour.setGeometry(QtCore.QRect(0, 0, 21, 21))
        self.T_hour.setSmallDecimalPoint(False)
        self.T_hour.setNumDigits(2)
        self.T_hour.setDigitCount(2)
        self.T_hour.setMode(QtGui.QLCDNumber.Dec)
        self.T_hour.setSegmentStyle(QtGui.QLCDNumber.Flat)
        self.T_hour.setProperty("value", 0.0)
        self.T_hour.setProperty("intValue", 0)
        self.T_hour.setObjectName(_fromUtf8("T_hour"))
        self.label_2 = QtGui.QLabel(self.TimeShow)
        self.label_2.setGeometry(QtCore.QRect(30, 0, 16, 21))
        self.label_2.setScaledContents(False)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 0, 741, 501))
        self.gridLayoutWidget.setObjectName(_fromUtf8("gridLayoutWidget"))
        self.gridLayout = QtGui.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setSizeConstraint(QtGui.QLayout.SetMaximumSize)
        self.gridLayout.setContentsMargins(-1, -1, 0, 0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.scrollArea = QtGui.QScrollArea(self.gridLayoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName(_fromUtf8("scrollArea"))
        self.scrollAreaWidgetContents = QtGui.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 737, 497))
        self.scrollAreaWidgetContents.setObjectName(_fromUtf8("scrollAreaWidgetContents"))
        self.gridLayout_2 = QtGui.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.labelImage = QtGui.QLabel(self.scrollAreaWidgetContents)
        self.labelImage.setLineWidth(1)
        self.labelImage.setText(_fromUtf8(""))
        self.labelImage.setAlignment(QtCore.Qt.AlignCenter)
        self.labelImage.setObjectName(_fromUtf8("labelImage"))
        self.gridLayout_2.addWidget(self.labelImage, 1, 1, 1, 1)
        self.line_2 = QtGui.QFrame(self.scrollAreaWidgetContents)
        self.line_2.setFrameShape(QtGui.QFrame.VLine)
        self.line_2.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_2.setObjectName(_fromUtf8("line_2"))
        self.gridLayout_2.addWidget(self.line_2, 1, 2, 1, 1)
        self.line = QtGui.QFrame(self.scrollAreaWidgetContents)
        self.line.setFrameShape(QtGui.QFrame.VLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.gridLayout_2.addWidget(self.line, 1, 0, 1, 1)
        self.line_3 = QtGui.QFrame(self.scrollAreaWidgetContents)
        self.line_3.setFrameShape(QtGui.QFrame.HLine)
        self.line_3.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_3.setObjectName(_fromUtf8("line_3"))
        self.gridLayout_2.addWidget(self.line_3, 2, 1, 1, 1)
        self.line_4 = QtGui.QFrame(self.scrollAreaWidgetContents)
        self.line_4.setFrameShape(QtGui.QFrame.HLine)
        self.line_4.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_4.setObjectName(_fromUtf8("line_4"))
        self.gridLayout_2.addWidget(self.line_4, 0, 1, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 924, 23))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        self.menuOpen = QtGui.QMenu(self.menuFile)
        self.menuOpen.setObjectName(_fromUtf8("menuOpen"))
        self.menuHelp = QtGui.QMenu(self.menubar)
        self.menuHelp.setObjectName(_fromUtf8("menuHelp"))
        self.menuEdit = QtGui.QMenu(self.menubar)
        self.menuEdit.setObjectName(_fromUtf8("menuEdit"))
        self.menuWindow = QtGui.QMenu(self.menubar)
        self.menuWindow.setObjectName(_fromUtf8("menuWindow"))
        self.menuTool_2 = QtGui.QMenu(self.menuWindow)
        self.menuTool_2.setObjectName(_fromUtf8("menuTool_2"))
        self.menuClassifier = QtGui.QMenu(self.menuWindow)
        self.menuClassifier.setObjectName(_fromUtf8("menuClassifier"))
        self.menuView = QtGui.QMenu(self.menubar)
        self.menuView.setObjectName(_fromUtf8("menuView"))
        self.menuTool = QtGui.QMenu(self.menubar)
        self.menuTool.setObjectName(_fromUtf8("menuTool"))
        MainWindow.setMenuBar(self.menubar)
        self.W_choose = QtGui.QDockWidget(MainWindow)
        self.W_choose.setObjectName(_fromUtf8("W_choose"))
        self.dockWidgetContents = QtGui.QWidget()
        self.dockWidgetContents.setObjectName(_fromUtf8("dockWidgetContents"))
        self.tab_choose = QtGui.QTabWidget(self.dockWidgetContents)
        self.tab_choose.setGeometry(QtCore.QRect(0, 0, 101, 271))
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab_choose.sizePolicy().hasHeightForWidth())
        self.tab_choose.setSizePolicy(sizePolicy)
        self.tab_choose.setObjectName(_fromUtf8("tab_choose"))
        self.tab_feature = QtGui.QWidget()
        font = QtGui.QFont()
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.tab_feature.setFont(font)
        self.tab_feature.setMouseTracking(False)
        self.tab_feature.setAcceptDrops(False)
        self.tab_feature.setObjectName(_fromUtf8("tab_feature"))
        self.LW_feature = QtGui.QListWidget(self.tab_feature)
        self.LW_feature.setGeometry(QtCore.QRect(0, 0, 101, 251))
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.LW_feature.sizePolicy().hasHeightForWidth())
        self.LW_feature.setSizePolicy(sizePolicy)
        self.LW_feature.setObjectName(_fromUtf8("LW_feature"))
        item = QtGui.QListWidgetItem()
        self.LW_feature.addItem(item)
        item = QtGui.QListWidgetItem()
        self.LW_feature.addItem(item)
        self.tab_choose.addTab(self.tab_feature, _fromUtf8(""))
        self.tab_train = QtGui.QWidget()
        self.tab_train.setObjectName(_fromUtf8("tab_train"))
        self.LW_train = QtGui.QListWidget(self.tab_train)
        self.LW_train.setGeometry(QtCore.QRect(0, 0, 101, 241))
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.LW_train.sizePolicy().hasHeightForWidth())
        self.LW_train.setSizePolicy(sizePolicy)
        self.LW_train.setObjectName(_fromUtf8("LW_train"))
        self.tab_choose.addTab(self.tab_train, _fromUtf8(""))
        self.tab_classifier = QtGui.QWidget()
        self.tab_classifier.setObjectName(_fromUtf8("tab_classifier"))
        self.LW_classifier = QtGui.QListWidget(self.tab_classifier)
        self.LW_classifier.setGeometry(QtCore.QRect(0, 0, 101, 251))
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.LW_classifier.sizePolicy().hasHeightForWidth())
        self.LW_classifier.setSizePolicy(sizePolicy)
        self.LW_classifier.setObjectName(_fromUtf8("LW_classifier"))
        item = QtGui.QListWidgetItem()
        self.LW_classifier.addItem(item)
        item = QtGui.QListWidgetItem()
        self.LW_classifier.addItem(item)
        item = QtGui.QListWidgetItem()
        self.LW_classifier.addItem(item)
        self.tab_choose.addTab(self.tab_classifier, _fromUtf8(""))
        self.W_choose.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.W_choose)
        self.ToolBox = QtGui.QDockWidget(MainWindow)
        self.ToolBox.setObjectName(_fromUtf8("ToolBox"))
        self.ToolBoxContents = QtGui.QWidget()
        self.ToolBoxContents.setObjectName(_fromUtf8("ToolBoxContents"))
        self.TB_1 = QtGui.QToolButton(self.ToolBoxContents)
        self.TB_1.setGeometry(QtCore.QRect(40, 0, 37, 31))
        self.TB_1.setObjectName(_fromUtf8("TB_1"))
        self.TB_2 = QtGui.QToolButton(self.ToolBoxContents)
        self.TB_2.setGeometry(QtCore.QRect(0, 0, 37, 31))
        self.TB_2.setObjectName(_fromUtf8("TB_2"))
        self.line_5 = QtGui.QFrame(self.ToolBoxContents)
        self.line_5.setGeometry(QtCore.QRect(-10, 0, 20, 541))
        self.line_5.setFrameShape(QtGui.QFrame.VLine)
        self.line_5.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_5.setObjectName(_fromUtf8("line_5"))
        self.ToolBox.setWidget(self.ToolBoxContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.ToolBox)
        self.printWidget = QtGui.QDockWidget(MainWindow)
        self.printWidget.setObjectName(_fromUtf8("printWidget"))
        self.dockWidgetContents_2 = QtGui.QWidget()
        self.dockWidgetContents_2.setObjectName(_fromUtf8("dockWidgetContents_2"))
        self.Label_print = QtGui.QTextBrowser(self.dockWidgetContents_2)
        self.Label_print.setGeometry(QtCore.QRect(0, 0, 131, 251))
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Label_print.sizePolicy().hasHeightForWidth())
        self.Label_print.setSizePolicy(sizePolicy)
        self.Label_print.setObjectName(_fromUtf8("Label_print"))
        self.Bar_print = QtGui.QScrollBar(self.dockWidgetContents_2)
        self.Bar_print.setGeometry(QtCore.QRect(0, 260, 71, 16))
        self.Bar_print.setOrientation(QtCore.Qt.Horizontal)
        self.Bar_print.setObjectName(_fromUtf8("Bar_print"))
        self.printWidget.setWidget(self.dockWidgetContents_2)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.printWidget)
        self.actionExit = QtGui.QAction(MainWindow)
        self.actionExit.setObjectName(_fromUtf8("actionExit"))
        self.actionVideo = QtGui.QAction(MainWindow)
        self.actionVideo.setObjectName(_fromUtf8("actionVideo"))
        self.actionImage = QtGui.QAction(MainWindow)
        self.actionImage.setObjectName(_fromUtf8("actionImage"))
        self.actionMusic = QtGui.QAction(MainWindow)
        self.actionMusic.setObjectName(_fromUtf8("actionMusic"))
        self.actionHelp = QtGui.QAction(MainWindow)
        self.actionHelp.setObjectName(_fromUtf8("actionHelp"))
        self.actionUndo = QtGui.QAction(MainWindow)
        self.actionUndo.setObjectName(_fromUtf8("actionUndo"))
        self.actionRedo = QtGui.QAction(MainWindow)
        self.actionRedo.setObjectName(_fromUtf8("actionRedo"))
        self.actionSave = QtGui.QAction(MainWindow)
        self.actionSave.setObjectName(_fromUtf8("actionSave"))
        self.actionSave_AS = QtGui.QAction(MainWindow)
        self.actionSave_AS.setObjectName(_fromUtf8("actionSave_AS"))
        self.actionNew = QtGui.QAction(MainWindow)
        self.actionNew.setObjectName(_fromUtf8("actionNew"))
        self.actionDrawBox = QtGui.QAction(MainWindow)
        self.actionDrawBox.setObjectName(_fromUtf8("actionDrawBox"))
        self.actionClassifier_2 = QtGui.QAction(MainWindow)
        self.actionClassifier_2.setObjectName(_fromUtf8("actionClassifier_2"))
        self.actionAbout = QtGui.QAction(MainWindow)
        self.actionAbout.setObjectName(_fromUtf8("actionAbout"))
        self.actionDrawBox_2 = QtGui.QAction(MainWindow)
        self.actionDrawBox_2.setObjectName(_fromUtf8("actionDrawBox_2"))
        self.menuOpen.addAction(self.actionImage)
        self.menuOpen.addAction(self.actionVideo)
        self.menuOpen.addAction(self.actionMusic)
        self.menuFile.addAction(self.actionNew)
        self.menuFile.addAction(self.menuOpen.menuAction())
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_AS)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menuHelp.addAction(self.actionHelp)
        self.menuHelp.addSeparator()
        self.menuHelp.addAction(self.actionAbout)
        self.menuEdit.addAction(self.actionUndo)
        self.menuEdit.addAction(self.actionRedo)
        self.menuEdit.addSeparator()
        self.menuTool_2.addAction(self.actionDrawBox)
        self.menuClassifier.addAction(self.actionClassifier_2)
        self.menuWindow.addAction(self.menuTool_2.menuAction())
        self.menuWindow.addAction(self.menuClassifier.menuAction())
        self.menuTool.addAction(self.actionDrawBox_2)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuWindow.menuAction())
        self.menubar.addAction(self.menuTool.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        self.tab_choose.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.B_last.setText(_translate("MainWindow", "<|", None))
        self.B_pause.setText(_translate("MainWindow", "||", None))
        self.B_next.setText(_translate("MainWindow", "|>", None))
        self.label.setText(_translate("MainWindow", ":", None))
        self.label_2.setText(_translate("MainWindow", ":", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.menuOpen.setTitle(_translate("MainWindow", "Open", None))
        self.menuHelp.setTitle(_translate("MainWindow", "Help", None))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit", None))
        self.menuWindow.setTitle(_translate("MainWindow", "Window", None))
        self.menuTool_2.setTitle(_translate("MainWindow", "Tool", None))
        self.menuClassifier.setTitle(_translate("MainWindow", "List", None))
        self.menuView.setTitle(_translate("MainWindow", "View", None))
        self.menuTool.setTitle(_translate("MainWindow", "Tool", None))
        self.W_choose.setWindowTitle(_translate("MainWindow", "Choose", None))
        self.tab_feature.setToolTip(_translate("MainWindow", "1", None))
        __sortingEnabled = self.LW_feature.isSortingEnabled()
        self.LW_feature.setSortingEnabled(False)
        item = self.LW_feature.item(0)
        item.setText(_translate("MainWindow", "SIFT", None))
        item = self.LW_feature.item(1)
        item.setText(_translate("MainWindow", "PCA", None))
        self.LW_feature.setSortingEnabled(__sortingEnabled)
        self.tab_choose.setTabText(self.tab_choose.indexOf(self.tab_feature), _translate("MainWindow", "Feature", None))
        self.tab_choose.setTabText(self.tab_choose.indexOf(self.tab_train), _translate("MainWindow", "Train", None))
        __sortingEnabled = self.LW_classifier.isSortingEnabled()
        self.LW_classifier.setSortingEnabled(False)
        item = self.LW_classifier.item(0)
        item.setText(_translate("MainWindow", "Kmeans", None))
        item = self.LW_classifier.item(1)
        item.setText(_translate("MainWindow", "SVM", None))
        item = self.LW_classifier.item(2)
        item.setText(_translate("MainWindow", "Knn", None))
        self.LW_classifier.setSortingEnabled(__sortingEnabled)
        self.tab_choose.setTabText(self.tab_choose.indexOf(self.tab_classifier), _translate("MainWindow", "Classifier", None))
        self.ToolBox.setWindowTitle(_translate("MainWindow", "Tool", None))
        self.TB_1.setText(_translate("MainWindow", "?", None))
        self.TB_2.setText(_translate("MainWindow", "draw", None))
        self.Label_print.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>", None))
        self.actionExit.setText(_translate("MainWindow", "Exit", None))
        self.actionVideo.setText(_translate("MainWindow", "Video", None))
        self.actionImage.setText(_translate("MainWindow", "Image", None))
        self.actionMusic.setText(_translate("MainWindow", "Music", None))
        self.actionHelp.setText(_translate("MainWindow", "Help", None))
        self.actionUndo.setText(_translate("MainWindow", "Undo", None))
        self.actionRedo.setText(_translate("MainWindow", "Redo", None))
        self.actionSave.setText(_translate("MainWindow", "Save", None))
        self.actionSave_AS.setText(_translate("MainWindow", "Save As …", None))
        self.actionNew.setText(_translate("MainWindow", "New", None))
        self.actionDrawBox.setText(_translate("MainWindow", "DrawBox", None))
        self.actionClassifier_2.setText(_translate("MainWindow", "Classifier", None))
        self.actionAbout.setText(_translate("MainWindow", "About", None))
        self.actionDrawBox_2.setText(_translate("MainWindow", "DrawBox", None))
