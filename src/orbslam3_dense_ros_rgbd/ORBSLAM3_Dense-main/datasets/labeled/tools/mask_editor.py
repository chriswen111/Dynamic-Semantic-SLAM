"""Local pixel correction companion for LabelPaw. Work only on dedicated copies."""
from pathlib import Path
import base64, io, json, os, sys, time, shutil
import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QImage,QPixmap,QPainter,QPen,QColor,QShortcut,QKeySequence,QFontDatabase,QFont
from PySide6.QtWidgets import (QApplication,QMainWindow,QGraphicsView,QGraphicsScene,QWidget,
    QVBoxLayout,QHBoxLayout,QComboBox,QSlider,QSpinBox,QPushButton,QLabel,QMessageBox,QListWidget,QSplitter)

ROOT=Path(__file__).resolve().parent
DATA=Path(os.environ.get('GRAPE_EDITOR_DATA',str(ROOT/'editable_pixel_annotations'))).resolve()
LABELS=['unlabeled','grape','branch','leaf','ground']
COLORS=np.array([[0,0,0],[210,35,145],[255,190,0],[30,190,70],[70,130,245]],np.uint8)
CHINESE=['未标注 / 擦除','葡萄果实','枝干','叶片','地面']
schema_file=DATA/'label_schema.json'
if schema_file.exists():
    schema=json.loads(schema_file.read_text(encoding='utf-8'))
    LABELS=schema['names'];COLORS=np.asarray(schema['colors_rgb'],dtype=np.uint8)
    CHINESE=['未标注 / 擦除']+[{'grape':'葡萄果实','branch':'枝干','ground':'地面','leaf':'叶片'}[name] for name in LABELS[1:]]
    if len(LABELS)!=5 or set(LABELS[1:])!={'grape','branch','ground','leaf'}:raise ValueError('Invalid four-class schema')

def configure_fonts():
    # Qt's offscreen backend can expose no system fonts; register Windows fonts explicitly.
    if getattr(configure_fonts,'done',False):return
    for name in ['msyh.ttc','msyhbd.ttc','segoeui.ttf','arial.ttf']:
        p=Path('C:/Windows/Fonts')/name
        if p.exists():QFontDatabase.addApplicationFont(str(p))
    QApplication.instance().setFont(QFont('Microsoft YaHei',10))
    configure_fonts.done=True

def encode_annotations(mask,name):
    h,w=mask.shape;shapes=[]
    for cls in range(1,5):
        count,comp,stats,_=cv2.connectedComponentsWithStats((mask==cls).astype(np.uint8),8)
        for idx in range(1,count):
            x,y,bw,bh,_=map(int,stats[idx]);a=(comp[y:y+bh,x:x+bw]==idx).astype(np.uint8)
            buffer=io.BytesIO();Image.fromarray(a).save(buffer,format='PNG')
            shapes.append(dict(label=LABELS[cls],points=[[x,y],[x+bw-1,y+bh-1]],group_id=None,
                shape_type='mask',flags={},mask=base64.b64encode(buffer.getvalue()).decode(),description='Pixel correction in local LabelPaw companion'))
    return dict(version='6.3.1',flags={'human_reviewed':False},shapes=shapes,imagePath=name,imageData=None,imageHeight=h,imageWidth=w)

def decode_annotations(data):
    mask=np.zeros((data['imageHeight'],data['imageWidth']),np.uint8)
    for s in data['shapes']:
        if s['shape_type']!='mask' or s['label'] not in LABELS:raise ValueError('Unsupported label or shape')
        (x,y),(x2,y2)=s['points'];a=np.array(Image.open(io.BytesIO(base64.b64decode(s['mask']))))>0
        mask[y:y2+1,x:x2+1][a]=LABELS.index(s['label'])
    return mask

def save_mask(mask,path):
    path=Path(path).resolve()
    if not path.is_relative_to(DATA.resolve()):raise ValueError('Save outside editor copies prohibited')
    annotation=encode_annotations(mask,path.name)
    assert np.array_equal(mask,decode_annotations(annotation))
    backups=DATA/'backups'/path.stem;backups.mkdir(parents=True,exist_ok=True)
    stamp=time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000)
    out=DATA/'masks'/path.name;js=path.with_suffix('.json')
    for src in [out,js]:
        if src.exists():shutil.copyfile(src,backups/(stamp+'_'+src.name))
    out.parent.mkdir(exist_ok=True)
    # Stage both formats first; PNG remains the authoritative pixel data.
    temp=out.with_suffix('.tmp.png');Image.fromarray(mask).save(temp);os.replace(temp,out)
    temp=js.with_suffix('.tmp');temp.write_text(json.dumps(annotation,ensure_ascii=False),encoding='utf-8');os.replace(temp,js)
    # Update batch review previews when these folders exist.
    if (DATA/'color_masks').is_dir():Image.fromarray(COLORS[mask]).save(DATA/'color_masks'/path.name)
    if (DATA/'overlays').is_dir():
        original=np.array(Image.open(path).convert('RGB'));overlay=original.copy();active=mask>0
        overlay[active]=np.round(original[active]*.6+COLORS[mask[active]]*.4).astype(np.uint8)
        Image.fromarray(overlay).save(DATA/'overlays'/path.with_suffix('.jpg').name,quality=92)

def pixmap(rgb):
    rgb=np.ascontiguousarray(rgb);h,w=rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy())

class PaintView(QGraphicsView):
    def __init__(self,owner):
        super().__init__();self.owner=owner;self.painting=False;self.last=None;self.erase=False
        self.setScene(QGraphicsScene(self));self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor('#111827'))
    def coordinate(self,event):
        p=self.mapToScene(event.position().toPoint());return (round(p.x()),round(p.y()))
    def mousePressEvent(self,event):
        if event.button() in (Qt.LeftButton,Qt.RightButton) and self.owner.mask is not None:
            self.owner.checkpoint();self.painting=True;self.erase=event.button()==Qt.RightButton
            self.last=self.coordinate(event)
            if self.owner.tool.currentIndex()==1:self.owner.fill(self.last,0 if self.erase else self.owner.category.currentIndex());self.painting=False
            else:self.owner.stroke(self.last,self.last,self.erase)
            event.accept()
        else:super().mousePressEvent(event)
    def mouseMoveEvent(self,event):
        if self.painting:
            point=self.coordinate(event);self.owner.stroke(self.last,point,self.erase);self.last=point;event.accept()
        else:super().mouseMoveEvent(event)
    def mouseReleaseEvent(self,event):
        if self.painting:self.painting=False;self.last=None;event.accept()
        else:super().mouseReleaseEvent(event)
    def wheelEvent(self,event):
        factor=1.2 if event.angleDelta().y()>0 else 1/1.2
        if .15<self.transform().m11()*factor<20:self.scale(factor,factor)
        event.accept()

class MaskEditor(QMainWindow):
    def __init__(self):
        configure_fonts();super().__init__();self.setWindowTitle('葡萄语义标注修正 · LabelPaw配套像素工具');self.resize(1380,880)
        self.mask=None;self.image=None;self.path=None;self.dirty=False;self.history=[];self.redo_history=[];self.current_index=-1
        self.paths=sorted((DATA/'images').glob('*.png'))
        layout=QVBoxLayout();bar=QHBoxLayout();self.category=QComboBox();self.category.addItems(CHINESE);self.category.setCurrentIndex(1)
        self.tool=QComboBox();self.tool.addItems(['画笔','连通区域换类']);self.size=QSpinBox();self.size.setRange(1,120);self.size.setValue(12)
        self.opacity=QSlider(Qt.Horizontal);self.opacity.setRange(0,80);self.opacity.setValue(40);self.opacity.setMaximumWidth(150)
        for widget in [QLabel('类别'),self.category,QLabel('工具'),self.tool,QLabel('笔宽'),self.size,QLabel('叠加'),self.opacity]:bar.addWidget(widget)
        for name,fn in [('撤销',self.undo),('重做',self.redo),('适应窗口',self.fit),('保存 Ctrl+S',self.save),('导出审核完成训练集',self.export_training)]:
            button=QPushButton(name);button.clicked.connect(fn);bar.addWidget(button)
        layout.addLayout(bar)
        hint=QLabel('左键涂选中类别；右键擦为未标注。连通换类作用于当前标签的连通区域。滚轮缩放，滚动条平移；A/D切图，1–4选类别，0擦除。');hint.setWordWrap(True);layout.addWidget(hint)
        self.list=QListWidget();self.list.addItems([p.name for p in self.paths]);self.list.setMaximumWidth(235)
        self.view=PaintView(self);split=QSplitter();split.addWidget(self.list);split.addWidget(self.view);layout.addWidget(split,1)
        center=QWidget();center.setLayout(layout);self.setCentralWidget(center)
        self.opacity.valueChanged.connect(self.refresh);self.list.currentRowChanged.connect(self.select)
        self.shortcuts=[]
        for key,fn in [('Ctrl+S',self.save),('Ctrl+Z',self.undo),('Ctrl+Y',self.redo),('A',lambda:self.move(-1)),('D',lambda:self.move(1))]:self.shortcut(key,fn)
        for c in range(5):self.shortcut(str(c),lambda cls=c:self.category.setCurrentIndex(cls))
        if self.paths:self.list.setCurrentRow(0)
    def shortcut(self,key,fn):
        s=QShortcut(QKeySequence(key),self);s.activated.connect(fn);self.shortcuts.append(s)
    def move(self,d):self.list.setCurrentRow(max(0,min(len(self.paths)-1,self.current_index+d)))
    def resolve_dirty(self):
        if not self.dirty:return True
        answer=QMessageBox.question(self,'未保存修改','保存当前修改后继续？',QMessageBox.Save|QMessageBox.Discard|QMessageBox.Cancel,QMessageBox.Save)
        if answer==QMessageBox.Cancel:return False
        if answer==QMessageBox.Save:return self.save()
        return True
    def select(self,index):
        if index<0:return
        if not self.resolve_dirty():
            self.list.blockSignals(True);self.list.setCurrentRow(self.current_index);self.list.blockSignals(False);return
        self.current_index=index;self.path=self.paths[index]
        self.image=np.array(Image.open(self.path).convert('RGB'))
        maskfile=DATA/'masks'/self.path.name
        self.mask=np.array(Image.open(maskfile)).astype(np.uint8) if maskfile.exists() else decode_annotations(json.loads(self.path.with_suffix('.json').read_text(encoding='utf-8')))
        self.history=[];self.redo_history=[];self.dirty=False;self.refresh();self.fit()
    def checkpoint(self):
        self.history.append(self.mask.copy());self.history=self.history[-40:];self.redo_history=[]
    def stroke(self,start,end,erase=False):
        cv2.line(self.mask,start,end,0 if erase else self.category.currentIndex(),self.size.value())
        cv2.circle(self.mask,end,max(0,self.size.value()//2),0 if erase else self.category.currentIndex(),-1)
        self.dirty=True;self.refresh()
    def fill(self,point,value):
        x,y=point;h,w=self.mask.shape
        if 0<=x<w and 0<=y<h:
            cv2.floodFill(self.mask,None,(x,y),value,flags=8);self.dirty=True;self.refresh()
    def undo(self):
        if self.history:self.redo_history.append(self.mask.copy());self.mask=self.history.pop();self.dirty=True;self.refresh()
    def redo(self):
        if self.redo_history:self.history.append(self.mask.copy());self.mask=self.redo_history.pop();self.dirty=True;self.refresh()
    def refresh(self):
        if self.mask is None:return
        alpha=self.opacity.value()/100;out=self.image.copy();active=self.mask>0
        out[active]=np.round(self.image[active]*(1-alpha)+COLORS[self.mask[active]]*alpha).astype(np.uint8)
        self.view.scene().clear();self.view.scene().addPixmap(pixmap(out));h,w=self.mask.shape;self.view.scene().setSceneRect(0,0,w,h)
        self.statusBar().showMessage(f'{self.current_index+1}/{len(self.paths)}  {self.path.name}  '+('有未保存修改' if self.dirty else '已加载 / 已保存')+'；保存不代表审核完成')
    def fit(self):
        self.view.fitInView(self.view.sceneRect(),Qt.KeepAspectRatio)
    def save(self):
        if self.mask is None:return True
        try:save_mask(self.mask,self.path)
        except Exception as e:QMessageBox.critical(self,'保存失败',str(e));return False
        self.dirty=False;self.refresh();return True
    def export_training(self):
        if not self.resolve_dirty():return
        answer=QMessageBox.question(self,'导出范围','仅导出已逐张人工修正并确认的图片。\n确认当前图片已完成审核，并将其加入训练导出列表？',QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if answer!=QMessageBox.Yes:return
        if not self.save():return
        out=DATA/'reviewed_training';(out/'images').mkdir(parents=True,exist_ok=True);(out/'masks').mkdir(exist_ok=True)
        shutil.copyfile(self.path,out/'images'/self.path.name)
        training=np.full_like(self.mask,255);valid=self.mask>0;training[valid]=self.mask[valid]-1
        Image.fromarray(training).save(out/'masks'/self.path.name)
        js=json.loads(self.path.with_suffix('.json').read_text(encoding='utf-8'));js['flags']['human_reviewed']=True
        self.path.with_suffix('.json').write_text(json.dumps(js,ensure_ascii=False),encoding='utf-8')
        training_labels={str(i):label for i,label in enumerate(LABELS[1:])};training_labels['255']='ignore'
        (out/'labels.json').write_text(json.dumps(training_labels,indent=2),encoding='utf-8')
        QMessageBox.information(self,'已导出',f'当前图片已导出至：\n{out}\n未标注像素转换为255忽略。')
    def closeEvent(self,event):
        event.accept() if self.resolve_dirty() else event.ignore()

if __name__=='__main__':
    app=QApplication(sys.argv);window=MaskEditor();window.show();sys.exit(app.exec())
