from pathlib import Path
import os,sys
root=Path(__file__).resolve().parent.parent
os.environ['GRAPE_EDITOR_DATA']=str(root)
from PySide6.QtWidgets import QApplication
from mask_editor import MaskEditor
app=QApplication(sys.argv)
window=MaskEditor()
window.setWindowTitle('葡萄全量语义标注 · 审核和像素修正')
window.show()
sys.exit(app.exec())
