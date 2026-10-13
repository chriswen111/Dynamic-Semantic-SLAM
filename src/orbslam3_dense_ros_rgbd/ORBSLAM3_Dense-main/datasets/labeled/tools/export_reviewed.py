"""Build a fresh timestamped training set from current approved annotations only."""
from pathlib import Path
import json,shutil,time,csv
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent.parent
schema=json.loads((ROOT/'label_schema.json').read_text(encoding='utf-8'))
accepted=[]
for js in sorted((ROOT/'images').glob('color_image_*.json')):
    annotation=json.loads(js.read_text(encoding='utf-8'))
    if annotation.get('flags',{}).get('human_reviewed') is True:accepted.append((js,annotation))
if not accepted:
    print('没有human_reviewed=true的图片。请先在像素修改程序中逐张审核确认。');raise SystemExit(0)
target=ROOT/('reviewed_export_'+time.strftime('%Y%m%d_%H%M%S'))
if target.exists():raise RuntimeError('同名导出目录已存在，请稍后重试。')
(target/'images').mkdir(parents=True);(target/'masks').mkdir()
records=[]
for js,annotation in accepted:
    name=annotation['imagePath'];image=ROOT/'images'/name;source_mask=ROOT/'masks'/name
    if Path(name).name!=name:raise ValueError('Image path must be a filename')
    mask=np.asarray(Image.open(source_mask));training=np.full(mask.shape,255,np.uint8);valid=mask>0;training[valid]=mask[valid]-1
    shutil.copyfile(image,target/'images'/name);Image.fromarray(training).save(target/'masks'/name)
    records.append({'image':name,'human_reviewed':True})
(target/'labels.json').write_text(json.dumps({**{str(i):label for i,label in enumerate(schema['names'][1:])},'255':'ignore'},indent=2),encoding='utf-8')
with (target/'approved_images.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=['image','human_reviewed']);writer.writeheader();writer.writerows(records)
print(f'已导出 {len(accepted)} 张当前审核完成图片到：\n{target}')
