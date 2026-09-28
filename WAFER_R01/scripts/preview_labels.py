"""Local-only source-label inspection, excluded from redistribution packages."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
m=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
fig,axs=plt.subplots(3,2,figsize=(16,14))
for ax,name in zip(axs.flat,m['names']):
    r=next(r for r in m['records'] if r['split']=='train' and any(o['class']==name for o in r['objects']))
    ax.imshow(Image.open(r['image']))
    for o in r['objects']:
        x1,y1,x2,y2=o['bbox_xyxy']; color='lime' if o['class']==name else 'cyan'
        ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color=color,linewidth=1.5))
        ax.text(x1,y1,o['class'],color='black',fontsize=7,bbox={'facecolor':color,'alpha':.75,'pad':1})
    ax.set_title(f"{r['id']} | {name}"); ax.axis('off')
fig.suptitle('Source microscopy images with provided VOC labels | local QA only',fontsize=17)
fig.tight_layout(); out=ROOT/'qa';out.mkdir(exist_ok=True)
fig.savefig(out/'source_label_samples.png',dpi=130);plt.close(fig)
