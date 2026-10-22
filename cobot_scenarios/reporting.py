"""Static preview gallery: shareable without installing the simulator."""
import html,json,shutil
from pathlib import Path

def gallery(roots,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);cards=[]
    for root in roots:
        root=Path(root)
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            if (folder/'resolved.json').exists():
                r=json.loads((folder/'resolved.json').read_text());m=r['request'];q=r['qualification']
                label=f"{m['task']} · {m['split']} · layout {m['layout']} / style {m['style']}"
                images=[]
                for camera in ('kinect','wrist'):
                    source=folder/(camera+'.png')
                    if source.exists():
                        name=folder.name+'_'+camera+'.png';shutil.copy2(source,out/name)
                        images.append(f'<a href="{name}"><img src="{name}" alt="{camera}"></a>')
                status=f"Initialization: {q['initialization']}; navigation screen: {q['navigation']['status']}. Reach/grasp and instruction qualification pending."
                cards.append(f'<article><h2>{html.escape(label)}</h2><p>{html.escape(r["instruction"])}</p>{"".join(images)}<p>{html.escape(status)}</p><small>{folder.name}</small></article>')
            elif (folder/'failure.json').exists():
                r=json.loads((folder/'failure.json').read_text())
                cards.append(f'<article><h2>Blocked: {html.escape(r.get("task",folder.name))}</h2><p>{html.escape(r["error"])}</p><small>{folder.name}</small></article>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>CoBot scenario previews</title><style>body{font:16px system-ui;background:#111827;color:#eef2ff;margin:24px}article{padding:18px;margin-bottom:18px;background:#1e293b;border-radius:12px}img{width:48%;margin-right:1%;border-radius:6px}h2{font-size:19px}small{color:#94a3b8}</style><h1>CoBot pilot — scenario previews</h1><p>These are initialization checks, not successful demonstrations. Blocked cells remain in the curriculum.</p>'+''.join(cards))
    return dict(previews=len(cards),gallery=str(out/'index.html'))
