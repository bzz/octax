"""Small reusable investigation card, rendered from actual pixel trajectories."""
import base64
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from PIL import Image
import seaborn as sns


def write_round_card(run, title, conclusion, rows, links):
    """Export a compact MinAtar-style policy GIF and a self-contained HTML card.

    The saved trajectory is illustrative; the table uses aggregate evaluation.
    No trajectory selection or editing is performed here.
    """
    run = Path(run)
    with np.load(run / 'visualizations/policy_trajectory.npz') as trace:
        cmap = ListedColormap([(0., 0., 0.), *sns.color_palette('cubehelix', 1)])
        norm = BoundaryNorm([0, 1, 2], 2)
        fig, ax = plt.subplots(figsize=(4.4, 2.65), dpi=100)
        fig.subplots_adjust(left=0, right=1, bottom=.03, top=.87)
        ax.set_axis_off()
        artist = ax.imshow(trace['frames'][0].astype(float) + .5, cmap=cmap, norm=norm, interpolation='none')
        heading = fig.suptitle('', fontsize=10, y=.97)
        images = []
        for i, frame in enumerate(trace['frames']):
            artist.set_data(frame.astype(float) + .5)
            heading.set_text(f"Policy · score {trace['scores'][i]:.0f} · step {trace['steps'][i]} · 4×")
            fig.canvas.draw()
            images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'))
        images[0].save(run / 'policy_card.gif', save_all=True, append_images=images[1:], duration=130, loop=0)
        images[len(images)//2].save(run / 'policy_card.png')
        seed, length, score = int(trace['seed']), int(trace['length']), float(trace['final_score'])
        plt.close(fig)
    animation = base64.b64encode((run / 'policy_card.gif').read_bytes()).decode()
    table = ''.join(f'<tr><th scope="row">{escape(label)}</th><td>{escape(str(value))}</td></tr>' for label, value in rows)
    footer = ' · '.join(f'<a href="{escape(url, quote=True)}">{escape(label)}</a>' for label, url in links)
    html = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title>
<style>body{{margin:0;padding:24px;background:#f4f3ef;color:#20251e;font:15px/1.5 system-ui,sans-serif}}article{{max-width:470px;margin:auto;padding:22px;background:white;border:1px solid #dddcd3;border-radius:16px}}h1{{font-size:20px;margin:0 0 12px}}img{{display:block;width:100%;height:auto}}table{{width:100%;border-collapse:collapse;margin:16px 0}}th,td{{text-align:left;padding:7px 0;border-bottom:1px solid #eee}}td{{text-align:right}}p{{margin:12px 0}}small{{color:#62665d}}a{{color:#28604e}}footer{{margin-top:18px}}</style>
<article><h1>{escape(title)}</h1><p>{escape(conclusion)}</p>
<img src="data:image/gif;base64,{animation}" width="440" height="265" alt="Actual pixel-policy Tetris gameplay, seed {seed}, final score {score:g}">
<small>Illustrative seed {seed} · {length:,} decisions · score {score:g}. Animation is not aggregate evidence.</small>
<table>{table}</table><footer>{footer}</footer></article></html>'''
    (run / 'result_card.html').write_text(html)
