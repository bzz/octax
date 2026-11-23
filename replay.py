"""Replay a saved NPZ trajectory with MinAtar's Matplotlib rendering approach.

Run from a local terminal with a GUI-capable Matplotlib backend. Close the window
to stop. Offline GIF/PNG export uses visualize_run.py and does not need a GUI.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns

p=argparse.ArgumentParser()
p.add_argument('trajectory')
p.add_argument('--speed',type=float,default=4.0)
a=p.parse_args()
if a.speed<=0:
    p.error('--speed must be positive')
t=np.load(a.trajectory)
fig,ax=plt.subplots(figsize=(8,4.5))
cmap=ListedColormap([(0,0,0),*sns.color_palette('cubehelix',1)])
im=ax.imshow(t['frames'][0].astype(float)+.5,cmap=cmap,norm=BoundaryNorm([0,1,2],2),interpolation='none')
ax.set_xlabel('CHIP-8 pixels (64 × 32)')
def update(i):
    im.set_data(t['frames'][i].astype(float)+.5)
    ax.set_title(f"Step {t['steps'][i]} | score {t['scores'][i]:.0f} | {a.speed:g}× playback")
    return [im]
animation=FuncAnimation(fig,update,frames=len(t['frames']),interval=(8*4/60)*1000/a.speed,repeat=True)
plt.show()
