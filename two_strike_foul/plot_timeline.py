"""Research plot of the actual gates, pitch proposals and scoreboard observations."""
from pathlib import Path


def plot(job, rows, gate, proposals):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    time=[r['time_sec'] for r in rows]
    fig,axes=plt.subplots(4,1,figsize=(15,9),sharex=True,gridspec_kw={'height_ratios':[2,1.2,1,1]})
    for window in gate['windows']:
        axes[0].axvspan(window['start_sec'],window['end_sec'],color='lightblue',alpha=.65)
    for proposal in proposals:
        axes[0].hlines(float(proposal['model_score']),float(proposal['start_sec']),float(proposal['end_sec']),color='red',linewidth=3)
    axes[0].set(ylim=(0,1.05),ylabel='Pitch model score',title=Path(job['video']).name)
    axes[0].legend(handles=[Patch(color='lightblue',label='VideoMAE input windows (not ground truth)'),Line2D([0],[0],color='red',linewidth=3,label='Pitch proposals')],loc='upper right')
    for key,color in [('balls','green'),('strikes','goldenrod'),('outs','firebrick')]:
        values=[(r.get('count') or {}).get(key,np.nan) for r in rows]
        axes[1].plot(time,values,'.-',ms=3,lw=1,color=color,label=key)
    axes[1].set(ylim=(-.2,3.2),ylabel='YOLO B/S/O')
    axes[1].legend(ncol=3,loc='upper right')
    if rows:
        bases=np.array([[-1 if (r.get('runners') or {}).get(key) is None else int(r['runners'][key]) for r in rows] for key in ['first','second','third']])
        edges=[0]+[(a+b)/2 for a,b in zip(time,time[1:])]+[job['duration_sec']]
        color=ListedColormap(['white','#dddddd','#e9b928'])
        axes[2].pcolormesh(edges,[0,1,2,3],bases,cmap=color,norm=BoundaryNorm([-1.5,-.5,.5,1.5],3),shading='flat')
    axes[2].set(yticks=[.5,1.5,2.5],yticklabels=['1st','2nd','3rd'],ylabel='YOLO runners')
    axes[2].legend(handles=[Patch(facecolor='white',edgecolor='gray',label='Unknown'),Patch(color='#dddddd',label='Predicted empty'),Patch(color='#e9b928',label='Predicted occupied')],ncol=3,loc='upper right')
    for key,color in [('upper_score','steelblue'),('lower_score','purple')]:
        values=[(r.get('scoreboard') or {}).get(key) for r in rows]
        axes[3].plot(time,[v if v is not None else np.nan for v in values],'.-',ms=3,lw=1,color=color,label=key)
    known_scores=[(r.get('scoreboard') or {}).get(k) for r in rows for k in ['upper_score','lower_score']]
    maximum=max([v for v in known_scores if v is not None]+[1])
    axes[3].set(ylabel='OCR score',xlabel='Source-video time [sec]',xlim=(0,job['duration_sec']),ylim=(-.1,maximum+.2))
    from matplotlib.ticker import MaxNLocator
    axes[3].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[3].legend(ncol=2,loc='upper right')
    for axis in axes:axis.grid(axis='y',alpha=.25)
    fig.tight_layout()
    fig.savefig(Path(job['output'])/'timeline.png',dpi=150)
    plt.close(fig)
