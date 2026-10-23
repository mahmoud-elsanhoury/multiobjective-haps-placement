#!/usr/bin/env python3
"""Execute the unchanged numerical experiment through its public notebook entrypoint."""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true',help='Run scientific known-answer checks; does not run the full experiment')
    args=p.parse_args()
    env=os.environ.copy()
    env.update({'HAPS_GNSS_ROOT':str(ROOT),'HAPS_GNSS_CONFIG':str(ROOT/'config/experiment.json'),
                'HAPS_GNSS_OUTPUT_ROOT':str(ROOT/'outputs'),'HAPS_GNSS_RUN_ID':('smoke' if args.smoke else 'reproduction_run'),
                'HAPS_GNSS_SMOKE_TEST':('1' if args.smoke else '0'),'MPLBACKEND':'Agg'})
    env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    if not args.smoke:
        subprocess.run([sys.executable,str(ROOT/'run/verify_inputs.py')],check=True,cwd=ROOT,env=env)
    code="from pathlib import Path\nimport nbformat\nfrom nbclient import NotebookClient\nroot=Path.cwd()\nsource=root/'notebooks/reproduction.ipynb'\nout=root/'outputs/reproduction_executed.ipynb'\nout.parent.mkdir(parents=True,exist_ok=True)\nnb=nbformat.read(source,as_version=4)\nclient=NotebookClient(nb,timeout=None,kernel_name='python3',resources={'metadata':{'path':str(root)}})\nwith client.setup_kernel():\n    for i,cell in enumerate(nb.cells):\n        try:client.execute_cell(cell,i)\n        finally:nbformat.write(nb,out)\nprint('[OK] execution completed')\n"
    subprocess.run([sys.executable,'-c',code],check=True,cwd=ROOT,env=env)
    if args.smoke:return
    run_root=ROOT/'outputs/reproduction_run'
    commands=[
      ['build_tables.py'],['plot_design_tradeoffs.py'],
      ['plot_selected_layouts.py','--icon',str(ROOT/'assets/haps_icon_hq_full.png'),'--slot-width','360','--slot-height','223','--pdf'],
      ['plot_common_population.py'],['plot_seed_variability.py'],
    ]
    for spec in commands:
        cmd=[sys.executable,str(ROOT/'analysis'/spec[0]),'--run-root',str(run_root),*spec[1:]]
        subprocess.run(cmd,check=True,cwd=ROOT,env=env)
    print('[OK] publication analyses finished')

if __name__=='__main__':main()
