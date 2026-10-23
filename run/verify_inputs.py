#!/usr/bin/env python3
"""Check the original input/source hashes before reproducing the study."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def verify():
    cfg=json.loads((ROOT/'config/experiment.json').read_text(encoding='utf-8'))
    missing=[];bad=[]
    expected={}
    for rel,sha in cfg['source_hashes'].items():expected[rel]=sha
    for name,sha in cfg['input_hashes'].items():
        expected['data/'+('brdc/' if name.endswith('.rnx') else 'ephemeris/')+name]=sha
    manifest=ROOT/'data/CAMERA_READY_INPUT_MANIFEST.json'
    if not manifest.is_file():missing.append(str(manifest.relative_to(ROOT)))
    else:
        m=json.loads(manifest.read_text(encoding='utf-8'))
        for roi in cfg['rois']:
            rid=roi['id'];expected[f'data/{rid}/footprints.gpkg']=m['cities'][rid]['sha256']
    for rel,sha in expected.items():
        path=ROOT/rel
        if not path.is_file():missing.append(rel)
        elif digest(path).lower()!=sha.lower():bad.append(rel)
    if missing or bad:
        if missing:print('MISSING:',*missing,sep='\n  ')
        if bad:print('HASH MISMATCH:',*bad,sep='\n  ')
        return False
    print('[OK] required scientific inputs and source files match published configuration')
    return True

if __name__=='__main__':
    raise SystemExit(0 if verify() else 2)
