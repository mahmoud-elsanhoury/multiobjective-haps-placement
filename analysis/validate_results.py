#!/usr/bin/env python3
"""Compare reproduced publication results with original reference CSVs, when supplied."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FILES=['camera_ready_headline_results.csv','common_population_peb_ecdf.csv','test_results_design_selected.csv',
       'paper_representatives_design_selected.csv','urban_geometry_summary.csv']

def compare(ref,got):
    a=pd.read_csv(ref);b=pd.read_csv(got)
    if a.columns.tolist()!=b.columns.tolist() or a.shape!=b.shape:
        return False,'column/shape mismatch'
    for k in a.columns:
        if pd.api.types.is_numeric_dtype(a[k]) and pd.api.types.is_numeric_dtype(b[k]):
            if not np.allclose(a[k].to_numpy(float),b[k].to_numpy(float),atol=1e-8,rtol=1e-8,equal_nan=True):
                return False,f'numeric mismatch: {k}'
        elif not a[k].fillna('__NULL__').astype(str).equals(b[k].fillna('__NULL__').astype(str)):
            return False,f'text mismatch: {k}'
    return True,'matched'

def main():
    errors=[]
    for name in FILES:
        ref=ROOT/'reference_results/raw'/name;got=ROOT/'outputs/reproduction_run/raw'/name
        if not ref.is_file() or not got.is_file():errors.append(f'Missing reference or reproduction: {name}');continue
        ok,why=compare(ref,got)
        print(('PASS' if ok else 'FAIL'),name,why)
        if not ok:errors.append(name+': '+why)
    if errors:
        print('Validation incomplete/failed:',*errors,sep='\n  ');return 2
    print('Reference comparisons passed (CSV tolerance 1e-8; scientific interpretation remains in the paper).');return 0

if __name__=='__main__':sys.exit(main())
