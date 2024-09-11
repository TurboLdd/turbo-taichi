# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 23:02:06 2024

@author: Administrator
"""

import taichi as ti #ti core#D:\ProgramData\Anaconda3\envs\taichi\lib\site-packages\taichi\__init__.py
import argparse
import meshtaichi_patcher as Patcher #tipatcher D:\ProgramData\Anaconda3\envs\taichi\lib\site-packages\meshtaichi_patcher\__init__.py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="models/deer.1.node")
parser.add_argument('--arch', default='gpu')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

#ti.init(arch=getattr(ti, args.arch), dynamic_index=True, random_seed=0)
ti.init(arch=getattr(ti, args.arch), dynamic_index=True, random_seed=0)
mesh = Patcher.load_mesh(args.model, relations=["CE", "CV", "EV"])