# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:17:44 2024

@author: Administrator
"""
import taichi as ti
import meshtaichi_patcher as Patcher
mesh = Patcher.load_mesh('boxx.1.hex.node', relations=["CV","CF","VF","FV","CE","CC"])
#mesh_instence
mesh.verts.place({'x':ti.math.vec3})
mesh.cells.place({'gc':ti.math.vec3})
mesh.cells.place({'volume':ti.float32})
mesh.cells.place({'StateVariable':ti.math.vec3})#1D Roe scheme test
mesh.cells.place({'Residual':ti.math.vec3})
mesh.faces.place({'gf':ti.math.vec3})
mesh.verts.x.from_numpy(mesh.get_position_as_numpy())
x = mesh.verts.x
gc = mesh.cells.gc
volume = mesh.cells.volume
StateVariable = mesh.cells.StateVariable
Residual= mesh.cells.Residual
gf= mesh.faces.gf