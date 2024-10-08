# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 22:48:15 2024

@author: Administrator
"""
import taichi as ti
import meshtaichi_patcher as Patcher
#import variableDef
ti.init(arch=ti.cpu, random_seed=0)
#define the mesh used in calculation
mesh = Patcher.load_mesh('boxx.1.hex.node', relations=["CE", "CV", "EV"])
#place the needed data for cell face and vertex
#where certain field is created automotically
mesh.verts.place({'x':ti.math.vec3})#node coordinates (x,y,z)
mesh.verts.place({'Velocity':ti.math.vec3})#velocity field (vx,vy,vz)
mesh.verts.place({'Thermol_Properiy':ti.math.vec3})#filed containing pressure density and temperature
mesh.cells.place({'gc':ti.math.vec3})#cell centroid (x,y,z)
mesh.cells.place({'volume':ti.f32})#cell centroid (x,y,z)

mesh.cells.place({'B':ti.f32})
mesh.verts.x.from_numpy(mesh.get_position_as_numpy())
x = mesh.verts.x
gc = mesh.cells.gc
volume=mesh.cells.volume



@ti.kernel
def Calculate_Volume():
    for c in mesh.cells:
        vec=ti.Vector.field(4,dtype=ti.f32, shape=3)
        for f in c.faces:
            i=0
            for v in f.verts:#quad
                vec[i,:]=v.x
                i+=1
            dv1=vec[0]-vec[1]
            dv2=vec[1]-vec[2]
            dv3=vec[2]-vec[3]
            dv4=vec[3]-vec[1]
            a1=0.5*dv1.cross(dv2)
            a2=0.5*dv3.cross(dv4)
            a=a1.norm()+a2.norm()
            d=c.gc.dot(a1)/a1.norm()/3
            c.volume+=a*d
@ti.kernel
def Calculate_Spatial_Gradient_GreenGauss(gradient:ti.template()):
    for c in mesh.cells:
        A=ti.Matrix()
        b=ti.Vector()
        i=0
        for v in c.verts():
            A[i:]=v.x-c.gc
            i+=1
        gradient=ti.solve(A,b)
    pass
@ti.kernel
def Roe_flux(a: ti.template(),flux:ti.template()):
    
    pass
@ti.fun
def Eigen_V(J_Matrix: ti.template(),eigen:ti.template()):
    
    pass
@ti.kernel
def Residual():
    pass

@ti.kernel
def Time_Marching():
    pass
@ti.kernel
def Seudo_Time_Marching():
    pass