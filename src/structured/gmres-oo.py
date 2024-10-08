# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 00:03:32 2024

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 00:57:52 2024

@author: Administrator
"""

import taichi as ti
import time
import numpy as np

@ti.data_oriented
class GMRESSolver:
    def __init__(self,offsets,numStep,variableNum,n):
        #variable
        self.variableNum=variableNum
        self.numStep=numStep
        self.n=n
        self.offsets=offsets#DIA format the offset of each block
        self.jacobi=ti.Matrix.field(variableNum,variableNum,ti.f32)# rho vx vy vz T p c nv
        self.p1=ti.root.dense(ti.i,len(self.offsets))
        self.u=ti.Vector.field(variableNum,ti.f32)#solution
        self.b=ti.Vector.field(variableNum,ti.f32)#RHS
        self.temp=ti.Vector.field(variableNum,ti.f32)#temperal vector
        self.temp2=ti.Vector.field(variableNum,ti.f32)#temperal vector2
        self.V=ti.Vector.field(variableNum,ti.f32)#krylov subspace
        #place
        self.dense1=self.p1.dense(ti.j,n).place(self.jacobi)
        ti.root.dense(ti.i,n).place(self.u)
        ti.root.dense(ti.i,n).place(self.b)
        ti.root.dense(ti.i,n).place(self.temp)
        ti.root.dense(ti.i,n).place(self.temp2)
        ti.root.dense(ti.ij,(self.numStep+1,self.n)).place(self.V)
    #def jacobi u b
    def build(self,jacobi,u,b):
        self.jacobi=jacobi
        self.u=u
        self.b=b

    #Jacobi@v
    @ti.kernel
    def matMul2(self,v:ti.template(),result:ti.template()):
        '''
        Av=result
        v: vector(n*numVar)
        result:vector(n*numVar)
        '''
        offcopy=self.offsets
        ti.loop_config(block_dim=128) 
        for offset,block in self.jacobi:
            #ti.block_local(u) seems cached by default
            blockVec=block+offcopy[offset]
            if (blockVec)<self.n and (blockVec)>=0:
                ti.atomic_add(result[block],self.jacobi[offset,block]@v[blockVec])


    #dot
    @ti.kernel
    def dot2(self,u:ti.template(),v:ti.template())->ti.f32:
        '''
        u dot v
        
        return result
        '''
        temp=0.0
        for i in u:
            ti.atomic_add(temp,u[i]@v[i])
        return temp
    #from u to V
    @ti.kernel
    def copy_toV(self,u:ti.template(),step:ti.i64):
         for i in u:
             self.V[step,i]=u[i]
    #from V to u
    @ti.kernel
    def copy_toT(self,u:ti.template(),step:ti.template()):
         for i in u:
             u[i]=self.V[step,i]
    #linear operator
    @ti.kernel
    def numMultAndDeduce(self,temp:ti.template(),temp2:ti.template(),num:ti.f32,pm:ti.f32):
        for i in temp:
            temp[i]=temp[i]+pm*num*temp2[i]
    #normalize krylov
    @ti.kernel
    def normalizeV(self,norm:ti.f32,step:ti.i64):
        for i in range(self.n):
            for j in ti.static(range(self.variableNum)):
                self.V[step,i][j]=self.V[step,i][j]/norm
    #py scope
    def Solve(self,n):
        
        ts=time.time()
        self.matMul2(self.u,self.temp)
        
        self.numMultAndDeduce(self.temp,self.b,1,-1)#r0
        
        g=np.zeros(self.numStep+1,dtype=np.float32)
        g[0]=np.sqrt(self.dot2(self.temp,self.temp))
        
        self.copy_toV(self.temp,0)
    
        self.normalizeV(g[0],0)#norm v0
        
        H=np.zeros((self.numStep+1,self.numStep))
        
        for step in range(self.numStep):
            self.copy_toT(self.temp2,step)#last vector:vi
            self.matMul2(self.temp2,self.temp)#temp=Avi
            for row in range(step+1):
                self.copy_toT(self.temp2,row)#temp2=v0--vi-1
                H[row,step]=self.dot2(self.temp,self.temp2)#
                self.numMultAndDeduce(self.temp,self.temp2,H[row,step],-1)
            H[step+1,step]=np.sqrt(self.dot2(self.temp,self.temp))
            self.copy_toV(self.temp,step+1)
            self.normalizeV(H[step+1,step],step+1)
        H,g=self.applyGivense(H, g)
        y=np.zeros(self.numStep)
        res=self.backSubst(H, g, y)
        for step in range(self.numStep):
            self.copy_toT(self.temp2,step)
            self.numMultAndDeduce(self.u,self.temp2,y[step],1)
        te=time.time()
        #print(te-ts)
        return res
        
        
    #py scope 
    def applyGivense(self,H,b):
        #check dim
        ti.static_assert(H.shape[0]==b.shape[0],"different shapr H and b")

        c,s=0.0,0.0
        for i in range(H.shape[1]):
            c=H[i,i]/np.sqrt(H[i,i]**2+H[i+1,i]**2)
            
            s=H[i+1,i]/np.sqrt(H[i,i]**2+H[i+1,i]**2)

            for j in range(i,H.shape[1]):
                hij=H[i,j]*c+H[i+1,j]*s
                hi1j=H[i,j]*-s+H[i+1,j]*c
                bi=b[i]*c+b[i+1]*s
                bi1=b[i]*-s+b[i+1]*c
                H[i,j]=hij
                H[i+1,j]=hi1j
                b[i]=bi
                b[i+1]=bi1
        return H,b
    #py scope
    def backSubst(self,H,b,y):
        n=y.shape[0]-1

        for i in range(n-1):
            for j in range(i):
                b[n-i]-=y[n-j]*H[n-i,n-j]
            y[n-i]=b[n-i]/H[n-i,n-i]    
        return b[n]
##################################finish define class##############################################

ti.init(arch=ti.gpu,kernel_profiler = True)


blocki=64
blockj=64
blockk=64
n=blocki*blockj*blockk
variableNum=8
stencilNum=7
jacobi=ti.Matrix.field(variableNum,variableNum,ti.f32)# rho vx vy vz T p c nv
p1=ti.root.dense(ti.i,stencilNum)
bm1=p1.dense(ti.j,n).place(jacobi)
u=ti.Vector.field(variableNum,ti.f32)
b=ti.Vector.field(variableNum,ti.f32)
ti.root.dense(ti.i,n).place(u)
ti.root.dense(ti.i,n).place(b)
offsets=ti.Vector([-blockk*blockj,-blockj,-1,0,1,blockj,blockk*blockj])

# ##i is the fastest index, then j finally k
# #filling
# #print(jacobi.shape)
# @ti.kernel
# def preRun():
#     temp=0
#     for i in range(1000000):
#         temp=temp+i
@ti.kernel
def fill(offset:ti.i32,n:ti.i64):
    for block in range(n):
        if offsets[offset]<0:
            if block <ti.abs(offsets[offset]):
                for I in ti.grouped(ti.ndrange(variableNum,variableNum)):
                    jacobi[offset,block][I]=0.0
            else:
                for I in ti.grouped(ti.ndrange(variableNum,variableNum)):
                    jacobi[offset,block][I]=ti.random(ti.f32)
                    #jacobi[offset,block][I]=(ti.random(ti.f32)-0.5)*2
        else:
            if block<ti.abs(n-offsets[offset]):
                for I in ti.grouped(ti.ndrange(variableNum,variableNum)):
                    jacobi[offset,block][I]=ti.random(ti.f32)
            else:
                for I in ti.grouped(ti.ndrange(variableNum,variableNum)):
                    jacobi[offset,block][I]=0.0
    

#in python scope
#loop diagnal
for offset in range(stencilNum):
    fill(offset,n)   

@ti.kernel
def fillVector(v:ti.template()):
    for I in v:
        for j in range(variableNum):
            v[I][j]=0.0
@ti.kernel
def fillVector2(v:ti.template()):
    for I in v:
        for j in range(variableNum):
            v[I][j]=ti.random(ti.f32)          
fillVector2(u)
fillVector2(b)

sol=GMRESSolver(offsets, 20, variableNum, n)

sol.build(jacobi, u, b)
for i in range(15):
    res=sol.Solve(n)
    print(res)


ti.profiler.print_kernel_profiler_info()
#ti.reset()