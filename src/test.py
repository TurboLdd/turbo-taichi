# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 22:48:15 2024

@author: Administrator
"""
import taichi as ti
import meshtaichi_patcher as Patcher
import time
#ti.mesh_local(mesh.verts.f)
#ti.loop_config(block_dim=32)
"""
1st week: jst and roe on taichi, and basic test env
    test env:brick mesh with aero dynamic parameter
    realized the 1d shock wave tube
2nd week: boundary condition process
    with a high efficiency
    first idea is wove a new boundary mesh data structure, which extract the boundary mesh 
    finish wall BC pressure in/out bc
3rd week: performance tuning using mesh_local and loop_config

4th week:periodic boundary condition
"""
ti.init(arch=ti.cpu, random_seed=17)#,print_ir = True)

mesh = Patcher.load_mesh('boxx.1.hex.node', relations=["CV","CF","VF","FV","CE","CC","FC"])
#mesh_instence
numVar=3
numDim=3
mesh.verts.place({'x':ti.math.vec3})
mesh.cells.place({'gc':ti.math.vec3})
mesh.cells.place({'volume':ti.float32})
mesh.cells.place({'stateVariable':ti.math.vec3})#1D Roe scheme test rho vx vy vz H
mesh.cells.place({'v':ti.f32})
mesh.cells.place({'residual':ti.math.vec3})
mesh.cells.place({'gradient':ti.Matrix(numVar,numDim)})
mesh.faces.place({'area':ti.math.vec3})
mesh.faces.place({'flux':ti.math.vec3})
mesh.faces.place({'faceCenter':ti.math.vec3})
mesh.verts.x.from_numpy(mesh.get_position_as_numpy())
x = mesh.verts.x
gc = mesh.cells.gc
volume = mesh.cells.volume
StateVariable = mesh.cells.stateVariable
Residual= mesh.cells.residual
area= mesh.faces.area
flux= mesh.faces.flux
faceCenter= mesh.faces.faceCenter
#index=mesh.faces.index
#vertID=ti.field(ti.int32,shape=())
###############################Geometry Processing####################################  
@ti.kernel
def CalculateCellCenterCoord():
    '''
    Calculate center of a cell
    '''
    for c in mesh.cells:
        for v in c.verts:
                c.gc+=v.x
        c.gc/=8
@ti.kernel
def CalculateFaceCenterCoord():
    '''
    Calculate center of a cell
    '''
    for f in mesh.faces:
        for v in f.verts:
            f.faceCenter+=v.x
        f.faceCenter/=4
@ti.func
def flip(gc:ti.template(),fc:ti.template(),norm:ti.template()):
    return 1.0 if norm.dot((gc-fc))>0 else -1.0
        
     
        
@ti.kernel
def CalculateFaceArea():
    '''
    Calculate face area

    '''
    vec=ti.Matrix.zero(ti.f32, 4, numDim)
    for f in mesh.faces:
        for i in range(f.verts.size):#quad
            vec[i,:]=f.verts[i].x
        dv1=vec[0,:]-vec[1,:]
        dv2=vec[1,:]-vec[2,:]
        dv3=vec[2,:]-vec[3,:]
        dv4=vec[3,:]-vec[0,:]
        a1=0.5*dv1.cross(dv2)
        a2=0.5*dv3.cross(dv4)
        if a1.dot(a2)<0:
            f.area=(a1-a2)
        else:
            f.area=(a1+a2)
        
            
@ti.kernel
def CalculateVolume():
    '''
    Calculate Hexahadral Volume

    '''
    for c in mesh.cells:
        for f in c.faces:
            d=-f.verts[0].x.dot(f.area)
            d=ti.abs((c.gc.dot(f.area)+d)/f.area.norm()/3)
            c.volume+=f.area.norm()*d

            
##############################Roe Flux####################################            
  
@ti.kernel
def Roe_flux():
    '''
    Currently on cell centrered scheme only 
    because the edge relationship was not ready for 
    Hex Mesh
    '''
    gamma=1.4
    #numVar=3 global no need
    for f in mesh.faces:
        f.flux=0
        if f.cells.size!=2:
            continue
        roeAve=ti.Vector.zero(ti.f32,numVar)
        vecF=ti.Vector.zero(ti.f32,numVar)
        vVectL=f.c[0].stateVariable[1:numDim+1]
        vVectR=f.c[1].stateVariable[1:numDim+1]
        rhoL=f.c[0].stateVariable[0]
        rhoR=f.c[1].stateVariable[0]
        pL=rhoL*(gamma-1)/gamma*(f.c[0].stateVariable[-1]-vVectL.norm()*vVectL.norm()*0.5)
        pR=rhoR*(gamma-1)/gamma*(f.c[1].stateVariable[-1]-vVectR.norm()*vVectR.norm()*0.5)
        deltaP=pL-pR
        deltaRho=rhoL-rhoR
        deltaVvec=vVectL-vVectR
        deltaV=vVectL.norm()-vVectR.norm()
        roeAve[0]=ti.sqrt(rhoL*rhoR)
        roeAveV=roeAve[1:1+numDim].norm()
        
        for i in ti.static(range(1,numVar-1)):
            roeAve[i]=(rhoL*f.c[0].stateVariable[i]+rhoR*f.c[1].stateVariable[i])/(ti.sqrt(rhoL)+ti.sqrt(rhoR))
        cTilda=ti.sqrt((gamma-1)*(roeAve[numVar-1]-roeAveV*roeAveV*0.5))
        mulOP1=(roeAveV-cTilda)*(deltaP-roeAve[0]*cTilda*deltaV)*0.5/(cTilda*cTilda)
        mulOP2=roeAveV*(deltaRho-deltaP/(cTilda*cTilda))
        mulOP3=(roeAveV+cTilda)*(deltaP+roeAve[0]*cTilda*deltaV)*0.5/(cTilda*cTilda)
        areaNorm=f.area/f.area.norm()
        vecF[0]+=(mulOP1*1+mulOP2*1+mulOP3*1)
        vecF[1:1+numDim]+=(mulOP1*(roeAveV-areaNorm*cTilda)\
                           +mulOP2*roeAve[1:1+numDim]+roeAve[0]*roeAveV*(deltaVvec-deltaV*areaNorm)\
                               +mulOP3*(roeAve[1:1+numDim]+areaNorm*cTilda))
        vecF[-1]+=(mulOP1*(roeAve[-1]-cTilda*roeAveV)+mulOP2*roeAveV*roeAveV*0.5\
                   +roeAve.dot(deltaVvec)-roeAveV*deltaV+mulOP3*(roeAve[-1]+cTilda*roeAve))
        f.flux+=vecF
        f.flux[1:]+=(f.c[0].stateVariable[0]*f.c[0].stateVariable[1:]+f.c[1].stateVariable[0]*f.c[1].stateVariable[1:])*0.5
        f.flux[0]+=(f.c[0].stateVariable[0]+f.c[1].stateVariable[0])*0.5
        
##############################JST Flux####################################  
@ti.func
def PseudoLaplacionStateVariable(cell):
    lap=ti.Vector.zero(3,dtype=ti.f32)#need to be a global variable
    for c in cell.cells:
        lap+=(c.stateVariable-cell.stateVariable)
        return lap
@ti.func
    
@ti.func
def thetaCorrection():
    pass    


@ti.kernel
def JST_flux():
    
    pass
##############################Residual and Jacobin####################################  
@ti.kernel
def Compute_Residual():
    for c in mesh.cells:
        for f in c.face:
            f.residual+=(f.flux*flip(c.gc,f.faceCenter,f.area))
##############################Gradient################################
@ti.kernel            
def Gradient_LeastSquare():
    for c in mesh.cells:
        nNeighbor=c.cells.size()
        b=ti.Vector.zero(nNeighbor,numVar)
        D=ti.Matrix.zero(nNeighbor,numDim)
            #g=D`b
        for j in ti.static(range(nNeighbor)):
            b[j,:]=c.cells[j].stateVariable-c.stateVariable
            D[j,:]=c.cells[j].gc-c.gc
        c.gradient=(D.transpose()@D@b).transpose()
@ti.kernel
def Gradient_GreenGauss():
    for c in mesh.cells:     
        for j in ti.static(range(c.cells.size)):
            for f in c.cells[j]:
                if f not in c.faces:
                    continue
                else:
                    c.gradient+=c.cells[j].stateVariable*f.area*flip(c.gc,f.faceCenter,f.area)
                    break
        c.gradient/=c.volume
t=time.time()   
CalculateCellCenterCoord()
CalculateFaceArea()
CalculateVolume()
Roe_flux()
t1 = time.time()
print(t1-t)

ti.reset()