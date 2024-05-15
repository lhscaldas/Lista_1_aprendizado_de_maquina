import numpy as np

pxy=[[1/3,1/3],[0,1/3]] # p(x,y)

px=[2/3,1/3] # p(x)
py=[1/3,2/3] # p(y)

px_y0=[(1/3)/py[0],(0)/py[0]] # p(x|y=0)
px_y1=[(1/3)/py[1],(1/3)/py[1]] # p(x|y=1)
py_x0=[(1/3)/px[0],(1/3)/px[0]] # p(y|x=0)
py_x1=[(0)/px[1],(1/3)/px[1]] # p(y|x=1)

#item a
Hx=-(px[0]*np.log(px[0])+px[1]*np.log(px[1]))
print(f"item a) H[x]={Hx:.4f}")

#item b
Hy=-(py[0]*np.log(py[0])+py[1]*np.log(py[1]))
print(f"item b) H[y]={Hy:.4f}")

#item c
Hy_x=-(
       pxy[0][0]*np.log(py_x0[0])+
       pxy[0][1]*np.log(py_x0[1])+
       #pxy[1][0]*np.log(py_x1[0])+
       pxy[1][1]*np.log(py_x1[1])
       )
print(f"item c) H[y|x]={Hy_x:.4f}")

#item d
Hx_y=-(
       pxy[0][0]*np.log(px_y0[0])+
       #pxy[1][0]*np.log(px_y0[1])+
       pxy[0][1]*np.log(px_y1[0])+
       pxy[1][1]*np.log(px_y1[1])
       )
print(f"item d) H[x|y]={Hx_y:.4f}")

#item e
Hxy=Hy_x+Hx
print(f"item e) H[x,y]={Hxy:.4f}")

#item f 
Ixy=Hx-Hx_y
print(f"item f) I[x,y]={Ixy:.4f}")
