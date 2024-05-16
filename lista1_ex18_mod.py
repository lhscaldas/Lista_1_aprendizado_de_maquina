import numpy as np

pxy=[[1/3,1/3],[0,1/3]] # p(x,y)

px0 = 2/3
px1 = 1/3
py0= 1/3
py1 = 2/3

py0_x0 = 1/3/px0
py1_x0 = 1/3/px0
py0_x1 = 0/px1
py1_x1 = 1/3/px1
print(py0_x0,py1_x0,py0_x1, py1_x1)

px0_y0 = 1/3/py0
px1_y0 = 0/py0
px0_y1 = 1/3/py1
px1_y1 = 1/3/py1
print(px0_y0,px1_y0,px0_y1, px1_y1)

#item a
Hx= - (px0*np.log2(px0) + px1*np.log2(px1))
print(f"item a) H[x]={Hx:.4f}")

# #item b
Hy=- (py0*np.log2(py0) + py1*np.log2(py1))
print(f"item b) H[y]={Hy:.4f}")

# #item c
Hy_x=-((1/3)*np.log2(py0_x0)+(1/3)*np.log2(py1_x0)+(1/3)*np.log2(py1_x1))
print(f"item c) H[y|x]={Hy_x:.4f}")

# #item d
Hx_y= -((1/3)*np.log2(px0_y0)+(1/3)*np.log2(px0_y1)+(1/3)*np.log2(px1_y1))
print(f"item d) H[x|y]={Hx_y:.4f}")

#item e
Hxy=Hy_x+Hx
print(f"item e) H[x,y]={Hxy:.4f}")

#item f 
Ixy=Hx-Hx_y
print(f"item f) I[x,y]={Ixy:.4f}")
