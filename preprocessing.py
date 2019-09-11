import numpy as np
#data=np.loadtxt('DatasetReal/internas.txt', delimiter= '\t') # 554 muest5ras
#data=np.loadtxt('DatasetReal/corona.txt', delimiter= '\t')  #864 muestras
data=np.loadtxt('DatasetReal/superficiales.txt', delimiter= '\t')  # 571 muestras
img=np.zeros((571, 16,16)) 
for k in range (0,571):
    data3=data[k]
    val=min(data3)
    val=abs(val)
    for i in range(0,200):
        data3[i]=data3[i]+val      
    val=max(data3)
    for i in range(0,200):
        data3[i]=data3[i]/val   
    count=0    
    for i in range(0,16):  
        for j in range (0,16):
            if count<200:
                img[k][i][j]=data3[count]
                count=count+1
            else:
                img[k][i][j]=0
                count=count+1
#np.save('DatasetImages/coronaimagenes.npy', img)
#np.save('DatasetImages/internasimagenes.npy', img)
np.save('DatasetImages/superficialesimagenes.npy', img)               
                
                

