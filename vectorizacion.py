import numpy as np
data=np.load('ArtificialImages/imagenes_artificiales_corona.npy')  
#data=np.load('ArtificialImages/imagenes_artificiales_superficiales.npy')  
#data=np.load('ArtificialImages/imagenes_artificiales_internas.npy')  

vec=np.zeros(200)
img=np.zeros((1000,200)) 
count=0
k=0
while(k<1000):
    x1=data[k, :, :, 0]
    for i in range (0,16):
        for j in range (0,16):
            if count<200:
                vec[count]=x1[i][j]
                count=count+1
            else:
                count=count+1
    img[k]=vec
    k=k+1
    count=0    
np.savetxt('ArtifcialSignals/coronaintetica.txt',img, delimiter='\t',fmt='%f')      
#np.savetxt('ArtifcialSignals/superficialesintetica.txt',img, delimiter='\t',fmt='%f')    
#np.savetxt('ArtifcialSignals/internasintetica.txt',img, delimiter='\t',fmt='%f')    