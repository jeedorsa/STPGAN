from keras.models import load_model
from keras.models import model_from_json
import json
import numpy as np
json_file = open('Models/model_Corona.json', 'r')
#json_file = open('Models/model_Internas.json', 'r')
#json_file = open('Models/model_Superficiales.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("Weights/modelCorona.h5")
#loaded_model.load_weights("Weights/modelInternas.h5")
#loaded_model.load_weights("Weights/modelSuperficiales.h5")
print("Model load Succesfully!!")
r, c = 10, 100
latent_dim = 100
noise = np.random.normal(0, 1, (r * c, latent_dim))
gen_imgs = loaded_model.predict(noise)
np.save('ArtificialImages/imagenes_artificiales_corona.npy', gen_imgs)
#np.save('ArtificialImages/imagenes_artificiales_internas.npy', gen_imgs)
#np.save('ArtificialImages/imagenes_artificiales_superficiales.npy', gen_imgs)
print("Images saved correctly!")
