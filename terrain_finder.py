import joblib
import os
import numpy as np
import tensorflow as tf
import keras as k
def terrain():
    imageSize=(224,224)
    class_name=['Grassy', 'Marshy', 'Rocky', 'Sandy']
    modelNew=joblib.load("/home/balaji/Documents/SIH/SIH Backend/terrain-system/website/terrain1.pkl")
    testImage=k.preprocessing.image_dataset_from_directory(r"/home/balaji/Documents/SIH/SIH Backend/terrain-system/uploads/a",seed=42,image_size=imageSize)
    index=modelNew.predict(testImage)
    print(index)
    predictedClass=class_name[np.argmax(index)]
    score=tf.nn.softmax(index[0])
    if(predictedClass=='Rocky'):
        return predictedClass+":        A rocky terrain is distinguished by its rugged, unlevel topography and profusion of rocks, boulders, and stones dispersed across the region. Many different forms of rock formations, including cliffs, canyons, and outcrops, can be found in rocky locations. These formations are the product of tectonic forces, weathering, and millions of years of geological activity."
    elif(predictedClass=='Marshy'):
        return predictedClass+":    Marshy terrain is distinguished by wet or saturated soil. The ground is frequently moist, and there may be standing water on the surface in several occasions."
    elif(predictedClass=='Sandy'):
        return predictedClass+": Sandy terrain, which is common in deserts, coastal locations, and certain interior regions, has particular qualities that distinguish it from other types of landscapes. Sandy terrain is mostly made up of sand particles, which are larger than silt but smaller than gravel. Sand is frequently loose and grainy in texture."
    elif(predictedClass=='Grassy'):
        return predictedClass+": Grassy terrains are distinguished by their relatively flat surface, which distinguishes them from hilly or densely forested areas. The terrain might range from gently sloping hills to large, nearly entirely flat areas."



