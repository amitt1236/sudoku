"""
Using the trained gan models to create digit images that we will use
to train a digit classifier model.
"""

from numpy.lib.function_base import select
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# selected trained models
selected = [[20, 34, 56, 57, 70, 71, 67, 86, 87, 88, 89, 98, 97, 91, 72],  # 0
            [62, 40, 41, 51, 56, 63, 64, 65, 66, 67, 70, 85, 86, 92, 94],  # 1
            [20, 28, 38, 27, 42, 55, 63, 71, 73, 74, 88, 87, 92, 59, 25],  # 2
            [23, 29, 25, 34, 36, 39, 53, 52, 50, 56, 68, 80, 75, 65, 57],  # 3
            [41, 42, 48, 58, 50, 56, 59, 64, 68, 74, 81, 83, 89, 86, 91],  # 4
            [32, 38, 43, 47, 53, 56, 64, 68, 76, 83, 90, 88, 93, 95, 99],  # 5
            [26, 30, 37, 45, 51, 53, 58, 57, 71, 74, 81, 83, 87, 91, 97],  # 6
            [45, 47, 50, 60, 66, 65, 63, 76, 77, 87, 85, 91, 93, 97, 98],  # 7
            [29, 27, 28, 51, 49, 56, 66, 72, 78, 81, 85, 86, 87, 97, 91],  # 8
            [34, 36, 31, 40, 44, 47, 55, 58, 72, 75, 50, 66, 62, 98, 99]  # 9
            ]

num_generate = 500

for cur in range(0, 10):
    for model_num in selected[cur]:

        dir = "/Users/amitaflalo/Desktop/sudoku/training_models/" + str(cur)
        try:
            os.mkdir(dir + "/output")
        except:
            pass

        _model = tf.keras.models.load_model(dir + "/model/" + "num-" + str(model_num) + ".h5", compile=False)

        # generator input
        noise_dim = 100
        num_examples_to_generate = num_generate
        seed = tf.random.normal([num_examples_to_generate, noise_dim])

        # getting model prediction
        predictions = _model.predict(seed)

        # saving model output
        for i in range(predictions.shape[0]):
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.imsave(dir + "/output" + "/" + str(i) + str(model_num) + ".png",
                       predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
