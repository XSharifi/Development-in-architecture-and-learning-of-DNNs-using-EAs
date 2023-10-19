# coding: utf-8

from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from The_DALDNNEA import DALDNNEA, config, population, chromosome, visualize
import numpy as np
import math
import keras
import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))



(x_train_all, y_train_all), (x_test, y_test) = cifar10.load_data()
y_train_all = y_train_all[:,0]
y_test = y_test[:,0]

x_train_all = np.reshape(x_train_all, (x_train_all.shape[0], 32, 32, 3)).astype('float32') / 255
x_test = np.reshape(x_test, (x_test.shape[0], 32, 32, 3)).astype('float32') / 255
y_train_all_one_hot = keras.utils.np_utils.to_categorical(y_train_all)
y_test = keras.utils.np_utils.to_categorical(y_test)

num_categories = 10
category_count = np.zeros(num_categories)

num_training = 42500

x_train = []
y_train = []
x_val = []
y_val = []


index = np.array(range(len(x_train_all)))
np.random.shuffle(index)
x_train_all = x_train_all[index]
y_train_all = y_train_all[index]
y_train_all_one_hot = y_train_all_one_hot[index]
for i in range(len(index)):
    if category_count[y_train_all[i]] < num_training/num_categories:
        x_train.append(x_train_all[i])
        y_train.append(y_train_all_one_hot[i])
        category_count[y_train_all[i]] += 1
    else:
        x_val.append(x_train_all[i])
        y_val.append(y_train_all_one_hot[i])
print(category_count)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)

data = [x_train, y_train, x_val, y_val, x_test, y_test]
print("data shapes")
print("  x train:", x_train.shape)
print("  y train:", y_train.shape)

print("  x val:", x_val.shape)
print("  y val:", y_val.shape)

print("  x test:", x_test.shape)
print("  y test:", y_test.shape)


def fitness(network, data):
    batch_size = 64
    num_epoch = 8

    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

    datagen.fit(data[0])
    network.fit_generator(datagen.flow(data[0], data[1], batch_size=batch_size),
                        steps_per_epoch=data[0].shape[0] // batch_size,
                        validation_data=(data[2], data[3]),
                        epochs=num_epoch, verbose=1, max_queue_size=100)
    loss, acc = network.evaluate(data[2], data[3])
    return acc, network



def stroing_list_data(file_name,status_list):

    generation = [i for i in range( len( status_list[0] ) )]
    #the best fitness of every generation is stored in fitness list
    fitness = [fit for fit in status_list[0]]
    avg_pop = [avg for avg in status_list[1]]


    with open( str(file_name)+"_fitness.txt", "w" ) as f:
        for s in fitness:
            f.write( str( s ) + "\n" )
        f.close()

    with open( str(file_name)+"_avg_pop.txt", "w" ) as f:
        for s in avg_pop:
            f.write( str( s ) + "\n" )
        f.close()


def retrieving_list(file_name):

    fitness = list()
    with open( str(file_name)+"_fitness.txt", "r" ) as f:
        for line in f:

            x = (float(line.strip())*100)
            v = math.modf(x)
            value = (math.modf((v[0])*100))[1]/100 + v[1]
            fitness.append(value)

    avg_pop = list()
    with open( str(file_name)+"_avg_pop.txt", "r" ) as f:
        for line in f:
            x = (float(line.strip())*100)
            v = math.modf(x)
            value = (math.modf((v[0])*100))[1]/100 + v[1]
            avg_pop.append(value)


def evolve(n, debugging=False):
    if debugging:
        debug = open("debug.txt", "w")
    else:
        debug = None
    config.load('CIFAR10.txt')
    module_pop = population.Population(15, chromosome.ModuleChromo, debug=debug)
    ind_pop = population.Population(10, chromosome.IndividualChromo, module_pop, debug=debug)
    DALDNNEA.evolve(n, ind_pop, module_pop, 25, fitness, data, save_best=True, name='CIFAR10', debug=debug)

    stroing_list_data("CIFAR10",module_pop.stats)
    stroing_list_data("CIFAR10",ind_pop.stats)


def main():
    evolve(25,True)
    fitness_value,avg_pop = retrieving_list("CIFAR10_module_pop")
    visualize.plot_stats_with_mathplot(fitness_value,avg_pop, name="CIFAR10mod_")
    fitness_value,avg_pop = retrieving_list("CIFAR10_blueprint_pop")
    visualize.plot_stats_with_mathplot(fitness_value,avg_pop, name="CIFAR10bp_")


if __name__ == "__main__":
    main()
