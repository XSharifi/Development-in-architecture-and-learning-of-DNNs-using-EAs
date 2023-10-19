from keras.datasets import reuters
from keras import preprocessing
from The_DALDNNEA import DALDNNEA, config, population, chromosome, visualize
import numpy as np
import keras
import math

#### A new version to evolve RNN for Penn Treebank dataset will be updated.

max_words = 10000
(x_train_all, y_train_all), (x_test, y_test) = reuters.load_data(num_words=max_words, test_split=0.2)
print(len(x_train_all), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train_all) + 1
print(num_classes, 'classes')


x_train_all = preprocessing.sequence.pad_sequences(x_train_all, maxlen=30)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=30)[:2240]
y_train_all = keras.utils.to_categorical(y_train_all, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)[:2240]


index = np.array(range(len(x_train_all)))
np.random.shuffle(index)
index_train = index[:7862]
index_val = index[7862:]
x_train = x_train_all[index_train][:7840]
y_train = y_train_all[index_train][:7840]
x_val = x_train_all[index_val]
y_val = y_train_all[index_val]
print('x_train shape', x_train.shape)
print('x_val shape', x_val.shape)
print('x_test shape', x_test.shape)

data = [x_train, y_train, x_val, y_val, x_test, y_test]



def fitness(network, data):
    network.fit(data[0], data[1],  epochs=5, batch_size=32)
    loss, acc = network.evaluate(data[2], data[3], batch_size=32)
    return acc


def storing_list_data(file_name,status_list):

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

    return fitness,avg_pop


def evolve(n, debugging=False):
    if(debugging):
        debug = open("debug.txt", "w")
    else:
        debug = None
    config.load('Penn_Treebank.txt')

    module_pop = population.Population(10, chromosome.ModuleChromo, debug=debug)
    ind_pop = population.Population(10, chromosome.IndividualChromo, module_pop, debug=debug)
    DALDNNEA.evolve(n, ind_pop, module_pop, 10, fitness, data, save_best=True, name='reuters', debug=debug)

    storing_list_data("Penn_Treebank_module_pop",module_pop.stats)
    storing_list_data("Penn_Treebank_Ind_pop",ind_pop.stats)


def main():

    evolve(10,True)
    fitness_value, avg_pop = retrieving_list("Penn_Treebank_module_pop")
    visualize.plot_stats_with_mathplot(fitness_value,avg_pop, name="Penn_Treebank_mod_")
    fitness_value, avg_pop = retrieving_list("Penn_Treebank_ind_pop")
    visualize.plot_stats_with_mathplot(fitness_value,avg_pop, name="Penn_Treebank_ind_")



if __name__ == "__main__":
    main()
