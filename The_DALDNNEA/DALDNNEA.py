# from . import population
# from . import species
# from . import chromosome
# from . import genome
from . import population,species,chromosome,genome, config
import time
from .config import Config
import pickle as pickle
import random
import h5py
import keras
import numpy as np
from keras import backend as k

def produce_net(ind,allocate_mutated_crossovered_weight):
    if Config.LSTM:

        inputs = keras.layers.Input(Config.input_nodes, name='input')
        #
        lstm_in = keras.layers.Embedding(input_dim=10000, output_dim=138, input_length=30)(inputs)
        x = ind.decode(lstm_in)

        x_dim = len(keras.backend.int_shape(x)[1:])
        if x_dim == 2:
            x = keras.layers.GlobalMaxPooling1D()(x)
        if x_dim != 2 and x_dim > 1:
            x = keras.layers.Flatten()(x)
        predictions = keras.layers.Dense(Config.output_nodes, activation='softmax')(x)
        net = keras.models.Model(inputs=inputs, outputs=predictions)
        net.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        return net

    else:

        inputs = keras.layers.Input(Config.input_nodes, name='input')
        x = ind.decode(inputs)
        x_dim = len(keras.backend.int_shape(x)[1:])
        if x_dim == 2:
            x = keras.layers.GlobalMaxPooling1D()(x)
        if x_dim != 2 and x_dim > 1:
            x = keras.layers.Flatten()(x)
            print("flatten is done")
        list_value = [128, 256]
        output = np.random.choice(list_value, 1)[0]
        print("\n---- output random value which is selected : ----",output)
        last_dens_layer = keras.layers.Dense(output, activation='relu')(x)
        if allocate_mutated_crossovered_weight:
            w,b = ind.getting_weights_last_layer()
            ind.showing_shape_of_last_layer_weight()
            predictions = keras.layers.Dense(Config.output_nodes, activation='softmax',weights=[w[:output,:],b])(last_dens_layer)
        else:
            predictions = keras.layers.Dense(Config.output_nodes, activation='softmax' )( last_dens_layer )
        net = keras.models.Model(inputs=inputs, outputs=predictions)
        net.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        return net


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def evaluate(Individual_pop, module_pop, num_networks, f, data, debug=None):
    # Reset fitnesses for both populations
    for module in module_pop:
        module.fitness = 0
        module.num_use = 0
    for individaul in Individual_pop:
        individaul.fitness = 0
        individaul.num_use = 0
    # Generate a new set of networks from Individuals
    networks = []
    best_model = None
    best_fit = 0
    avg_fit = 0
    for i in range(num_networks):
        ind = random.choice(Individual_pop)
        if debug is not None:
            s = '\n ------------- Evaluating --------------- \n'
            s += 'Network ' + str(i) + '\n' + 'individual: ' + str(ind) + '\n'
            debug.write(s)
        net = produce_net(ind,True)
        if debug is not None:
            s = ""
            for module in list(ind._species_indiv.values()):
                s += 'Module: ' + str(module) + '\n'
            debug.write(s)
        ind.num_use += 1
        for module in list(ind._species_indiv.values()):
            module.num_use += 1
        print('Network '+ str(i))
        print("-----------------------------  Summary of network -----------------------------\n")
        print("----------------------------                      -------------------------------\n")
        net.summary()
        fit, TheNetwork = f(net, data)
        w = TheNetwork.layers[len(TheNetwork.layers) - 1].get_weights()[0]
        b = TheNetwork.layers[len(TheNetwork.layers) - 1].get_weights()[1]
        ind.setting_weights_of_last_layer(w,b)
        # GPU memory releasing to prevent OOM( out of memory) error
        #k.clear_session()
        avg_fit += fit
        print()
        print('Network '+ str(i) + ' Fitness: ' + str(fit))
        if fit > best_fit:
          best_fit = fit
          best_model = net
        ind.fitness += fit
        for module in list(ind._species_indiv.values()):
            module.fitness += fit
    avg_fit /= num_networks
    for module in module_pop:
        if module.num_use == 0:
            if debug is not None:
                debug.write('Unused module ' + str(module.id) + '\n')
            module.fitness = avg_fit + .01
        else:
            module.fitness /= module.num_use
    for individual in Individual_pop:
        if individual.num_use == 0:
            if debug is not None:
                debug.write('Unused module ' + str(module.id) + '\n')
            individual.fitness = avg_fit + .01
        else:
          individual.fitness /= individual.num_use
    # return the highest performing single network
    return best_model



def evolve(n, pop1, pop2, num_networks, f, data, save_best, name='', report=True, debug=None):
    try:
        for g in range(n):
            print('-----Generation '+str(g)+'--------')
            if debug is not None:
                print_populations(pop1, pop2, debug)
            best_model = evaluate(pop1, pop2, num_networks, f, data, debug)
            print('-----Modules-----------')
            k = pop2.epoch(g,f, report=report, save_best=True, name=name+'_m')
            print('-----Individuals----------')
            j = pop1.epoch(g,f, report=report, save_best=False, name=name)
            if save_best:
                filename = 'best_model_' + str(g)
                if name != '':
                    filename = name + '_' + filename
                best_model.save(filename)
            if j < 0 or k < 0:
                break
    except Exception as err:
        if debug is not None:
            debug.write('ERROR\n'+str(err.args))
            debug.close()
        raise err

def print_populations(ind_pop, mod_pop, debug):
    debug.write('\n ----------------- Individual Population --------------------- \n')
    for ind in ind_pop:
        debug.write(str(ind)+'\n')
    debug.write('\n ------------------ Module Population ----------------------- \n')
    for mod in mod_pop:
        debug.write(str(mod)+'\n')
