import random, math
from .config import Config
from . import genome
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras import backend
import os


class Chromosome(object):
    _id = 0
    def __init__(self, parent1_id, parent2_id, gene_type):

        self._id = self.__get_new_id()

        self._gene_type = gene_type
        self._genes = []

        self.fitness = None
        self.num_use = None
        self.species_id = None
        self.parent1_id = parent1_id
        self.parent2_id = parent2_id

    genes      = property(lambda self: self._genes)
    id         = property(lambda self: self._id)

    @classmethod
    def __get_new_id(cls):
        cls._id += 1
        return cls._id

    def mutate(self):
        """ Mutates this chromosome """
        raise NotImplementedError

    def crossover(self, other):

        assert self.species_id == other.species_id, 'Different parents species ID: %d vs %d' \
                                                         % (self.species_id, other.species_id)

        if self.fitness > other.fitness:
            parent1 = self
            parent2 = other
        elif self.fitness == other.fitness:
            if len(self._genes) > len(other._genes):
                parent1 = other
                parent2 = self
            else:
                parent1 = self
                parent2 = other
        else:
            parent1 = other
            parent2 = self

        # creates a new child
        child = self.__class__(self.id, other.id, self._gene_type)

        child._inherit_genes(parent1, parent2)

        child.species_id = parent1.species_id

        return child

    def _inherit_genes(child, parent1, parent2):
        """ Applies the crossover operator. """
        raise NotImplementedError

    # compatibility function
    def distance(self, other):
        """ Returns the distance between this chromosome and the other. """
        if len(self._genes) > len(other._genes):
            chromo1 = self
            chromo2 = other
        else:
            chromo1 = other
            chromo2 = self

        return chromo1._genes-chromo2._genes

    def size(self):
        """ Defines chromosome 'complexity': number of hidden nodes plus
            number of enabled connections (bias is not considered)
        """
        return len(self._genes)
    def copy(self):
        """
        Default copy method on Chromo is not an actual copy, must be overriden for shallow copy
        """
        return self

    def __cmp__(self, other):
        return cmp(self.fitness, other.fitness)

    def __lt__(self, other):
        return self.fitness <  other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __str__(self):
        s = "ID: "+str(self._id)+" Species: "+str(self.species_id)+"\nGenes:"
        for ng in self._genes:
            s += "\n\t" + str(ng)
        return s

class IndividualChromo(Chromosome):
    """
    A chromosome for deep neural network IndividualChromo.
    """
    def __init__(self, parent1_id, parent2_id, module_pop, active_params={}, gene_type='MODULE'):
        super(IndividualChromo, self).__init__(parent1_id, parent2_id, gene_type)
        self._species_indiv = {}
        self._all_params = {
                "learning_rate": [.0001, .1],
                "momentum": [.68,.99],
                "hue_shift": [0, 45],
                "sv_shift": [0, .5],
                "sv_scale": [0, .5],
                "cropped_image_size": [26,32],
                "spatial_scaling": [0,0.3],
                "horizontal_flips": [True, False],
                "variance_norm": [True, False],
                "nesterov_acc": [True, False],
                }
        self._active_params = active_params
        self._module_pop = module_pop

        self._weights_of_last_layer = np.zeros(shape = (256,Config.output_nodes)) # used for storing weight of last layer in the architecture
        self._bias_last_layer = list()
        self._randomly_trained_for_the_first_time = False
        self._status_fitness = "uncertain"
        self.setting_weights_last_layer()

    def showing_shape_of_last_layer_weight(self):

        print("\n ------->   shape of last layer of ind :", np.shape(self._weights_of_last_layer)[0])


    def setting_weights_of_last_layer(self,w,b):

        print("\n current shape of last layer of ind :",np.shape(self._weights_of_last_layer)[0])
        print("\n new shape of weight of the last layer that be merged with :",np.shape(w)[0])

        if np.shape(self._weights_of_last_layer)[0] > np.shape(w)[0]:
            w_shape = int(np.shape(w)[0])
            current_shape = int(np.shape(self._weights_of_last_layer)[0])
            my_array = np.concatenate(
                (w,self._weights_of_last_layer[np.arange(w_shape, current_shape), :]))

            del self._weights_of_last_layer
            del self._bias_last_layer

            self._weights_of_last_layer = my_array
            self._bias_last_layer = b

            print("\n After merging on current shape of last layer of ind : current is bigger that new one",
                  np.shape(self._weights_of_last_layer)[0])

        elif np.shape(self._weights_of_last_layer)[0] == np.shape(w)[0]:
            del self._weights_of_last_layer
            del self._bias_last_layer
            self._weights_of_last_layer = w
            self._bias_last_layer = b

            print("\n After merging on current shape of ind of the last layer : they are equal",
                  np.shape(self._weights_of_last_layer)[0])

        else:
            print("\n --------  it is not acceptable because current must be bigger than new weight")




    def setting_weights_last_layer(self):

        w = np.array([0.01*np.random.randn() for i in range(2560)]).reshape(256,10)
        b = np.array([0.01*np.random.randn() for i in range(10)])
        self._weights_of_last_layer = w
        self._bias_last_layer = b

    def getting_weights_last_layer(self):
        return self._weights_of_last_layer,self._bias_last_layer


    def weight_crossover(self,othre_ind):

        alfa = random.random()
        if alfa < 0.66:
            alfa = 1 - alfa
        beta = 1 - alfa
        w = np.array( [(alfa*x + beta*y) / 2 for x, y in zip( self._weights_of_last_layer.ravel(),
                                    othre_ind._weights_of_last_layer.ravel() )] ) \
            .reshape( 256, Config.output_nodes )

        b = np.array( [(x + y) / 2 for x, y in zip( self._bias_last_layer.ravel(), othre_ind._bias_last_layer.ravel() )] )

        return w,b

    def weight_mutation(self):

        dim = np.shape(self._weights_of_last_layer)

        rate_of_weight_mutating = np.random.choice(int(dim[0])*int(dim[1]),1)

        for _ in range( int(0.2 * (int(dim[0])*int(dim[1]))) ):
            # choosing representation mutate of row and column of the last layer of model
            # the last layer weights can be presented as a two dimension matrix (weights of last layer)
            rep_mutate_row = np.random.choice( dim[0], 1 )
            rep_mutate_column = np.random.choice( dim[1], 1 )
            random_value = 0.01*np.random.randn()
            self._weights_of_last_layer[rep_mutate_row][rep_mutate_column] += random_value

        rate_of_bias_mutating = np.random.choice(int(np.shape(self._bias_last_layer)[0]),1)
        for _ in range(int(0.2 * int(np.shape(self._bias_last_layer)[0]))):
            #also, process of mutating is done for bias
            rep_mutate_bias = 0.01*np.random.choice(int(np.shape(self._bias_last_layer)[0]),1)
            self._bias_last_layer[rep_mutate_bias] += np.random.randn()



    def crossover(self, other):
        """ Crosses over parents' chromosomes and returns a child. """

        # This can't happen! Parents must belong to the same species.
        assert self.species_id == other.species_id, 'Different parents species ID: %d vs %d' \
                                                         % (self.species_id, other.species_id)

        if self.fitness > other.fitness:
            parent1 = self
            parent2 = other
        elif self.fitness == other.fitness:
            if len(self._genes) > len(other._genes):
                parent1 = other
                parent2 = self
            else:
                parent1 = self
                parent2 = other
        else:
            parent1 = other
            parent2 = self

        # creates a new child
        child = self.__class__(self.id, other.id, self._module_pop, self._active_params)

        child._inherit_genes(parent1, parent2)

        child.species_id = parent1.species_id

        return child


    def __get_species_indiv(self):
        # This method chooses an individule for each unique Module Species Gene
        for g in self._genes:
            # Check that each gene actually points to an existing species
            if not self._module_pop.has_species(g.module):
                raise ValueError('Missing species at decode: '+str(g.module))
        self._species_indiv = {}
        for g in self._genes:
            if g.module.id not in self._species_indiv:
                self._species_indiv[g.module.id] = random.choice(g.module.members)

    def decode(self, inputs):
        # create keras Model from component modules
        self.__get_species_indiv()
        next = inputs
        for g in self._genes:
            mod = self._species_indiv[g.module.id].decode(next)
            next = mod(next)
        return next

    def distance(self, other):
        # measure distance between two individual chromosomes for speciation purpose
        dist = 0
        if len(self._genes) > len(other._genes):
            chromo1 = self
            chromo2 = other
        else:
            chromo1 = other
            chromo2 = self
        dist += (len(chromo1._genes)-len(chromo2._genes))*Config.excess_coefficient
        chromo1_species = []
        chromo2_species = []
        for g in chromo1._genes:
            chromo1_species.append(g.module.id)
        for g in chromo2._genes:
            chromo2_species.append(g.module.id)
        for id in chromo1_species:
            if id in chromo2_species:
                chromo2_species.remove(id)
            else:
                dist += Config.disjoint_coefficient
        return dist

    def _inherit_genes(child, parent1, parent2):
        """ Applies the crossover operator. """
        assert(parent1.fitness >= parent2.fitness)

        # Crossover layer genes
        for i, g1 in enumerate(parent1._genes):
            try:
                # matching node genes: randomly selects the neuron's bias and response
                child._genes.append(g1.get_child(parent2._genes[i]))
            except IndexError:
                # copies extra genes from the fittest parent
                child._genes.append(g1.copy())
            except TypeError:
                # copies disjoint genes from the fittest parent
                child._genes.append(g1.copy())
        # Insure inherited module genes point to actual species
        for g in child._genes.copy():
            if not (child._module_pop.has_species(g.module)):
                new_species = child._module_pop.get_species()
                fails = 0
                while True:
                    try:
                        g.set_module(new_species)
                        break
                    except TypeError:
                        fails += 1
                        new_species = child._module_pop.get_species()
                        if fails > 500:
                            child._genes.remove(g)
                            break


    def mutate(self):
        """ Mutates this chromosome """

        r = random.random
        if r() < Config.prob_addmodule:
            self._mutate_add_module()
        elif len(self._active_params) > 0:
            for param in list(self._active_params.keys):
                if r() < 0.5:
                    self._active_params[param] = random.choice(self._all_params[param])
        g = random.choice(self._genes)
        if r() < Config.prob_switchmodule:
            new_species = self._module_pop.get_species()
            fails = 0
            while True:
                try:
                    g.set_module(new_species)
                    break
                except TypeError:
                    fails += 1
                    new_species = self._module_pop.get_species()
                    if fails > 500:
                        self._genes.remove(g)
                        break
        return self

    def _mutate_add_module(self):
        """ Adds a module to the IndividualChromo"""
        ind = random.randint(0,len(self._genes))
        if Config.conv and ind > 0:
            valid_mod = False
            while not valid_mod:
                module = self._module_pop.get_species()
                mod_type = module.members[0].type
                if self._genes[ind-1].type != 'CONV' and mod_type == 'CONV':
                    valid_mod = False
                else:
                    valid_mod = True
        else:
            module = self._module_pop.get_species()
            mod_type = module.members[0].type
        self._genes.insert(ind, genome.ModuleGene(None, module, mod_type))

    def copy(self):
        for g in self._genes:
            if not (self._module_pop.has_species(g.module)):
                new_species = self._module_pop.get_species()
                fails = 0
                while True:
                    try:
                        g.set_module(new_species)
                        break
                    except TypeError:
                        fails += 1
                        new_species = self._module_pop.get_species()
                        if fails > 500:
                            self._genes.remove(g)
                            break
        return self

    @classmethod
    def create_initial(cls, module_pop):
        c = cls(None, None, module_pop)
        n = random.randrange(2,5)
        mod = module_pop.get_species() # get a random module individual and then return its species
        mod_type = mod.members[0].type
        c._genes.append(genome.ModuleGene(None, mod, mod_type))
        for i in range(1,n):
            c._genes.append(genome.ModuleGene(None, mod, mod_type))
        if Config.conv and Config.prob_addconv < 1:
            conv_ind = []
            for i in range(len(c._genes)):
                if c._genes[i].type == 'CONV':
                    conv_ind.append(i)
            conv_mods = []
            for j in range(len(conv_ind)-1, 0, -1):
                mod = c._genes.pop(conv_ind[j])
                conv_mods.append(mod)
            for k in range(len(conv_mods)):
                c._genes.insert(0,conv_mods.pop())
        return c


class ModuleChromo(Chromosome):
    # TODO: remove most of the connection sorts
    """
    A chromosome for deep neural network "modules" which consist of a small
    number of layers and their associated hyperparameters.
    """
    def __init__(self, parent1_id, parent2_id, gene_type='DENSE'):
        super(ModuleChromo, self).__init__(parent1_id, parent2_id, gene_type)
        self._connections = []

    type = property(lambda self: self._gene_type)
    def decode(self, inputs):
        """
        Produces Keras Model of this module
        """
        inputdim = backend.int_shape(inputs)[1:]
        if self._gene_type == 'LSTM':
            inputlayer = Input(inputdim)
        else:
            inputlayer = Input(shape=inputdim)
        mod_inputs = {0: inputlayer}
        self.__connection_sort()
        for conn in self._connections:
            conn.decode(mod_inputs, self._gene_type)
        mod = Model(inputs=inputlayer, outputs=mod_inputs[-1])
        return mod

    def distance(self, other):
        dist = 0
        if len(self._genes) > len(other._genes):
            chromo1 = self
            chromo2 = other
        else:
            chromo1 = other
            chromo2 = self
        dist += (len(chromo1._connections)-len(chromo2._connections))*Config.excess_coefficient
        if (chromo1._gene_type != chromo2._gene_type):
            return (dist+1000)
        else:
            if len(chromo1._connections) > len(chromo2._connections):
                for i, conn in enumerate(chromo2._connections):
                    for j in range(len(chromo1._connections[i]._in)):
                        if chromo1._connections[i]._in[j] not in list(conn._in):
                            dist += Config.connection_coefficient
            else:
                for i, conn in enumerate(chromo1._connections):
                    for j in range(len(chromo2._connections[i]._in)):
                        if chromo2._connections[i]._in[j] not in list(conn._in):
                            dist += Config.connection_coefficient

            size_diff = 0
            for j, layer in enumerate(chromo2._genes):
               size_diff += math.log2(chromo1._genes[j]._size) - math.log2(layer._size)
            dist += abs(size_diff)*Config.size_coefficient
        return dist

    def _inherit_genes(child, parent1, parent2):
        """ Applies the crossover operator. """
        assert(parent1.fitness >= parent2.fitness)

        if (parent1 == parent2):
            child._genes = parent1._genes.copy()
            child._connections = parent1._connections.copy()
            return

        parent1.__connection_sort()
        # Crossover layer genes
        for i, g1 in enumerate(parent1._genes):
            try:
                child._genes.append(g1.get_child(parent2._genes[i]))
            except IndexError:
                child._genes.append(g1.copy())
            except TypeError:
                child._genes.append(g1.copy())
        child._connections = parent1._connections.copy()
        child.__connection_sort()
        for i, conn in enumerate(parent1._connections):
            for j, inlayer in enumerate(conn.input):
                if inlayer.type != 'IN':
                    try:
                        if len(parent1._genes) == 1:
                            child._connections[i]._in[j] = child._genes[0]
                        else:
                            ind = parent1._genes.index(inlayer)
                            child._connections[i]._in[j] = child._genes[ind]
                    except ValueError:
                        print(inlayer.type)
                        print(conn.input[j].type)
                        print(str(conn))
                        for g in parent1._genes:
                            print(str(g))
            for k, outlayer in enumerate(conn.output):
                if outlayer.type != 'OUT':
                    try:
                        if len(parent1._genes) == 1:
                            child._connections[i]._out[k] = child._genes[0]
                        else:
                            ind = parent1._genes.index(outlayer)
                            child._connections[i]._out[k] = child._genes[ind]
                    except ValueError:
                        # below code used for debugging, this exception isn't actually handled
                        print(outlayer.type)
                        print(conn)
                        for g in parent1._genes:
                            print(str(g))
        child.__connection_sort()


    def mutate(self):
        """ Mutates this chromosome """

        r = random.random
        if r() < Config.prob_addlayer:
            self._mutate_add_layer()

        else:
            for g in self._genes:
                if r() < Config.prob_mutatelayer:
                   g.mutate() # mutate layer params

        return self

    def _mutate_add_layer(self):

        # dirpath = os.getcwd()
        #
        # file_path = dirpath + '\\'+"checking_mutate_status.txt"
        # status_file = open(file_path,'a+')

        r = random.random
        self.__connection_sort()
        # create new either conv or dense gene
        if self._gene_type == 'CONV':
            ng = genome.ConvGene(None, random.choice(genome.ConvGene.layer_params['_size']))
            self._genes.append(ng)
        elif self._gene_type == 'LSTM':
            ng = genome.LSTMGene(None, random.choice(genome.LSTMGene.layer_params['_size']))
            self._genes.append(ng)
        else:
            ng = genome.DenseGene(None, random.choice(genome.DenseGene.layer_params['_size']))
            self._genes.append(ng)
        # add the possibility of inserting new layer in front of existing layers linearly
        n = genome.LayerGene(0, 'IN', 0)
        x = genome.LayerGene(-1, 'OUT', 0)
        conns = self._connections.copy()
        conns.insert(0, genome.Connection([n], []))
        # randomly pick location for new layer
        inind = random.randrange(len(conns))
        inconn = conns[inind].copy()
        inconn._out.append(ng)


        # status_file.write( " ------------------------------------------------------------------------------- \n" )
        # status_file.write("the number of connection in module chromosome : %d  \r  "%len( conns))
        # status_file.write("len of the fist random range for in ind : %d \r  "%len( self._connections[inind - 1]._out ))
        # TODO: it is done for controling the scale of the neural network
        if len( self._connections[inind - 1]._out ) > 20:
            return

        # if in front was chosen, add a new connection to the connection list
        if inind == 0:
            self._connections[0]._in.pop(0)
            self._connections[0]._in.insert(0,ng)
            self._connections.insert(0,inconn.copy())
            self.__connection_sort()
            return
        # otherwise add the new layer to the output of the chosen input connections
        else:
            self._connections[inind-1]._out.append(ng)

        # add possibility of inserting layer at end linearly
        conns.append(genome.Connection([], [x]))
        # choose random out location
        outind = random.randrange(inind,len(conns))
        output = conns[outind].copy()

        # status_file.write("len of second random range for outind : %d \r\n"%len( output._in ))
        # TODO: it is done for controling the scale of the neural network
        if len(output._in)>20:
            #TODO : reseting everything for inind
            self._connections[inind - 1]._out.pop(len(self._connections[inind-1]._out)-1)
            return

        # if the output location is the same as the input location
        if outind == inind:
            # if this layer is the only layer that takes output, choose a new out location
            if len(output._in) == 1 or outind == 0:
                outind = random.randrange(inind+1,len(conns))
                output = conns[outind].copy()
            else:
            # otherwise, insert a new connection
                inlayers = output._in.copy()
                self._connections[inind-1]._out.remove(ng)
                self._connections[inind-1]._in.append(ng)
                self._connections.insert(inind-1, genome.Connection(inlayers, [ng]))
                self.__connection_sort()
                return
        output._in.append(ng)
        if len(output._in) == 1 and output._out[0].type == 'OUT':
            self._connections[-1]._out.pop(0)
            self._connections[-1]._out.insert(0,ng)
            self._connections.append(output.copy())
            self.__connection_sort()
            return
        else:
            self._connections[outind-1]._in.append(ng)
        self.__connection_sort()

    def _mutate_add_layer_debug(self):
        r = random.random
        self.__connection_sort()
        # create new either conv or dense gene
        if self._gene_type == 'CONV':
            ng = genome.ConvGene(None, random.choice(genome.ConvGene.layer_params['_size']))
            self._genes.append(ng)
        elif self._gene_type == 'LSTM':
            ng = genome.LSTMGene(None, random.choice(genome.LSTMGene.layer_params['_size']))
            self._genes.append(ng)
        else:
            ng = genome.DenseGene(None, random.choice(genome.DenseGene.layer_params['_size']))
            self._genes.append(ng)
        # add the possibility of inserting new layer in front of existing layers linearly
        n = genome.LayerGene(0, 'IN', 0)
        x = genome.LayerGene(-1, 'OUT', 0)
        conns = self._connections.copy()
        conns.insert(0, genome.Connection([n], []))
        # randomly pick location for new layer
        inind = random.randrange(len(conns))
        inconn = conns[inind].copy()
        inconn._out.append(ng)
        print(len(conns))

        # if in front was chosen, add a new connection to the connection list
        if inind == 0:
            self._connections[0]._in.pop(0)
            self._connections[0]._in.insert(0,ng)
            self._connections.insert(0,inconn.copy())
            self.__connection_sort()
            return
        # otherwise add the new layer to the output of the chosen input connections
        else:
            self._connections[inind-1]._out.append(ng)

        # add possibility of inserting layer at end linearly
        conns.append(genome.Connection([], [x]))
        # choose random out location
        outind = random.randrange(inind,len(conns))
        output = conns[outind].copy()
        # if the output location is the same as the input location
        if outind == inind:
            # if this layer is the only layer that takes output, choose a new out location
            if len(output._in) == 1 or outind == 0:
                outind = random.randrange(inind+1,len(conns))
                output = conns[outind].copy()
            else:
            # otherwise, insert a new connection
                inlayers = output._in.copy()
                self._connections[inind-1]._out.remove(ng)
                self._connections[inind-1]._in.append(ng)
                self._connections.insert(inind-1, genome.Connection(inlayers, [ng]))
                self.__connection_sort()
                return
        output._in.append(ng)
        if len(output._in) == 1 and output._out[0].type == 'OUT':
            self._connections[-1]._out.pop(0)
            self._connections[-1]._out.insert(0,ng)
            self._connections.append(output.copy())
            self.__connection_sort()
            return
        else:
            self._connections[outind-1]._in.append(ng)
        self.__connection_sort()


    def __connection_sort(self):
        assert self._connections[0]._in[0].id == 0, str(self._connections[0])
        assert self._connections[-1]._out[0].id == -1, str(self._connections[-1])
        new_conns = []
        new_conns.append(self._connections[0].copy())
        del self._connections[0]
        avail_layers = new_conns[0]._out.copy()
        ind = 0
        pos_correct = False
        while len(self._connections) > 1:
            need_layers = self._connections[ind]._in
            for layer in need_layers:
                if layer not in avail_layers:
                    ind += 1
                    pos_correct = False
                    break
                else:
                    pos_correct = True
            if(pos_correct):
                next_conn = self._connections[ind].copy()
                del self._connections[ind]
                avail_layers.extend(next_conn._out.copy())
                new_conns.append(next_conn)
                ind = 0
                pos_correct = False
        assert self._connections[0]._out[0].id == -1
        new_conns.append(self._connections[0].copy())
        self._connections = new_conns.copy()


    @classmethod
    def create_initial(cls):
        c = cls(None,None)
        n = genome.LayerGene(0, 'IN', 0)
        x = genome.LayerGene(-1, 'OUT', 0)
        if Config.conv and random.random() < Config.prob_addconv:
            c._gene_type = 'CONV'
            g = genome.ConvGene(None, random.choice(genome.ConvGene.layer_params['_size']))
            c._genes.append(g)
            c._connections.append(genome.Connection([n],[g]))
            c._connections.append(genome.Connection([g],[x]))
        elif Config.LSTM and random.random() < Config.prob_addLSTM:
            c._gene_type = 'LSTM'
            g = genome.LSTMGene(None, random.choice(genome.LSTMGene.layer_params['_size']))
            c._genes.append(g)
            c._connections.append(genome.Connection([n],[g]))
            c._connections.append(genome.Connection([g],[x]))
        else:
            g = genome.DenseGene(None, random.choice(genome.DenseGene.layer_params['_size']))
            c._genes.append(g)
            c._connections.append(genome.Connection([n],[g]))
            c._connections.append(genome.Connection([g],[x]))
        return c
