## Title: 
Development in architecture and learning of deep neural networks using evolutionary algorithms (DALDNNEA).

Department: Faculty of Electrical and Computer Engineering University of Zanjan.

## Summary

In this work, DALDNNEA, a graph-based genetic algorithm method is developed. 
It co-evolves CNN architectures and their last layer weights. 
Moreover, It employs the Lamarckian method to tune an Individual's last layer weights;Lamarckian inheritance posits
that acquired traits obtained during an organism's lifetime can be passed on to its offspring. 
The proposed method draws inspiration from a well-known state-of-the-art NAS method, CoDeepNEAT, in terms of the structure of populations 
and their associated operators. The DALDNNEA is evaluated on the MNIST, CIFAR-10, and Penn Treebank datasets.


## How to run the framework?


In the first step, you need to download the repositories. 
Then, for generating CNN architecture based on the selected datasets such as MNIST, CIFAR-10, 
and Penn Treebank you must run the same name python file like ``MNIST.py``. 
Every dataset has specific configurations and these configuration parameters 
must be set for every dataset, as in the example ``CIFAR-10.txt`` must be imported in``config.py``.

## Outputs

The framework generates a series of files logging the evolution of populations (into .bp, txt), including information related to:
- Model assembling, evaluation, speciation, reproduction, and generation summary details  ``debug.txt``
- classification accuracy for individuals over generations, average classification populations ``(dataset name)_Ind_pop_avg_pop.txt`` and ``(dataset name)_Ind_pop_fitness ``
- Lin graphs representing the state average accuracy and accuracy of the best model in every generation, and 
then they will be saved in (.png format)
- Keras models related to the best models for every generation (\models directory in .bp format).

## Requirements
```
- Keras 2.4.3
- tensorflow 2.3.0
- numpy 1.21.2
- pickle
- matplotlib 3.5.0
- gzip 3.2
```

##Contact
If you have any questions, do not hesitate to contact us at zaniar.sharifi89@Gmail.com, we will be happy to assist.

## Dev infos
Code developed and tested by [Zaniar Sharifi]
