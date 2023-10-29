# SBNN-PlanarRecon
Implementation of plane recognition using a spiking neural network

The scope of this project is an alternative implementation of a digital network using SNN's for the same task.

Starting out, its vital to be able to get a simple classifier working using the SNN network, so the KTH dataset is used as a base.

Currently, the best accuracy on the test-set is 40%, which is not satisfactory enough. Once the accuracy gets to ~80-90%, we can start transforming the problem into a ground plane estimation one.

RecurrentSNN - 40%, 30 epochs
RecurrentSNN + topk + attention - 63%, 30 epochs


TODO:

Understand why load + eval mode does not function as intended.
Breakpoint at eval after 1 train loop with low LR + breakpoint at eval only.