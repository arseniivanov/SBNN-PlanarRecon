# SBNN-PlanarRecon
Implementation of plane recognition using a spiking neural network

The scope of this project is an alternative implementation of a digital network using SNN's for the same task.

Starting out, its vital to be able to get a simple classifier working using the SNN network, so the KTH dataset is used as a base.

Currently, the best accuracy on the test-set is 40%, which is not satisfactory enough. Once the accuracy gets to ~80-90%, we can start transforming the problem into a ground plane estimation one.

RecurrentSNN - 40%, 30 epochs
RecurrentSNN + topk + attention - 63%, 30 epochs


Example confusion matrix:

['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping']

[[14  3  3  0  0  0]
 [ 7  2  8  0  0  0]
 [ 2  5 12  0  0  0]
 [ 1  0  1 14  4  0]
 [ 1  0  2  2 12  2]
 [ 0  0  1  3  1 12]]
Test Accuracy: 58.93%

We can see the network has a hard time classifying jogging people. It is putting then mostly in walking or running bucket.
One can theorize that the 5x5 filters do not have a large enough receptive field to capture this motion in the 120x160 resolution frames

It seems like thresholding the difference yields faster training time, but convergeance to the same value.

TODO:

Re-make/cache the frame preprocessing in a way where we keep the generators while having information about frame counts and batch buckets

Understand why load + eval mode does not function as intended.
Breakpoint at eval after 1 train loop with low LR + breakpoint at eval only.
Use first N frames to understand an area of movement, crop the movement only while tracking any movement of the area