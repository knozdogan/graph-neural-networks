# graph-neural-networks

Convolutional neural network models, which are a type of deep learning model, are frequently used in target detection. A new approach is presented by using geometric deep learning for target detection in ground-penetrating radar systems. While convolutional neural network models capture Euclidean structures very successfully, graph neural network models can also reveal relationships independent of Euclidean space. A 6-layer convolutional graph neural network is created using more than 8000 thousand radar images.

Obtaining a graph from the image, which is one of the challenging parts of the project, is solved with a simple linear iterative clustering (SLIC) algorithm. Various features created from the image and graph structure have been added to the graph nodes. Among the nodes in the graph, those that fall on the target are labeled as 1, and those that do not are labeled as 0. The output of the model is the probability value of which class each node belongs to.

In the last stage, hyper-parameters of the model were determined and these parameters were optimized with the industry standard Ray Tune framework.

Thanks to this project, a different perspective was tried to be brought to the target detection problem in the field of computer vision.