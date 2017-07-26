# PersonReID

## task description

It aims at spotting a person of interest in other cameras.

The video surveillance can be broken down into three modules,

* person detection
* person tracking
* person retrieval (Person-ReID)

Two  method used in current Re-ID tasks.

* end-to-end re-ID
* fast re-ID

Data:

* image-based
* video-based



## insights

* there cannot be separate objects or entities that have all their properties in common
* foreground and background

## Methods



## the issues in this task and corresponding solver

How to correctly match two images of the same person under intensive appearance changes

* lighting
* pose : can we apply affine transform on it?
* viewpoint
* low resolution





## Other

**person-reID vs classification vs retrieval**

* image classification :  training images are available for each class, and testing images
  fall into these predefined classes.
* instance retrieval : usually there is no training data because one does not know the content of the query in advance and the gallery may contain various types of objects.
* person-ReID : person re-ID is similar in that the training classes are available, which includes images of different identities. Person re-ID is also similar to instance retrieval in that the testing identities are unseen.



## Deep Metric Learning for Person Re-Identification

**key contribution**

* learn similarity metric from image pixels directly
* combine the separate modules together that is learning the color feature, texture feature and metric in a unified framework.



**intuition**

* two sub-networks does not share the same weights and biases, each sub-network can adapt to its corresponding view..... (??????)



**detail**

* does not need the two sub-networks share the same weights and biases.
* siamese network,  `sample-pair-->label`
* learn three SCNNs for image part
* Hinge loss is robust to outliers



## Terminology

* **cross database experiment** : train on one dataset and test on the other dataset
* **metric learning** : 
* **object recognition** : 
* **classification** : 
* **probe image** : query image