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
* pose
* viewpoint





## Other

**person-reID vs classification vs retrieval**

* image classification :  training images are available for each class, and testing images
  fall into these predefined classes.
* instance retrieval : usually there is no training data because one does not know the content of the query in advance and the gallery may contain various types of objects.
* person-ReID : person re-ID is similar in that the training classes are available, which includes images of different identities. Person re-ID is also similar to instance retrieval in that the testing identities are unseen.



## Deep Metric Learning for Person Re-Identification

**key contribution**

* learn similarity metric from image pixels directly
* siamese deep neural network
* ​