# Point descriptor

## How to match point?

## Criteria for Point Descriptor

* We want the descriptors to be the (almost) same in both image - invariant.
* We also need the descriptors to be distinctive.

## SIFT :  Scale Invariant Fearure Transform

* motivation: The Harris operator was not invariant to scale and correlation was not invariant to rotation.
* For better image matching, Lowe's goals were:
  * To develop an interest operator - a detector - that is invariant to scale and rotation.
  * Also: create a descriptor that was robust to the variations corresponding to typical viewing conditions. The descriptor is the most-used part of SIFT.

## Idea of SIFT

* Image content is represented by a constellation(一系列) of local features that are invariant to translation, rotation, scale, and other imaging parameters.

## Overall SIFT Procedure

* scale-space extrema Detection
* keypoint localization

* orientation assignment
* keypoint description

## Keypoint Descriptors

* Next is to compute a descriptor for the local image region about each keypoint that is:
  * Highly *distinctive*
  * As *invariant* as possible to variations such as changes in viewpoint and illumination

## First  normalization
在计算`Keypoint Descriptors`的时候，首先要：
**Normalization**

* Rotate the window to standard orientation
* Scale the window size based on the scale on the scale at which point was found.

## Compute a feature vector based upon

* histograms of gradients, (HOG)
* weighted by the magnitude of the gradient

## How to build feature vector

* Normalization
  * Rotate the window to standard orientation (dominate direction to north..)
  * Scale the window size based on the scale on the scale at which point was found.
* histograms of gradients
  * compute the histograms of gradients in the window.
* put a gaussian over the HOG (if you get small shift, you don't have large change of descriptor)
* Build little histograms
  * look at whole image gradients of window, divide it into 16 pieces
  * histogram has 8 bin (8 direction)
  * then we have 16*8 features per window. then we have descriptor.

## Reduce effect of illumination(光照)

* Clip gradient magnitudes to avoid excessive influence of high gradients
  * after rotation normalization, clamp gradients > 0.2
* 128-dim vector normalized to magnitude 1.0

## Evaluating the SIFT descriptors

* Database images  were subjected to rotation, scaling, affine stretch, brightness nad contrast changes, and added noise.
* Feature point detectors and descriptor were compared before and after the distortions
* Mostly lookng for stability with respect to change.
