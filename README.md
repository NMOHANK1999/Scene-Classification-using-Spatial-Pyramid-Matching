# Scene-Classification-using-Spatial-Pyramid-Matching
 You will implement a scene classification system that uses the bag-of-words
approach with its spatial pyramid extension. The paper that introduced the pyramid matching kernel [2] is
K. Grauman and T. Darrell. The Pyramid Match Kernel: Discriminative Classification with Sets of Image Features. ICCV 2005. http://www.cs.utexas.edu/
~grauman/papers/grauman_darrell_iccv2005.pdf
Spatial pyramid matching [4] is presented in
S. Lazebnik, C. Schmid, and J. Ponce, Beyond Bags of Features: Spatial Pyramid
Matching for Recognizing Natural Scene Categories, CVPR 2006. http://www.di.
ens.fr/willow/pdfs/cvpr06b.pdf

You will be working with a subset of the SUN database2
. The data set contains 1600 images from various
scene categories like “aquarium, “desert” and “kitchen”. And to build a recognition system, you will:
• take responses of a filter bank on images and build a dictionary of visual words, and then
• learn a model for images based on the bag of words (with spatial pyramid matching [4]), and use
nearest-neighbor to predict scene classes in a test set.
In terms of number of lines of code, this assignment is fairly small. However, it may take a few hours to
finish running the baseline system, so make sure you start early so that you have time to debug things. Also,
try each component on a subset of the data set first before putting everything together. We provide
you with a number of functions and scripts in the hopes of alleviating some tedious or error-prone sections
of the implementation. You can find a list of files provided in Section 4. Though not necessary, you are
recommended to implement a multi-processing3
version to make use of multiple CPU cores to speed up the
code. Functions with n worker as input can benefit greatly from parallel processing. This homework was
tested with Anaconda 2019.10 (Python 3.7).
Hyperparameters We provide you with a basic set of hyperparameters, which might not be optimal.
You will be asked in Q3.1 to tune the system you built and we suggest you to keep the defaults before you
get to Q3.1. All hyperparameters can be found in a single configuration file opts.py.
