# OXFORD-IIIT PET Dataset
-----------------------

This is a processed version of the OXFORD-IIIT PET Dataset dataset by Haocheng Yuan, the original authors are Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman and C. V. Jawahar. The dataset is used and can only be used for teaching purpose.

## Contents
--------
- Dataset/**/color: RGB images of pets
- Dataset/**/label: semantic masks (in PIL format) with pixel annotations.

    - 0: background
    - 1: cat
    - 2: dog

## Note
----
Please ensure that your model is trained using the trainval dataset and evaluated on the test set.