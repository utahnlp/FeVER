FeVER
=====

This repository contains the python implementation of the paper **Beyond Context: A New Perspective for Word Embeddings**.

FeVER stands for **Feature embeddings for Vector Representation**.

Before Training
---------------

Overview
~~~~~~~~

The input of this implementation is not the pure text.
Instead, a multi-label data format is used as the input of our system.
Because of this, we need to preprocess the text into a suitable format, which is described in `Training files`_.


Prerequisites
~~~~~~~~~~~~~

.. code:: console

    pytorch 1.1.0
    numpy
    ExAssist



Installation
~~~~~~~~~~~~

1. Go to the directory of the project.
2. Install the code in development mode.

.. code:: console

    python setup.py develop


Tracking Experiments
~~~~~~~~~~~~~~~~~~~~

This python implementation used ExAssist_ to track each experiments.
Every time you run an experiment, all the output files (include experiment settings and details) will be saved in ``Experiments`` directory. If ``Experiments`` directory does not exist, a new one will be created.

Running Example
---------------

In this subsection, a small example is used to show how to use this repository.
The behavior of our code is controlled by a config file.
After installation, you can directly run our code like:

.. code:: console

    python FeVER/main.py example/config.ini

``config.ini`` file contains all the configuation for running.

A toy dataset is stored in the ``example/toy``.
Different files in this directory has different usage.

Training files
~~~~~~~~~~~~~~

To train the feature embeddings based on multi-label classification, you need to prepare three files:

1. ``context_feature_training.txt``: This file contains all training data in
   format of multi-label_ data. It contains the predicting word index and
   features of the context (output of psi function in the paper). Each word is
   mapped to an index by the ``vocabulary.txt`` file. A file contains following
   content:

.. code:: console

    2 4 3
    idx1, idx2 feat1:1.0 feat2:1.0
    idx1, idx3 feat3:1.0 feat4:1.0

In this file, the first line indicates the number of training examples,
features and labels.  For example, in this tiny file, it means there are 2
traiing examples, 4 features in total and at most 3 words.
Starting from second line, all the content in the file is training examples.
Second line means word ``idx1`` and ``idx2`` are showed in
the same context and this context has features ``feat1`` and ``feat2``.
Third line means word ``idx1`` and ``idx3`` are showed in the same context and
this context has the features of ``feat3`` and ``feat4``.


2. ``label_feature_training.txt``: This file contains the word features with
   the same format of ``context_feature_training`` file. Eeah line in the file
   represents a word in the vocabulary and its features (output of phi
   function in the paper). Suppose we have tiny file looks like this:

.. code:: console

    3 4 3
    idx1 feat1:1.0 feat2:1.0
    idx2 feat3:1.0 feat4:1.0
    idx3 feat4:1.0 feat5:1.0

In this file, there are 3 words, 4 features. Second line means word ``idx1`` has features ``feat1`` and ``feat2``.

3. ``frequency.txt``: This file contains the frequency of each word in the context. Each line in this file is corresponding to each line in the ``context_feature_training.txt`` file.


Generating files
~~~~~~~~~~~~~~~~

After training, the model needs feature files to extract word embeddings.
Be note, here we can use different vocabulary as long as we can extract feature
from this vocabulary.
feature files are in the same format as ``label_feature_training.txt``.

.. _ExAssist: https://exassist.readthedocs.io/en/latest/
.. _multi-label: http://manikvarma.org/downloads/XC/XMLRepository.html
