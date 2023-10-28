.. _installation:

Installation
=============

**Prerequisites**

 * FleetRL requires Python >=3.8

.. note::

    Python >= 3.10 is strongly recommended.

.. note::

    The creation of a virtual environment is strongly recommended. To be able to use GPU compute,
    CUDA drivers must be installed (11.8 was mostly used during development).

Installation via Github repository:
 * Unzip the package
 * Rename directory from FleetRL-master to FleetRL
 * cd into /FleetRL
 * pip install -r requirements.txt

.. note::

    On remote environments on vast.ai it can be necessary to run ``pip install -U numpy`` prior to
    installing FleetRL

**Miniconda Windows**

In this example, FleetRL can be installed completely from scratch, only Miniconda is required.
Run the commands below consecutively.

.. code-block::

    conda create -n **environment_name** python=3.10
    conda activate **environment_name**
    pip install jupyter
    jupyter notebook

Inside the Jupyter Notebook, being in the FleetRL directory:

.. code-block::

    !pip install -r requirements.txt
    # restart kernel
    import FleetRL

At this point, the ``complete_pipeline.ipynb`` should run completely. To use GPU, CUDA must be
properly configured.
