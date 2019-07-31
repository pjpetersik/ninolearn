############
Installation
############

NinoLearn works with virtual Conda environments. If you do not have Anaconda
or Miniconda installed check out https://conda.io/docs/user-guide/install/.

To install NinoLearn on your machine clone the master branche of the NinoLearn
GitHub repository by executing the following snippet in your terminal:

.. code-block:: console

    git clone git@github.com:pjpetersik/ninolearn.git

Now put the PYTHONPATH to the base folder of Ninolearn into your .bashrc file:

.. code-block:: bash

    export PYTHONPATH="$PYTHONPATH:/path/to/ninolearn"

Open a new terminal or run:

.. code-block:: bash

    source .bashrc

From now on, NinoLearn will be available for your python. Now, a new conda
environment needs to be initialized such that you have the
right dependencies within your environment to work with NinoLearn.

Currently, a .yml file is just generated for python3 and linux.
For this run the  following command in your terminal:

.. code-block:: console

    conda env create -f py3_linux.yml

If you are on another system you might try:

.. code-block:: console

    conda create --name ninolearn -c conda-forge python=3.6 tensorflow keras matplotlib basemap pandas xarray dask scikit-learn netcdf4 xesmf esmpy python-igraph nbsphinx jupyter spyder

The particular package is used for the following purpose:

* :code:`tenserflow` and :code:`keras` for neural networks
* :code:`matplotlib` and :code:`basemap` for plotting
* :code:`pandas` and :code:`xarray` for data handeling
* :code:`dask` for reading large data files
* :code:`scikit-learn` for machine  learning
* :code:`netcfd4` to open files with the NetCDF4 format
* :code:`xesmf` and :code:`esmpy` for regridding data
* :code:`python-igraph` for fast computation of complex network metrics
* :code:`nbsphinx` to include jupyter notebooks in the documentation
* :code:`jupyter` to work in jupyter notebooks (e.g. in the tutorials)
* :code:`spyder` as integrated development environment (optional)

The environment activated by running

.. code-block:: console

    source $HOME/YOURCONDA/bin/activate ninolearn

Here, :code:`YOURCONDA` is a placeholder for your conda base directory. The
environment can be deactivated by running:

.. code-block:: console

    conda deactivate

You might consider to put the following :code:`alias` into your :code:`.bashrc`
file to shorten the activation process:

.. code-block:: bash

    alias ninolearn_env="source $HOME/YOURCONDA/bin/activate ninolearn"

Open a new terminal or source the :code:`.bashrc` file (as above). From now on
you can activate the environment by running :code:`ninolearn_env` in your terminal.
Within this environment NinoLearn will be available for you. Note, that the
package is still in the beginning of its development. Hence, its certainly not
free from bugs. If you encouter some problems, feel free to post them as an
issue on the GitHub repository of
`NinoLearn <https://github.com/pjpetersik/ninolearn>`_.