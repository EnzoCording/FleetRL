[![Documentation Status](https://readthedocs.org/projects/fleetrl/badge/?version=latest)](https://fleetrl.readthedocs.io/en/latest/?badge=latest)

[![HitCount](https://hits.dwyl.com/EnzoCording/FleetRL.svg?style=flat-square)](http://hits.dwyl.com/EnzoCording/FleetRL)

<img width="200" src="https://github.com/EnzoCording/FleetRL/blob/master/docs/_static/FleetRL_logo.jpg">

**Overview**

FleetRL is a Reinforcement Learning (RL) environment for EV charging optimization with a 
special focus on commercial vehicle fleets. Its main function is the modelling of real-world
charging processes. FleetRL was developed with a modular approach, keeping in mind that
improvements and additions are important to maintain the value of this framework.
Emphasis was therefore laid on readability, ease of use and ease of maintenance.
For example, the base-derived class architecture was used throughout the framework,
allowing for easy interchangeability of modules such as battery degradation. Emphasis was also
laid on customizability: own schedules can be generated,
electricity prices can be switched by changing a csv file, episode length, time
resolution, and EV charging-specific parameters can be changed in their respective config files.

[Documentation](https://fleetrl.readthedocs.io/)

**Physical model**

As seen in the graphic below, the physical scope currently includes the EVs, 
the building (load + PV), a limited grid connection, and electricity prices
(both for purchase and feed-in). The EVs have probabilistic schedules that are 
generated beforehand. The main objective is thereby to minimize charging cost
while respecting the constraints of the schedules, grid connection and SOC requirements.
Battery degradation is modelled, both linearly and non-linearly.
Electricity prices are taken from the EPEX spot market. PV production data is taken
from the MERRA-2 open data set, available [here](https://www.renewables.ninja/).
Building load was taken from the [NREL TMY-3 dataset](https://doi.org/10.25984/1876417).

<img width="600" src="https://github.com/EnzoCording/FleetRL/blob/master/docs/_static/FleetRL_overview.jpg">

**Installation**

> **_NOTE:_**  Python >= 3.10 is strongly recommended.

> **_NOTE:_**  The creation of a virtual environment is strongly recommended.
> To be able to use GPU compute, CUDA drivers must be installed
> (11.7 was mostly used during development).

> **_NOTE:_** If you encounter errors during installation, try:
>```
>pip install --upgrade pip setuptools
>```

**Installation via Github repository**

```
git clone https://github.com/EnzoCording/FleetRL.git
cd FleetRL
pip install -r requirements.txt
```

- Unzip the package
- Rename directory from FleetRL-master to FleetRL
- cd into /FleetRL
- pip install -r requirements.txt

> **_NOTE:_** On remote environments on vast.ai it can be necessary to run 
> pip install -U numpy prior to installing FleetRL

**Installation via Miniconda on Windows**

In this example, FleetRL can be installed completely from scratch, only Miniconda is required.
Run the commands below consecutively.

```
    conda create -n **environment_name** python=3.10
    conda activate **environment_name**
    pip install jupyter
    jupyter notebook
```

Inside the Jupyter Notebook, being in the FleetRL directory:

```
    !pip install -r requirements.txt
    # restart kernel
    import FleetRL
```
**License**

As of now, the repository uses the GPL 3.0 License. If this is
significantly constricting you, please reach out to me!

[License](LICENSE)
