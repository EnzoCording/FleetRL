.. FleetRL documentation master file, created by
   sphinx-quickstart on Fri Sep  8 15:07:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FleetRL - Realistic RL environments for commercial EV fleets
============================================================

Github: https://github.com/EnzoCording/FleetRL

Software features:
------------------

- Base-derived class architecture and easily interchangeable configurations
- Modular implementation approach for easily extendable model and methods
- PEP8 compliant (unified code style)
- Documented functions and classes
- Stable-Baselines3 integration, parallelization, tensorboard support
- Extensive data logging and evaluation possibilities

Unique Implementations:
-----------------------

- Non-linear battery degradation
- Fleet schedule generation - inspired by emobpy and applied to commercial fleets
- Bi-directional charging
- Building load, grid connection limit, PV, spot price
- Arbitrarily variable episode length, 15-min time resolution
- Benchmarking of RL methods with static benchmarks: uncontrolled charging, night charging, distributed charging
- **Benchmarking of RL methods with linear optimization**

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/overview
   guide/installation
   guide/custom_env
   guide/agent_training
   guide/agent_eval
   guide/benchmarking

.. toctree::
   :maxdepth: 1
   :caption: Modules

   FleetRL.fleet_env
   FleetRL.utils.data_logger
   FleetRL.utils.load_calculation
   FleetRL.utils.normalization
   FleetRL.utils.observation
   FleetRL.utils.schedule
   FleetRL.utils.time_picker
   FleetRL.utils.data_processing
   FleetRL.utils.ev_charging
   FleetRL.utils.battery_degradation

Indices and tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
