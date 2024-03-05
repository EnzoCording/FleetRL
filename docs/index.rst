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

   fleetrl.fleet_env
   fleetrl.benchmarking
   fleetrl.agent_eval
   fleetrl.utils.data_logger
   fleetrl.utils.load_calculation
   fleetrl.utils.normalization
   fleetrl.utils.observation
   fleetrl.utils.schedule
   fleetrl.utils.time_picker
   fleetrl.utils.data_processing
   fleetrl.utils.ev_charging
   fleetrl.utils.battery_degradation
   fleetrl.utils.event_manager
   fleetrl.utils.rendering

Indices and tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
