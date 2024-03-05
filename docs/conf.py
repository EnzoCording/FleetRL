# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import datetime
import os
import sys
from typing import Dict

# import FleetRL
# from FleetRL.fleet_env.config.ev_config import EvConfig
# from FleetRL.fleet_env.config.score_config import ScoreConfig
# from FleetRL.fleet_env.config.time_config import TimeConfig

# from FleetRL.fleet_env.episode import Episode

# from FleetRL.utils.data_processing.data_processing import DataLoader
# from FleetRL.utils.ev_charging.ev_charger import EvCharger
# from FleetRL.utils.load_calculation.load_calculation import LoadCalculation, CompanyType

# from FleetRL.utils.normalization.normalization import Normalization
# from FleetRL.utils.normalization.oracle_normalization import OracleNormalization
# from FleetRL.utils.normalization.unit_normalization import UnitNormalization

# from FleetRL.utils.observation.observer_with_building_load import ObserverWithBuildingLoad
# from FleetRL.utils.observation.observer_price_only import ObserverPriceOnly
# from FleetRL.utils.observation.observer import Observer
# from FleetRL.utils.observation.observer_with_pv import ObserverWithPV
# from FleetRL.utils.observation.observer_bl_pv import ObserverWithBoth
# from FleetRL.utils.observation.observer_soc_time_only import ObserverSocTimeOnly

# from FleetRL.utils.time_picker.random_time_picker import RandomTimePicker
# from FleetRL.utils.time_picker.static_time_picker import StaticTimePicker
# from FleetRL.utils.time_picker.eval_time_picker import EvalTimePicker
# from FleetRL.utils.time_picker.time_picker import TimePicker

# from FleetRL.utils.battery_degradation.batt_deg import BatteryDegradation
# from FleetRL.utils.battery_degradation.empirical_degradation import EmpiricalDegradation
# from FleetRL.utils.battery_degradation.rainflow_sei_degradation import RainflowSeiDegradation
# from FleetRL.utils.battery_degradation.log_data_deg import LogDataDeg

# from FleetRL.utils.data_logger.data_logger import DataLogger

# from FleetRL.utils.schedule.schedule_generator import ScheduleGenerator, ScheduleType

# We CANNOT enable 'sphinxcontrib.spelling' because ReadTheDocs.org does not support
# PyEnchant.
try:
    import sphinxcontrib.spelling  # noqa: F401
    enable_spell_check = True
except ImportError:
    enable_spell_check = False

# Try to enable copy button
try:
    import sphinx_copybutton  # noqa: F401
    enable_copy_button = True
except ImportError:
    enable_copy_button = False

sys.path.insert(0, os.path.abspath(".."))

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "../fleetrl", "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()

project = 'fleetrl'
copyright = '2023, Enzo Alexander Cording, GNU GPL v3.0'
author = 'Enzo Alexander Cording'

# The short X.Y version
version = "master (" + __version__ + " )"
# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    # 'sphinx.ext.intersphinx',
    # 'sphinx.ext.doctest'
]

if enable_spell_check:
    extensions.append("sphinxcontrib.spelling")

if enable_copy_button:
    extensions.append("sphinx_copybutton")

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_logo = "_static/FleetRL_logo.jpg"
html_static_path = ['_static']
#html_logo = "_static/img/logo.png"
