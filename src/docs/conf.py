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
version_file = os.path.join(os.path.dirname(__file__), "../FleetRL", "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()

project = 'FleetRL'
copyright = '2023, Enzo Alexander Cording'
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
html_static_path = ['_static']
#html_logo = "_static/img/logo.png"
