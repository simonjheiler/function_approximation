## Project overview

* This repository provides a simulation study of alternative function interpolation / approximation methods
* The goal is to give an overview of multi-dimensional interpolation methods and to assess implementability for respy

## Notebook viewer

<a href="https://nbviewer.jupyter.org/github/HumanCapitalAnalysis/student-project-simonjheiler/blob/master/student_project.ipynb"
   target="_parent">
   <img align="center"
  src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.png"
      width="109" height="20">

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/HumanCapitalAnalysis/student-project-simonjheiler/master)

## Reproducibility

To ensure reproducibility, this repository is under continuous integration control using Travis CI.

Current build status:

[![Build Status](https://travis-ci.org/HumanCapitalAnalysis/student-project-simonjheiler.svg?branch=master)](https://travis-ci.com/simonjheiler/function_approximation.svg?branch=master)

The following elements are under CI control:
* testing harness (tests contained in ./src/)
* notebooks in ROOT


## Structure of notebook

* Motivation of the study and relation to respy

* Introduction of the test functions used for the study

* Benchmark case: multilinear interpolation on a regular grid

* Comparison of alternative interpolation methods:
   - multidimensional spline interpolation
   - smolyak methods

* Adaptations required for functions with discrete domain

* Outlook on potential implementation in respy


## References

* Surjanovic, S. & Bingham, D. (2013). Virtual Library of Simulation Experiments: Test Functions and Datasets. Retrieved February 1, 2020, from http://www.sfu.ca/~ssurjano.
* Zhou, Y. (1998). Adaptive importance sampling for integration. Ph.D. Thesis, Stanford University.


## License
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/HumanCapitalAnalysis/student-project-simonjheiler/master/LICENSE)
