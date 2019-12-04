# Course project template

This is a template for course projects. We use [GitHub Classroom](https://classroom.github.com) to administrate our student projects and so you need to sign up for a [GitHub Account](http://github.com).

You can use the [Jupyter Notebook](https://github.com/HumanCapitalAnalysis/template-course-project/blob/master/student_project.ipynb) to work on your project. It contains an example replication of a paper by Carneiro & al. (2011) who study the marginal return to a college education in the United States using the National Longitudinal Survey of Youth 1979 (NLSY79).

* Carneiro, P., Heckman, J. J., & Vytlacil, E. J. (2011). [Estimating marginal returns to education.](https://www.aeaweb.org/articles?id=10.1257/aer.101.6.2754) *American Economic Review, 101*(6), 2754–81.

## Project overview

Please ensure that a brief description of your project is included in the [README.md](https://github.com/HumanCapitalAnalysis/template-course-project/blob/master/README.md), which provides a proper citation of your baseline article. Also, please set up the following badges that allow to easily access your project notebook.

<a href="https://nbviewer.jupyter.org/github/HumanCapitalAnalysis/student-project-template/blob/master/student_project.ipynb"
   target="_parent">
   <img align="center"
  src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.png"
      width="109" height="20">
</a>
<a href="https://mybinder.org/v2/gh/HumanCapitalAnalysis/student-project-template/master?filepath=student_project.ipyn"
    target="_parent">
    <img align="center"
       src="https://mybinder.org/badge_logo.svg"
       width="109" height="20">
</a>

## Reproducibility

To ensure full reproducibility of your project, please try to set up a [Travis CI](https://travis-ci.org) as your continuous integration service. An introductory tutorial for [conda](https://conda.io) and [Travis CI](https://docs.travis-ci.com/) is provided [here](https://github.com/HumanCapitalAnalysis/template-course-project/blob/master/tutorial_conda_travis.ipynb). While not at all mandatory, setting up a proper continuous integration workflow is an extra credit that can improve the final grade.

[![Build Status](https://travis-ci.org/HumanCapitalAnalysis/template-course-project.svg?branch=master)](https://travis-ci.org/HumanCapitalAnalysis/template-course-project)

In some cases you might not be able to run parts of your code on  [Travis CI](https://travis-ci.org) as, for example, the computation of results takes multiple hours. In those cases you can add the result in a file to your repository and load it in the notebook. See below for an example code.

```python
# If we are running on TRAVIS-CI we will simply load a file with existing results.
if os.environ['TRAVIS']:
  rslt = pkl.load(open('stored_results.pkl', 'br'))
else:
  rslt = compute_results()

# Now we are ready for further processing.
...
```

However, if you decide to do so, please be sure to provide an explanation in your notebook explaining why exactly this is required in your case.

## Structure of notebook

A typical project notebook has the following structure:

* presentation of baseline article with proper citation and brief summary

* using causal graphs to illustrate the authors' identification strategy

* replication of selected key results

* critical assessment of quality

* independent contribution, e.g. additional external evidence, robustness checks, visualization

There might be good reason to deviate from this structure. If so, please simply document your reasoning and go ahead. Please use the opportunity to review other student projects for some inspirations as well.

## Frequently asked questions and answers

* *Where can I look for publications that provide the data behind their research?* Some journals provide the data for their published articles as data supplements directly on their website. In addition, the [Replication Wiki](http://replication.uni-goettingen.de/wiki/index.php/Main_Page)  and the [Harvard Dataverse](https://dataverse.harvard.edu) compile a lot such information.

* *What are other useful resources for research data?* There is a tremendous amount of data available online. For example, MDRC provides a host of data files for public use [here](https://www.mdrc.org/available-public-use-files) from the evaluation of public policy initiatives.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/HumanCapitalAnalysis/template-course-project/blob/master/LICENSE)
