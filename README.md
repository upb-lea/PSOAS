# Particle Swarm Optimization Assisted by Surrogates
This repository proposes new approaches for global, nonlinear and gradient-free optimization that combine the advantages
of particle swarm optimization (PSO) and Bayesian optimization. Baseline and inspiration for this work is credited to 
the article [Directed particle swarm optimization with Gaussian-process-based function forecasting](https://doi.org/10.1016/j.ejor.2021.02.053).

[Read our introduction!](https://github.com/upb-lea/PSOAS/blob/master/psoas_doc.pdf)

## Installing PSOAS
- Download or clone the repository (``git clone git@github.com:upb-lea/PSOAS.git``)
- Go to the PSOAS directory and run: ``python setup.py build_ext --inplace``
- A usage example can be found at https://github.com/upb-lea/PSOAS/blob/master/notebooks/example_usage_optimizer.ipynb

## Installing CEC-2013 to use it in the evaluation framework
- Download from https://github.com/yyamnk/cec2013single or ``git clone git@github.com:yyamnk/cec2013single.git``
- Go to cec2013single/cec2013single/cec2013_func.c line 91
- Insert the absolute path to cec2013_data (e.g.: PATH-TO-DIR/cec2013single/cec2013single/cec2013_data)
- After inserting the path make sure to recompile: ``gcc cec2013_func.c``
- Go back to cec2013single and run: ``python setup.py build_ext --inplace``
- CAUTION: Make sure that you adjusted the import in the example notebook to reflect your folder structure

## Citing
Please use the following BibTeX entry for citing us:

    @Misc{MVSW2021,
      author = {Marvin Meyer and Hendrik Vater and Maximilian Schenke and Oliver Wallscheid},
      note   = {Paderborn University},
      title  = {Particle Swarm Optimization Assisted by Surrogates},
      year   = {2021},
      url    = {https://github.com/upb-lea/PSOAS},
    }
