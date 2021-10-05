# Installing PSOAS
- Download or clone the repository (git clone git@github.com:upb-lea/PSOAS.git)
- Go to the PSOAS directory and run: ``python setup.py build_ext --inplace``

## Installing CEC-2013 to use it in the evaluation framework
- Download from https://github.com/yyamnk/cec2013single or ``git clone git@github.com:yyamnk/cec2013single.git``
- Go to cec2013single/cec2013single/cec2013_func.c line 91
- Insert the absolute path to cec2013_data (e.g.: PATH-TO-DIR/cec2013single/cec2013single/cec2013_data)
- After inserting the path make sure to recompile: ``gcc cec2013_func.c``
- Go back to cec2013single and run: ``python setup.py build_ext --inplace``
- CAUTION: Make sure that you adjusted the import in the example notebook to reflect your folder structure

