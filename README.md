This repository contains all the code necessary to run a demo of the capabilities of the Nanowire Statistics software, as well as the Jupyter notebook that runs the demo. It does not contain the entire codebase of the project. This will be released during the summer of 2017.

To run the code, you need to install a distribution of Python, and the necessary packages. The easiest way to make sure you have the right packages is to install Anaconda, and import the Anaconda environment from the file NWstats.yml

Note: This tutorial is written from my experiences running the software on Windows 10. It might or might not work the same way for other operating systems.

Anaconda is a Python distribution, that aims to simplify package management and deployment through the package, dependency and environment manager conda. It can be found at https://www.continuum.io/downloads

After installing Anaconda, you need to import the provided NWstats environment, which will give you the necessary packages to run the demo. To do this, run the shortcut named "Anaconda Prompt", that came with your Anaconda installation. Use the prompt to navigate to your downloaded NWstats repository, and enter the following command:

conda env create -f NWstats.yml

This will start installing the necessary packages. 

To run the notebook, you need to start Jupyter Notebook in the correct Anaconda environment. In the Anaconda Prompt, enter the command

activate NWstats

to activate the NWstats environment. Then enter the command

jupyter notebook

This will open Jupyter Notebook in a browser window. Navigate to the NWstats files, and run the file "NWstats_demo.ipynb".
