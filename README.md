# Python for Medical Statistics (PMS)

This project should be a framework for medical researchers
Because there are so many use-cases for medical research, and the data to work with has so many formats, this projects offers the researchers, a technical ground to allow them clean the initial data-sets, mix the data, and get the most out of it.

The current state of the project includes a concrete sollution for a specific data-set.
The project runs on top of Python 3.12.2 and it requires the following libraries to work:
- statistics
- pandas
- matplotlib
- scipy
- sklearn
- numpy
- seaborn
- statsmodels
- bioinfokit

Before running the script, there should be present 3 xlsx files, inside of a "data" local folder: ["data\\Weight_Behavior.xlsx", "data\\Elevated_Maze.xlsx", "data\\NOR-Test.xlsx"]
The execution of the scripts takes about 70 seconds and about 90 figures are created.
While all the effort could have been reduced to 1/3 using multiprocessing, at the moment this is not possible.

ToDo:
- an interactive application
- support for custom filtering of the initial data-sets
- support for building mixed data-sets, based on the initial data-sets
- support for interactive building of pipelined processings and for choosing the right functions to process the data
- support for specific clustering algorithms, to highlight data grouping behaviors
- support for machine learning and custom models, to enable and test data prediction

