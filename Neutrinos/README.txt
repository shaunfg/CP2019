###############################################

  SHAUN FENDI GAN - Computing Project 2019
  01331868, sfg17@ic.ac.uk

#############################################

#######################################################################################
## Extracting Neutrino Oscillations Parameters from a Log-Likelihood Fit (Project 1) ##
#######################################################################################

Negative Log Likelihood (NLL) for the survival probability of neutrino oscillations was minimised to find model values that described the physical phenomenon. A 2D Univariate parabolic minimiser and N-dimensional Simulated Annealing minimiser was applied to obtain these parameters. Errors for Univariate were found by assessing the curvature and errors for Simulated Annealing assessing NLL shifted by $0.5$. 
    
The  aim  of  this  project  was  to  minimise  the  negative  log-likelihood  function,  given  a  sample  of  data  and  the  survival probability  for  neutrino  oscillations.  The  Super-Kamiokande detector was the first to present this phenomenon of neutrino mixing  in  1998,  indicating  that  these  neutrino  particles  had mass. The three neutrino flavours are the: electron, muon and  tau  neutrinos  and  the  mixing  occurs  due  to  a  mixing  of their mass and flavour eigenstates.

Additionally, the methods were verified and tested using the Ackley and Sphere function, with colour maps plotted to assess the steps taken by each minimiser. Error using the second derivative of the negative log likelihood was also explored to assess closeness of fit. A Probabilistic Cooling Scheme in Simulated Annealing was implemented to reduce temperatures efficiently. 

################
## Motivation ##
################

Project 1 was selected as the outcome of the entire project appeared the most appealing, through thorough analysis, real results relating to a physical phenomenon could be found from the project. I thought it would give a taste into some aspects of how neutrino physics is carried out.

Additionally, minimisation is currently an extremely important field in computing as in currently underpins many of the modern day machine learning algorithms - e.g. gradient descent being one that is being implemented in neural networks. 

################
## Code style ##
################

Code style follows a chronological order as stated in the project problem sheet. Sections are answered and completed by coding functions and run-codes that run the functions. The section being answered is labelled as comments at the beginning of segments. Each function and run-code are segmented by ‘#%%’ separators, which only operate on Spyder and Visual Studio Code.

Pandas is used extensively within the code in order for data management and indexing. It serves as a convenient alternative to numpy as x-y values are linked together in a dataframe, where indexing is automatically taken care of when filtering data. 

Comments are made throughout to explain purposes of certain lines of code. Docstrings are used in order to highlight a specific result or required answers comments, or to highlight the operation of certain functions. 

######################
## Running the code ##
######################

- Open neutrino_oscillations.py in Spyder or Visual Studio Code
    - This will allow you to run the code in segments, and assess each component clearly 
- Run code is segments or code blocks, chronologically.

######################
## Common Bug Fixes ##
######################
- Ensure pandas is installed if errors such as ‘pd is not defined occur’ 
- Change working directory to the folder containing the scripts and data, using the in built python os package
    - Use os.chdir(<folder directory>)
    - Issues like this could occur when using IDEs such as spyder
- Do not convert into Jupyter notebook as resulted in floating point rounding errors.
- If other errors persist, please attempt on running using a Mac OS hardware or emailing me at sfg17@ic.ac.uk. 

###########
## Files ##
###########
1. neutrino_oscillations.py
    - Main script - contains all the analysis of the project 
2. plot_functions.py
    - secondary file, contains functions used in the main script to display data for thorough analysis 
    - Imported by the main script 
3. Data.txt
    - contains personalised raw data file (previously sfg17.txt) 

#########################
## Tech/framework used ##
#########################
Built with:
- [Spyder](https://www.spyder-ide.org)
    - Based in the default anaconda environment (https://www.anaconda.com)

##############
## Features ##
##############
- Univariate Parabolic Minimisation - 1D & 2D
- Simulated Annealing Minimisation - N Dimensional
    - Includes probabilistic cooling scheme for efficient cooling
- Colour map plots / Heat maps
- Errors in linear, curvature and second derivatives 
- Histogram and plot functions to analyse phenomena discovered 
- Test functions - Ackley & Sphere 

#################
## Validations ##
#################
- Tested Univariate against sphere and Ackley functions
- Tested Simulated Annealing against the Ackley function
- Compared plot of rates against energy of predicted rates and measured rates to display successful minimisation representing the underlying phenomena
- Compared error in parabolic curvature fit in univariate method, to actual value from NLL (Error in 2nd Derivative)
- Visualised steps taken by minimisers, to verify that it works as expected by theory.

#############
## Credits ##
#############
Thank you Mark Scott for the help in labs.

© [Shaun Fendi Gan](sfg17)

14th December 2019