# PollutionModelling Project
Any questions can be directed to Philipp Hähnel, `phahnel@hsph.harvard.edu`.



## Set-up

### Requirements
- Install Python 3. We recommend anaconda from: https://www.anaconda.com/download/ to simplify the development environment
- Create an anaconda environment using the supplemented environment file

      conda env create -f environment.yml

    This will install Python 3.7.2, and Tensorflow version 1.13.1, and all required packages in an environment called `pollution_modelling`. The environment file was created on MacOS Mojave 10.14.5 using GCC version clang-1001.0.46.4 from a custom build of Tensorflow supporting FMA, AVX, AVX2, SSE4.1, and SSE4.2. Also check: https://github.com/lakshayg/tensorflow-build
    
    Activate the environment using

      conda activate pollution_modelling
        
    If you use an IDE, e.g. PyCharm, choose this environment as project interpreter. Otherwise just run the scripts (see below) with that environment active from the command line.
- The compressed database collections for the pollution measurements, the traffic volumes, and the weather data are located in the folder `PollutionModelling/data/databases`. To use them, we need to install MongoDB.

    Create folders

      sudo mkdir -m 777 /data
      sudo mkdir -m 777 /data/db
    
    They are used by mongod as the default places to dump the databases.
    
    Get the community server version at 
https://www.mongodb.com/download-center?initial=true#community 
and install it. Alternatively, on MacOS use

      brew tap mongodb/brew
      brew install mongodb-community@4.0

    for that. Under Windows, change the directory to

      cd C:/"Program Files"/MongoDB/Server/4.0/bin

    or whatever your path to the executables is. Under Mac OS, this path should automatically get added to `$PATH`, so changing to the directory is not necessary. Now, connect to the port 27018

      mongod --port 27018
    
    Then decompress the compressed collections at `PollutionModelling/data/databases` and import them using

      mongoimport --port 27018 --db db_air_quality --collection <collection name> --file <path to collection dump>

    Check 
    https://docs.mongodb.com/manual/reference/program/mongoimport/ for more info.

- Caline is a Windows executable. If on MacOS, in order to run Caline, install XQuartz >=2.7.11 and WineHQ >=4.7. Check the `PollutionModelling/caline/wine_wrapper.sh` whether the path points to the right directory for the installed Wine version. Also check
    https://wiki.winehq.org/MacOS_FAQ for further info. The first time you run Wine it may want to automatically install a few more packages. After that, Caline should be able to run on MacOS. The Linux set-up should be similar.

- If you want to look at the data and browse through it, get Robo3T GUI from 
https://robomongo.org/
and connect to port 27018 to check out the database.



## Getting Started

If the set-up has been successfully completed, we can start running some scripts. First off, let's run some tests to see if everything works:

    python -m pytest unit_tests/

Next, we can prepare the input data:

    python run/process_weather.py
    python run/process_traffic.py
    python run/process_measurements.py
    
The weather data and traffic data might have broken API's. Therefore, it is better to import the Mongo collections given in `data/databases`. From the pollution measurements we find:

    - NO2:  {min: 0.0, avg: 23.61, max: 259.58}
    - PM10: {min: 4.5, avg: 11.38, max:  54.86}
    - PM25: {min: 2.4, avg:  6.83, max:  33.54}

After that, we can run Caline4

    python run/run_caline_model.py

Please check within the header for adjustable parameters. You may want to run the Caline model with a variety of settings before pre-process the estimates and train the machine learning model. This generates a denser data set. 

The receptor positions 
are determined randomly for each new run. For the publication, runs have been carried out with receptors at distances of 
6, 12, 18, 20, 24, 26, 30, 33, 40, 52, 60, 66, 78, 80, 99, 
100, 104, 105, 125, 130, 132, 160, 165, 210, 250, 315, 320, 375, 420, 480, 500, 
525, 600, 625, 640, 800, 750, 1200, 1500, 1800, 2250, 2400, and 3000 
meters away from the line sources.

A benchmark for one run to estimate pollution levels using Caline (under Windows):
        
        Total time: 9541.41s
        Total Caline run time: 9022.87s
        Average Caline run time for each tile, hour and pollutant: 0.05s
        Standard Deviation: 0.01s

The Caline script, as well as the pre-processing, and the ML model scripts have headers with user-adjustable parameters.

    python run/run_pre_processing.py
    python run/run_ml_model.py

The last remaining step is to evaluate and plot the data.

    python plotting/run_evaluation.py
        
If of interest, the weather and traffic data, and the Caline estimates can be plotted too:

    python plotting/plot_weather_data.py
    python plotting/plot_traffic_volumes.py
    python plotting/plot_pollution_estimates.py

Check the header of each file for more detail. 

Work in progress: more plotting functions are in `PollutionModelling/plotting/paper_plots.py`



## package `unit_tests`
- This package contains (a few) tests for the database connection and utility functions. Run it from the console from the root of the repository:
    
      python -m pytest unit_tests/
