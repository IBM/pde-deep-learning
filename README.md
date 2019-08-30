# Deep Learning for PDE-based Models

Consider a complex non-linear forecasting problem, e.g. from weather- and air pollution data. A general issue with those problems is that forecasting methods based on solving partial differential equations (PDEs) require a lot of computing power in the model-application phase, especially for applications to large domains, where domain decomposition methods are applied. An idea to circumvent this issue is to use deep-learning techniques to reduce the run-time of the model-application phase at a cost of increasing the run-time of the model-training phase. In this project, we propose domain decomposition methods to scale the deep-learning model to larger domains by imposing consistency constraints during the training across sub-domain boundaries. We demonstrate the methods at the example of an air pollution forecasting problem, which is governed by an advection-diffusion process and described through a PDE.

If you use the code, please also cite our paper: https://arxiv.org/abs/1810.09425

The databases are currently not supplemented since they are proprietary. We are working on providing artificially generated replacements. 

Any questions can be directed to Philipp Hähnel, `phahnel [at] hsph.harvard.edu`.



## Set-up

### Requirements

- Install Python 3. We recommend anaconda from: https://www.anaconda.com/download/ to simplify the setup of the development environment
- Create an anaconda environment using the supplemented environment file

      conda env create -f environment.yml

    This will install Python 3.7.2, and Tensorflow version 1.13.1, and all required packages in an environment called `pollution_modelling`. The environment file was created on MacOS Mojave 10.14.5 using GCC version clang-1001.0.46.4 from a custom build of Tensorflow supporting FMA, AVX, AVX2, SSE4.1, and SSE4.2. Also check: https://github.com/lakshayg/tensorflow-build
    
    Activate the environment using

      conda activate pollution_modelling
        
    If you use an IDE, e.g. PyCharm, choose this environment as project interpreter. Otherwise just run the scripts (see below) with that environment active from the command line.
- The compressed database collections for the pollution measurements, the traffic volumes, and the weather data are located in the folder `PollutionModelling/data/databases`. To use them, install MongoDB.

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

      cd C:\Program Files\MongoDB\Server\4.0\bin\

    or whatever your path to the executables is. Under Mac OS, this path should automatically get added to `$PATH`, so changing to the directory is not necessary. Now, connect to the port `27018`

      mongod --port 27018
    
    Then decompress the compressed collections at `PollutionModelling/data/databases` and import them using

      mongoimport --port 27018 --db db_air_quality --collection <collection name> --file <path to collection dump>

    Check 
    https://docs.mongodb.com/manual/reference/program/mongoimport/ for more info.

- If you want to look at the data and browse through it, get Robo3T GUI from 
https://robomongo.org/
and connect to port `27018` to check out the database.


### Requirements (Part 2, not supplemented for now)

If you are also interested in running the PDE solver Caline 4.0 to generate some training data for the deep learning model, then there is an additional requirement needed:

- Caline is a Windows executable. If on MacOS, in order to run Caline, install XQuartz >=2.7.11 and WineHQ >=4.7. Check the `PollutionModelling/caline/wine_wrapper.sh` whether the path points to the right directory for the installed Wine version. Also check
    https://wiki.winehq.org/MacOS_FAQ for further info. The first time you run Wine it may want to automatically install a few more packages. After that, Caline should be able to run on MacOS. The Linux set-up should be similar.



## Getting Started

If the set-up has been successfully completed, you can start running some scripts. 


### From Scratch

Not all of the listed scripts and steps are supplemented in this repository. We include them here to allow the user to follow and understand the steps of how the data was generated.

The input data has been prepared using

    python run/process_weather.py
    python run/process_traffic.py
    python run/process_measurements.py
    
You can also just import the MongoDB collections given in `data/databases`. From the pollution measurements we find:

    - NO2:  {min: 0.0, avg: 23.61, max: 259.58}
    - PM10: {min: 4.5, avg: 11.38, max:  54.86}
    - PM25: {min: 2.4, avg:  6.83, max:  33.54}

The units are micro gram per cubic centimeter. After that, we can run Caline 4.0 using

    python run/run_caline_model.py

Please check within the header of the file for adjustable parameters. 

This script estimates the traffic-induced air pollution levels of NO2, PM2.5, and PM10 for 
defined receptors across the domain. The prediction framework consisted of an air-pollution 
dispersion model, inputs of traffic volumes for a number of roadway links across the city, 
and weather data. Outputs consist of periodic estimates of pollution levels. The PDE-based model 
used is based on the Gaussian Plume model, a standard model in describing the steady-state 
transport of pollutants. Caline 4 implements the Gaussian Plume model and is one of the 
“Preferred and Recommended Air Quality Dispersion Models” of the Environmental Protection Agency 
in the USA as of 2018.
    
The input to the Caline model on each sub-domain consists of:

- (normalized) coordinates of 20 line sources, padded with zeros if fewer sources are in the partition;
- integrated traffic volumes over one hour for each source, padded with zeros;
- the average wind direction, the standard deviation of wind direction, the wind speed, and the temperature in this hour;
- model specific parameters, such as atmospheric stability standard, emission factor, 
  aerodynamic roughness coefficient standard, settling velocity standard, disposition 
  velocity standard, and mixing zone dimensions;
- (normalized) coordinates of 20 receptors.

The output consists of:
- Average NO2, PM2.5, and PM10 concentrations at the receptors in ㎍/cm3 for that hour. 

Caline 4 is limited to 20 line sources and 20 receptors per computational run. The bigger domains of Dublin and the demo have been decomposed into sub-domains with a maximum of 20 line sources. See 
`util.util_domain_decomposition.decompose_domain` for more info on the decomposition. 
The receptors are placed at random positions, but in intervals at certain distances to the line 
sources, based on the `contour_distance` in the parameter initialization. See 
`util.util_domain_decomposition.get_receptors` for more info on their placement.

The Caline estimates are very low for low traffic volumes and are in a regime of low numerical 
resolution. We therefore scale the traffic volumes by a `scaling_factor` and divide the 
pollution concentration outputs by it. Schematically,

    Caline(traffic, background) = Caline(scaling_factor * traffic, 
                                         scaling_factor * background) / scaling_factor
                                         
This assumes a linear dependence, which we could empirically confirm to good approximation. In any case, for this project, we are only interested in using this PDE solver as a generator for ground truth data, so the physical accuracy of this step is of no major concern.

Caline is only modeling the contribution to the pollution levels coming from traffic and weather influence. Thus, we remove the default background pollution from the output again.

    output = Caline(traffic, weather, default background) - default background

A benchmark for one run to estimate pollution levels using Caline (under Windows):
        
        Total time: 9541.41s
        Total Caline run time: 9022.87s
        Average Caline run time for each tile, hour and pollutant: 0.05s
        Standard Deviation: 0.01s

The Caline script has a header with user-adjustable parameters.

You may want to run the Caline model with a variety of settings before pre-processing the estimates and train the machine learning model. Doing so generates a denser data set. 

The receptor positions are determined randomly for each new run. For the publication, runs have been carried out with receptors at distances of multiples of
`6, 11, 17, 20, 26, 33, 37, 47, 77, 105, 125, 160, 550, 600, 750`
meters away from the line sources for the Dublin data, and `5, 6, 7, 8, 9, 10, 11, 13, 17, 27, 37, 53, 71, 101, 103` for the demo.


### From the Database

The supplemented database contains all generated data up to this point. Next up is the pre-processing for the training of the ML model. This involves rescaling all different input units to lie within the same order of magnitude. Check the headers of the files for user-adjustable parameters.

    python run/run_pre_processing.py

Within the user-chosen parameters, you can select the tags for identifying the Caline data in the collection, which has the identifiers

    'run_tag': (str) anything, but we used the date of the run in format 'YYYY-MM-DD' 
    'case': (str) 'Dublin' or 'Demo'
    'contour_distance': (int) interval distance for placing receptors from line sources

From the collection, different pre-processing runs are identified via the ‘case’ (Dublin or Demo, as given in the get_parameters() L.99), and 'mesh_size', which is the length of the sub domain selection in L.86. So, it would be possible to have a couple of pre-processed data together without creating and managing new collections: 

1) the Demo
2) the full Dublin example
3) some subset of the Dublin example, restricted to a number of tiles

Make sure that there is no other data in the pre_proc collection!

Now we can run the deep learning model.

    python run/run_ml_model.py

The last remaining step is to evaluate the data.

    python plotting/run_evaluation.py
        
If of interest, many things can be plotted:

    python plotting/plot_maps.py
    python plotting/plot_traffic_volumes.py
    python plotting/plot_weather_data.py

Check the header of each file for more detail. 


## Thank you!

Thank you for your interest in this project! Don't hesitate to contact us if you have any questions, comments, or concerns. 
