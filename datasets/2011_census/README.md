# 2011 Commuter Flows
This is just a quick overview of the data being used, but you don't need this overview if you just want to grab the matrix (classic).
To get Numpy formatted data straight into your code, use the functions in `grab_data.py` in the root of the repo:
1. Commuter matrix: `get_commuter_matrix()`.
2. Population sizes: `get_population_sizes()`. *Each element corresponds to the respective patch in commuter matrix*


**@TODO: Download MSOA version by seperating via LADS on Nomis and aggregate to get MSOA matrix.**
## Census Data
### Resolution
I'm using Local Authority Districts (LADS) districts instead, which group people by administrative area instead of
grouping by number of people. The smallest spatial resolution possible here is Middle Layer Super Output Area.
LAD is a much more coarse resolution, but the data won't download from NOMIS due to being too large otherwise.


### Geography 
This data contains commuting flows between patches within England & Wales as of the 2011 Census Day.
In small patches, individuals are sometimes swapped between patches to anonymise data, however at the LAD resolution
this should not be a problem.

## Commuter Flows
### Source

Obtained from: "WF02EW - Location of usual residence and place of work (with outside UK
collapsed) (OA/WPZ level)".

Location: `WF02EW LAD2011.csv`.

Also find here: https://www.nomisweb.co.uk/census/2011/wf02ew

### Column, or row, wait what's happening?

I keep getting confused which way round the data is orientated, so this is here:
Row is home, Column is workplace (can be verified by looking at the original dataset which has this labelled).

So $C_{ij}$ is the number of commuters from $i$ to $j$, and the total number of individuals in $i$ that commute is $C_i = \sum_j C_{ij}$.

## Population 
### Source
Obtained from: "QS102EW - Population density".

Location: `QS102EW LAD2011.csv`.

Also find here: https://www.nomisweb.co.uk/census/2011/qs102ew