# Deprecated (see 2011 instead)
2021 data was preliminarily used for a commuter flow matrix. 2021 was a weird year, especially for a census, see here: 
https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/methodologies/traveltoworkqualityinformationforcensus2021

As such, the data from 2011 was deemed more reliable, so was used. This is here so I can switch all relevant work over to 2011, and still have 2021 data for later (might be interesting to contrast models on both matrices.

# 2021 Commuter Data

Commuting data is preprocessed in a Colab notebook from a previous project on kernel homogeneity. The original data sources are below, as well as a link to the notebook.

**Commuter flow data:**

https://www.nomisweb.co.uk/sources/census_2021_od

Using dataset ODWP01EW, LTLA geography.

**Geography data**:

https://geoportal.statistics.gov.uk/datasets/ons::local-authority-districts-december-2021-boundaries-gb-bfc/about

**Preprocessing notebook**:

https://colab.research.google.com/drive/1GuFEJxbzmVI5kIyNtBUAmIjcSZdUTxpn

**Patch census data**:

We just want the people who are registered on the census, so we sum down columns.