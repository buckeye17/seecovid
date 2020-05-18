# seecovid
This is a data science portfolio project in which plotly|Dash was used to visualize COVID-19 data.

More information about the finished app can be found on my portfolio website: https://buckeye17.github.io/COVID-Dashboard/

The deployed app can be found here: https://seecovid.herokuapp.com/

# Replication
The local Python environment can be re-created either using all_requirements.txt with pip or using environment.yml with conda.  

The requirements.txt file is used by Heroku to build the required containerized environment for the heroku app.  The heroku_gitignore.txt file is meant to define the files to be ignored for heroku deployment.

The github_gitignore.txt is meant to define the files to be ignored when archiving on Github.

# Files & Folders
The Procfile is needed for heroku deployment.  It tells the linux containerized environment how to start the app.

app.py fully defines the Python Dash app.

data_clean folder contains:
 * Clean Pandas dataframes as pickle files.  These files provide COVID data & population data.
 * Geo JSON files used for heat map region boundaries.
 * An initial plotly figure data structure for the heatmap seen when landing on the site.  This seems to reduce initial loading time from 25 seconds to 15 seconds!

data_raw folder contains:
* csv files as well as xlsx files used to generate csv files.  This is largely population data copied and pasted from Wikipedia tables.  The sub-folder canada_provinces contains raw Geo JSON files for Canada.

images folder contains static images used by the app.

jupyter_notebooks folder contains:
* PopulationData notebook to transform raw population data into clean Pandas dataframe pickle files.
* Covid19_Extract_Transform notebook uses the clean population data and downloads Johns Hopkins COVID data to produce a clean Pandas dataframe pickle file for COVID data.
* Covid19_Visualization3 was used to initially explore plotly visualizations which were eventually combined into the Dash app.

secret_credentials is a folder omitted from this Github repository because it contains authentication keys:
* .mapbox_token provides a key for accessing mapbox maps used for all heat map figures.
* client_secrets.json provides Google Drive access
* github_token.txt provides access to Github, enabling all the desired files in the Johns Hopkins COVID respository to be downloaded
