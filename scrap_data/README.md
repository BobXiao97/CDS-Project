## This folder contains the files we used to scrap the data and data collected from these scripts

-   `scrap_data.ipynb`:
    - This file scraps spotify data from spotify top 200 charts daily. 
    - The range of date to scrap the data from is the input to the main function (we scrap from 2017-01-01 to 2020-10-25)
    - The output will be  the raw data files downloaded from spotify with names eg. `2020-10-25.csv`
    - the outputfile will then be further processed by adding audio features as new columns of the csv files and giving them the name eg. `feature_2020-10-25.csv`

-   `label_data_top30.ipynb`:
    -  This is the file we use to combine all the csv files with audio features eg. `feature_2020-10-25.csv` and give each song a label based on whether a song had ever made it to the top 30 of the chart
    -   Although we did not use this criteria eventually, the dataset we use now is built upon output of this file as this combines all the unique songs across songs collected for all 3 years and keep the highest possible `Streams` value obtained by each song 
    - The labeled output file, combining all `feature_yy_mm_dd.csv` files is named as `labeled_spotify_data.csv`
    - The `labeled_spotify_data.csv`, after adding genre and sorting based on the `Streams` to test the performance for various models, it is renamed to `labeled_spotify_data_genre_sort_streams.csv` which we used it later

-   `combine_dataset_w_kaggle.ipynb`:
    -   This file is used to combine the file `labeled_spotify_data_genre_sort_streams.csv` and the kaggle dataset `dataset-of-10s.csv` we found on kaggle
    -   3000 non-hit songs were taken from  `dataset-of-10s.csv` and labeled as non-hit (`Label=0`)
    -   top 3000 songs with highest streams from `labeled_spotify_data_genre_sort_streams.csv` we taken and labeled as hit-song (`Label=1`)
    -   The combined file is saved as `song_data_combine.csv`
-   `scrap_genre.ipynb`:
    -    This file is used to scrap the genre for each song in `song_data_combine.csv` and add them into one column as a string separated by `,`
    -    The output file is saved as `song_data_combined_genre.csv`

-   `label_genre.ipynb`:
    -   Given all the genres as one string in `song_data_combined_genre.csv` , this file filter out the top 10 most frequently appearing genre and use each of them as a feature (column in the csv file)
    -   If a song contains a particular genre in the columns, that column will be labeled as `1`. `0`, otherwise
    - The output of the file is saved as `song_data_combined_genre_label_final.csv`

After severial rounds of developing and testing on various dataset. We will eventually be building and comparing our model on 
    
-   `song_data_combined_genre.csv` - the dataset without Genres being used as feature
-   `song_data_combined_genre_label_final.csv` - the dataset with Genres being used as feature

the `spotify_data.zip` is a zip folder storing all the `yyyy-mm-dd.csv` and `feature_yyyy-mm-dd.csv` produced by `scrap_data.ipynb`
