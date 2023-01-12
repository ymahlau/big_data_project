#!/bin/bash

for b in 512 128 32
do
    for c in 512 128 32
    do
        python3 -m main.main_search_index git 50 qcr data/real_data/song_data.csv song_name song_popularity $c $b -c song
    done
done

# python3 -m main.main_search_index git 10 qcr data/real_data/movie_dataset.csv Title Popularity 512 512
# python3 -m main.main_search_index git 10 qcr data/real_data/song_data.csv song_name song_popularity $c $b -c song
# python3 -m main.main_search_index git 10 qcr expert_review/long_query.csv Country "Life expectancy" $c $b -c who
