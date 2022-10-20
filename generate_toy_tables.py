import pandas as pd
import numpy as np
import random
from scipy import stats
from itertools import combinations_with_replacement


POSSIBLE_IDS = list(x+y for x,y in combinations_with_replacement("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 2))
CLASSES = "ABCDEF"

def main():
    np.random.seed(0)
    random.seed(0)

    for class_ in CLASSES:
        # Create master table for class
        master = pd.DataFrame()
        master['id'] = random.sample(POSSIBLE_IDS, 30)
        master['value'] = np.random.uniform(-1, 1, size=len(master['id']))
        master.to_csv(f"toy_tables/master/master_{class_}.csv", index=False, sep=";")


        # Create 10 correlated tables for class
        for i in range(10):
            scale = np.random.uniform(-2, 2)
            
            table = master.sample(frac=np.random.uniform(0.1, 1)).reset_index(drop=True)
            
            table['value'] += np.random.uniform(-1, 1, size=len(table))
            table['value'] = table['value'] * scale

            table.to_csv(f"toy_tables/{class_}_{i}.csv", index=False, sep=";")
            sanity = master.merge(table, on='id', how='inner', suffixes=('', '_y'))
            print(scale, stats.pearsonr(sanity['value'], sanity['value_y']))
        
        # Create 10 uncorrelated tables for class
        for i in range(10):
            scale = np.random.uniform(-2, 2)
            table = master.sample(frac=np.random.uniform(0.1, 1)).reset_index(drop=True)

            table['value'] = np.random.uniform(-1, 1, size=len(table))

            table.to_csv(f"toy_tables/{class_}_u_{i}.csv", index=False, sep=";")
            sanity = master.merge(table, on='id', how='inner', suffixes=('', '_y'))
            print(stats.pearsonr(sanity['value'], sanity['value_y']))



if __name__ == '__main__':
    main()