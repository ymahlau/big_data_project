import pandas as pd
from scipy import stats

def main():
    input = loadDB('TestData/Input.csv')
    tables = []
    ### import other tables
    for table in {"TestData/AnimalsWeight.csv","TestData/CauseOfDeath.csv","TestData/Mascots.csv"}:
        tables.append(loadDB(table))
    ### select input_column
    input_column = input.select_dtypes(include=['object']).columns
    input_column = input_column[0]
    feature_column = input.select_dtypes(exclude=['object']).columns
    feature_column = feature_column[0]

    result = pd.DataFrame(columns=['InputFeature','JoinableFeature','Joinability','PearsonCoefficient'])

    for t in tables:
        joinability = (0,)
        ### check tables for joinable columns

        for c in t.select_dtypes(include=['object']).columns:
            tempJoinability = len(set(input[input_column]).intersection((set(t[c]))))/len(set(input[input_column]))
            if (joinability[0]<tempJoinability):
                joinability = (tempJoinability,c)

        ### calculate pearon coefficient for tables with more than 50% joinable rows
        if (joinability[0] > 0.5):

            inputSet = set(input[input_column])
            cSet = set(t[joinability[1]])
            filterSet = inputSet.intersection(cSet)
            filterInput = input[input[input_column].isin(filterSet)]
            filterTable = t[t[joinability[1]].isin(filterSet)]
            filterInput = filterInput.sort_values(by=[input_column])
            filterTable = filterTable.sort_values(by=[joinability[1]])

            for cNumeric in filterTable.select_dtypes(exclude=['object']).columns:
                   numericC = filterTable[cNumeric]
                   inputC = filterInput[feature_column]
                   pearson = stats.pearsonr(numericC, inputC)
                   result.loc[len(result.index)] = [feature_column, joinability[1] ,joinability[0], pearson]
    print(result)              
                    

def loadDB(path):
    df = pd.read_csv(path, sep=";")
    return df

if __name__ == '__main__':
    main()
