import pandas as pd

datasetlist = ['Data_Actind_Term_epi.pkl' , 'Data_Actind_Term_epi_test.pkl' , 'ubduces_test.pkl' , 'Transition_Data_Discrete.pkl' , 'Transition_Data_Discrete_test.pkl' , 'Transition_Data.pkl' , 'Transition_Data_test.pkl']

def generate_overview(filename):
    dataframe = pd.read_pickle(filename)
    print(dataframe[0][0])
    overview = {
        'Dataset': filename,
        'Column Names': dataframe.columns.tolist(),
        'Sample Count': len(dataframe),
        'Data Types': dataframe.dtypes.to_dict(),
        'Summary Statistics': dataframe.describe().to_dict()
    }

    for key, value in overview.items():
        print(f"{key}:")
        print(value)
        print("\n")

for d in datasetlist:
    generate_overview(d)