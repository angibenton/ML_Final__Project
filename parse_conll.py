import pandas as pd
from tweebo_parser import API, ServerError

filename = input('Enter the file name/path from here: ')
start = int(input('Enter row number to start from (inclusive): '))
end = int(input('Enter row number to end at (exclusive): '))
out = input('Enter filename of output csv: ')

file = pd.read_csv(filename, index_col=False)

file = file.iloc[start:end, :]

tweets = file['tweet'].tolist()

tweebo_api = API()
try:
    result = tweebo_api.parse_conll(tweets)
except ServerError as e:
    print(f'{e}\n{e.message}')

file['CoNLL'] = result

file.to_csv(out, index = False)
