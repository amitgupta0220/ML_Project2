from data import data_set

new_data_set = [[row[i] for i in range(len(row)-1)] for row in data_set]

with open('new_data.txt', 'w') as file:
    for row in new_data_set:
        row_str = [str(i) for i in row]
        row_line = '[' + ', '.join(row_str) + ']'
        file.write(row_line + ',')
