from data_loader import DataLoader



x = DataLoader('./obama_data/')

y = x.preprocess('./data_saves')

print(y)
