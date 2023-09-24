import os

os.mkdir(os.path.join('.', '01_Data'))
data_file = os.path.join('.', '01_Data', '01_house_tiny.csv')
print(data_file)

with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
