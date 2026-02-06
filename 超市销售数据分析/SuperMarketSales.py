
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# 打开csv文件
data = open('超市销售数据集.csv')
# pandas读取csv文件
data_pd = pd.read_csv('超市销售数据集.csv')



# 清洗数据，删除空行空列
def data_clean():
    data_pd.dropna(axis = 0, how = 'all')
    data_pd.dropna(axis = 1, how = 'all')


# 场景1：ABC三个分支机构的销售总额
def branch_total():
    data_clean()

    # 转化为numpy数组
    data_np = np.array(data_pd)

    # 取出需要的列组成新的数组
    new_np = data_np[:, [1, 9]]

    # 定义函数，计算total的sum
    def count(X):
        sum = 0
        for i in range(len(new_np)):
            if new_np[i, 0] in X:
                sum += new_np[i, 1]
        return sum
    
    # 创建新的数组
    branch_total_np = [['A', count('A')], ['B', count('B')], ['C', count('C')]]
    header = ['Branch', 'Total']

    # 转换为DataFrame
    branch_total_pd = pd.DataFrame(branch_total_np, columns = header)

    # 条形图实现
    sns.set_style('whitegrid')
    # 可以输出中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize = (10, 6))
    bar_plot = sns.barplot(x = 'Branch', y = 'Total', data = branch_total_pd, palette = 'viridis')
    plt.title('ABC三个分支机构的销售总额')
    plt.xlabel('分支机构')
    plt.ylabel('总计')
    # 将数字显示在图表上
    for p in bar_plot.patches:
        bar_plot.annotate(int(p.get_height()), (p.get_x() + 0.375, p.get_height() + 1), ha = 'center', va = 'bottom')
    plt.show()

# branch_total()

# 场景2：ABC三个分支机构在不同月份的销售趋势
def branch_date_total():
    data_clean()

    # 转化为numpy数组
    data_np = np.array(data_pd)

    # 取出需要的列组成新的数组
    new_np = data_np[:, [1, 9, 10]]

    # 定义一个函数，分别计算ABC三个分支机构在三个月份的销售额
    # count(分支机构,月份)
    def count(X, y):
        # 定义一个数组，分别容纳三个分支机构的数据(total, date)，将相应的数据筛选出来
        array1 = []
        for i in range(len(new_np)):
            if new_np[i, 0] in X:
                array1.append([new_np[i, 1], new_np[i, 2]])

        # 再定义一个数组，分别容纳三个月的数据(total, date)，将相应的数据筛选出来
        array2 = []
        for i, j in array1:
            array1 = datetime.strptime(j, '%m/%d/%Y')
            if array1.month == y:
                array2.append([i, j])

        # 将array2的数据total求和
        sum = 0
        for i, j in array2:
            sum += i   
        return sum

    # 创建新的字典
    branch_date_total_np = {'A': {1: count('A', 1), 2: count('A', 2), 3: count('A', 3)}, 
                            'B': {1: count('B', 1), 2: count('B', 2), 3: count('B', 3)}, 
                            'C': {1: count('C', 1), 2: count('C', 2), 3: count('C', 3)}}

    # 数据转换为DataFrame
    branch_date_total_dp = pd.DataFrame(branch_date_total_np)
    
    branch_date_total_dp = branch_date_total_dp.reset_index().rename(columns = {'index': '月份'})

    # 折线图实现
    # 可以输出中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize = (10, 6))
    for branch in branch_date_total_dp.columns[1:]:
        plt.plot(branch_date_total_dp['月份'], branch_date_total_dp[branch], marker = 'o', label = f'分支机构 {branch}')
        # 将数据显示在图表上
        for i, txt in enumerate(branch_date_total_dp[branch]):
            plt.text(branch_date_total_dp['月份'][i], txt, f'{int(txt)}', ha = 'center', va = 'bottom')
    # sns.lineplot(x = 'Month', y = 'Total', hue = 'Branch', data = branch_date_total_dp, marker = 'o')
    plt.title('不同分支机构在不同的月份的销售趋势')
    plt.xlabel('月份')
    plt.ylabel('总计')
    plt.xticks([1, 2, 3], ['1月', '2月', '3月'])
    plt.grid(True)
    plt.legend()
    plt.show()

# branch_date_total()

# 场景3：不同城市的客户评分分布
def city_rating():
    data_clean()

    # 创建透视表，按城市和评分进行计数
    pivoit_table = pd.pivot_table(data_pd, values = 'Invoice ID', index = 'City', columns = 'Rating', aggfunc = 'count', fill_value = 0)
    print(pivoit_table)

    # 热力图实现
    # 可以输出中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize = (10, 6))
    sns.heatmap(pivoit_table, annot = True, fmt = 'd', cmap = 'YlGnBu')
    plt.title('不同城市的客户评分分布')
    plt.xlabel('评分')
    plt.ylabel('所在城市')
    plt.show()

# city_rating()

# 场景4：不同产品线的销售额占比
def productline_total():
    data_clean()

    # 计算不同产品线的销售总额
    productline_total_pd = data_pd.groupby('Product line')['Total'].sum()
    print(productline_total_pd)

    # 饼图实现
    # 可以输出中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize = (10, 6))
    plt.pie(productline_total_pd, labels = productline_total_pd.index, autopct = '%1.1f%%', startangle = 140, colors = sns.color_palette("pastel"))
    plt.title('不同产品线的销售额占比')
    plt.axis('equal')
    plt.show()

# productline_total()

# 场景5：超市销售额在不同时间段的分布
def time_total():
    data_clean()

    # 确保时间一列是x:x类型
    data_pd['Time'] = pd.to_datetime(data_pd['Time'], format = '%H:%M').dt.time

    # 提取小时和分钟的数据
    data_pd['Hour'] = data_pd['Time'].apply(lambda x:x.hour)
    data_pd['Minute'] = data_pd['Time'].apply(lambda x:x.minute)

    # 创建一个新列，将原来的x:x的格式转换成浮点数
    data_pd['Time_float'] = data_pd['Hour'] + data_pd['Minute'] / 60.0

    print(data_pd['Time_float'])

    # 散点图实现
    # 可以输出中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize = (10, 6))
    sns.scatterplot(x = 'Time_float', y = 'Total', data = data_pd, alpha = 0.8)
    plt.title('不同时间段的销售额分布')
    plt.xlabel('小时')
    plt.ylabel('总计')
    plt.xticks(range(1, 25))
    plt.grid(True)
    plt.show()

time_total()

# 场景6：不同客户类型在各个城市的分布情况
def customertype_branch_city():
    data_clean()

    # 创建透视表，按客户类型和所在城市进行计数
    pivot_table = pd.pivot_table(data_pd, values = 'Invoice ID', index = 'Customer type', columns = 'City', aggfunc = 'count', fill_value = 0)
    print(pivot_table)

    # 堆叠柱状图实现
    # 可以输出中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    ax = pivot_table.plot(kind = 'barh',figsize = (10, 6), stacked = True, color = sns.color_palette("pastel"))
    plt.title('不同客户类型在各个城市的分布情况')
    plt.xlabel('数量')
    plt.ylabel('客户类型')
    plt.legend(title = '所在城市')
    # 将数字显示在图表上
    for container in ax.containers:
        ax.bar_label(container, label_type = 'center')
    plt.show()

# customertype_branch_city()

# 场景7：不同性别的客户在消费上的差异
def gender_total():
    data_clean()

    # 计算不同性别贡献的销售额
    gender_pd = data_pd.groupby('Gender')['Total'].sum()
    print(gender_pd)

    # 饼图实现
    # 可以输出中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize = (10, 6))
    plt.pie(gender_pd, labels = gender_pd.index, autopct = '%1.1f%%', startangle = 140, colors = sns.color_palette("pastel"))
    plt.title('不同性别的客户在消费上的差异')
    plt.axis('equal')
    plt.show()

# gender_total()

# 场景8：不同付款方式的使用频率
def payment():
    data_clean()

    #  统计每种付款方式的数量
    payment_pd = data_pd['Payment'].value_counts()
    print(payment_pd)

    # 条形图实现
    # 可以输出中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize = (10, 6))
    bar_plot = payment_pd.plot(kind = 'barh', color = ['skyblue', 'lightgreen', 'lightcoral'])
    plt.title('不同付款方式的使用频率')
    plt.xlabel('数量')
    plt.ylabel('付款方式')
    plt.grid(axis = 'y')
    # 将数字显示在图表上
    for p in bar_plot.patches:
        bar_plot.annotate(p.get_width(), (p.get_width(), p.get_y() + 0.25), ha = 'left', va = 'center')
    plt.show()

# payment()