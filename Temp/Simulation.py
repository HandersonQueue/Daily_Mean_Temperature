import numpy as np
from matplotlib import pyplot as plt


# 这个代码是用来根据实际数据生成相应的模拟数据的，基本思路就是根据概率函数逆映射得到满足相应分布的数据，在使用时需要指定生成的数量，
def simulation_temp(avg_temp, var_temp, years, which_day=np.array(range(365))):
    # 生成一组符合正态分布的数据
    # 如果不指定的话，就调用生成相应数组
    which_day_size = len(which_day)
    x = np.zeros([which_day_size, years])
    for j in range(which_day_size):
        # 生成指定年份数量的数据
        x[j] = np.random.normal(avg_temp[which_day[j]], var_temp[which_day[j]], years)
    print(f"The Daily Mean Air Temperature of {years} years have been simulated")

    fig_simu, ax_simu = plt.subplots(figsize=(23.5, 10))
    rnd_seed = np.random
    for j in range(years):
        rnd_seed.seed(j)
        ax_simu.scatter(which_day+1, x[:, j], marker='o', facecolor='none',
                        color=(rnd_seed.random(), rnd_seed.random(), rnd_seed.random()))
    # change the size
    # 改变坐标轴字号大小
    ax_simu.tick_params(axis='x', labelsize=14)
    ax_simu.tick_params(axis='y', labelsize=14)
    # 设置横纵坐标轴标注
    ax_simu.set_xlabel('Time/ d', fontsize=18)
    ax_simu.set_ylabel('Daily Mean Temperature/ ℃', fontsize=18)
    # 图名
    ax_simu.set_title('The Scatters of Daily Mean Temperature Simulated', fontsize=18)
    # plt.savefig('Simulation.png', bbox_inches='tight', dpi=300)
    plt.show()

    return x
