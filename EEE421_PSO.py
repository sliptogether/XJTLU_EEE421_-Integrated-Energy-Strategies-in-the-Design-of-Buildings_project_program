# -*- codeing = utf-8 -*-
# @Time : 2023/11/15 10:47
# @Author : ZhangQH
# @File : PSO.py
# @Software: PyCharm


import numpy as np
import matplotlib.pyplot as plt

#定义一些变量
Energy_Consume = 76.49 # 能源消耗预测
Tariff = 0.25 # 电价
Floor_Area = 40 # 面积
PV_Area = 1.74 # 一块PV的面积
Cost_Forecasts = Energy_Consume * Tariff
Discount_rate = 0.05 # 折损率
# Interest_Rate = 0.2 # 投资利率

# lifetime
wall_insulation_lifetime = 50
heating_system_lifetime = 6
glazing_lifetime = 20
pv_lifetime = 10

# Functions for future value and one-year cost
def future_value(cost, interest_rate, years):
    return cost * ((1 + interest_rate) ** years)

def one_year_cost(investment, discount_rate, lifetime):
    return (investment * ((1 + discount_rate) ** lifetime)) / lifetime
# 粒子群
class Particle:
    #初始化粒子的位置、速度和适应度
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=dim) # 初始位置
        self.velocity = np.random.uniform(low=-1, high=1, size=dim) # 初始速度
        self.best_position = np.copy(self.position) # 将粒子最佳位置复制过来
        self.fitness = self.calculate_fitness() # 初始适应度 calculate_fitness函数见下
        self.best_fitness = self.fitness # 将粒子的初始适应度值赋给 self.best_fitness，表示粒子当前的最佳适应度。

    # 计算粒子的适应度
    def calculate_fitness(self):
        # 定义不同的决策变量
        wall_insulation_costs = [4000, 6000, 10000, 14000]
        heating_system_costs = [2000, 3000, 4500, 6000]
        glazing_costs = [1200, 3300, 4500, 6500]
        pv_costs = [0, 2000, 2500, 3000, 4000, 6000, 7000, 9000, 10000]

        total_initial_cost = sum([
            wall_insulation_costs[int(self.position[0])],
            heating_system_costs[int(self.position[1])],
            glazing_costs[int(self.position[2])],
            pv_costs[int(self.position[3])]
        ])

        # Future values and one-year costs
        fv_wall_insulation = future_value(wall_insulation_costs[int(self.position[0])], Discount_rate,
                                          wall_insulation_lifetime)
        fv_heating_system = future_value(heating_system_costs[int(self.position[1])], Discount_rate, heating_system_lifetime)
        fv_glazing = future_value(glazing_costs[int(self.position[2])], Discount_rate, glazing_lifetime)
        fv_pv = future_value(pv_costs[int(self.position[3])], Discount_rate, pv_lifetime)

        # wall_oneyear_cost
        wall_accumulation = 0
        for year in range(1, wall_insulation_lifetime + 1):
            accumulation = wall_insulation_costs[int(self.position[0])] * (1 + Discount_rate) ** year / year
            wall_accumulation += accumulation
        # heating_oneyear_cost
        heat_accumulation = 0
        for year in range(1, heating_system_lifetime + 1):
            accumulation = heating_system_costs[int(self.position[1])] * (1 + Discount_rate) ** year / year
            heat_accumulation += accumulation
        # glazing_oneyear cost
        glazing_accumulation = 0
        for year in range(1, glazing_lifetime + 1):
            accumulation = glazing_costs[int(self.position[2])] * (1 + Discount_rate) ** year / year
            glazing_accumulation += accumulation
        # pv_oneyear cost
        pv_accumulation = 0
        for year in range(1, pv_lifetime + 1):
            accumulation = pv_costs[int(self.position[3])] * (1 + Discount_rate) ** year / year
            pv_accumulation += accumulation

        """   
        one_year_wall_insulation = one_year_cost(wall_insulation_costs[int(self.position[0])], Discount_rate,
                                                 wall_insulation_lifetime)
        one_year_heating_system = one_year_cost(heating_system_costs[int(self.position[1])], Discount_rate,
                                                heating_system_lifetime)
        one_year_glazing = one_year_cost(glazing_costs[int(self.position[2])], Discount_rate, glazing_lifetime)
        one_year_pv = one_year_cost(pv_costs[int(self.position[3])], Discount_rate, pv_lifetime)
        """
        # 能耗节约
        Energy_Savings = sum(self.position[:3]) * Floor_Area + self.position[3] * 320 / 1000

        # 惩罚
        penalty = max(0, Energy_Consume - Energy_Savings) * 10000

        #return (one_year_wall_insulation + one_year_heating_system + one_year_glazing + one_year_pv -
        #        fv_wall_insulation - fv_heating_system - fv_glazing - fv_pv + total_initial_cost + penalty
        #         ) # 数值越低能耗成本越低
        return (wall_accumulation + heat_accumulation + glazing_accumulation + pv_accumulation -
                fv_wall_insulation - fv_heating_system - fv_glazing - fv_pv + total_initial_cost + penalty
                )  # 数值越低能耗成本越低

# PSO模型
class PSO:
    #在粒子群类中初始化算法参数和粒子群
    def __init__(self, dim, bounds, num_particles, max_iter):
        self.dim = dim # 决策变量维度
        self.bounds = bounds # 限制搜索空间
        self.num_particles = num_particles # 粒子的数量
        self.max_iter = max_iter # 最大迭代次数
        self.gbest_value = float('inf') # 保存全局最佳适应度值
        self.gbest_position = np.zeros(dim) # 保存全局最佳位置
        self.swarm = [Particle(dim, bounds) for _ in range(num_particles)] # 粒子的初始群体

    # 优化函数
    def optimize(self):
        for t in range(self.max_iter):
            for particle in self.swarm:
                particle.fitness = particle.calculate_fitness() # 适应度值越小表示性能越好

                # 检查当前位置是否是粒子的最佳位置
                if particle.fitness < particle.best_fitness:
                    particle.best_position = np.copy(particle.position)
                    particle.best_fitness = particle.fitness

                # 检查当前位置是否是有史以来全局的最佳位置
                if particle.fitness < self.gbest_value:
                    self.gbest_value = particle.fitness
                    self.gbest_position = np.copy(particle.position)

            # 更新粒子的速度和位置
            for particle in self.swarm:
                inertia_weight = 0.5
                cognitive_weight = 1.0
                social_weight = 1.0

                new_velocity = (inertia_weight * particle.velocity +
                                cognitive_weight * np.random.rand(self.dim) * (particle.best_position - particle.position) +
                                social_weight * np.random.rand(self.dim) * (self.gbest_position - particle.position))
                particle.velocity = new_velocity

                # 用新速度更新位置
                particle.position += new_velocity

                # 应用边界条件
                particle.position = np.clip(particle.position, self.bounds[:, 0], self.bounds[:, 1])

            # 保存每次迭代的最佳值
            yield self.gbest_value

# 定义每个决策变量的界限
bounds = np.array([[0, 3], [0, 3], [0, 3], [0, 8]])

# 创建 PSO 实例
num_dimensions = 4 # 维度
num_particles = 30 # 粒子数量
max_iterations = 100 # 迭代次数

# 运行pso
pso_instance = PSO(num_dimensions, bounds, num_particles, max_iterations)
best_cost_progression = list(pso_instance.optimize())

# 画图
plt.figure(figsize=(10, 6))
plt.plot(best_cost_progression, 'b-', label='Best Cost')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Best Cost Progression')
plt.legend()
plt.grid(True)
plt.show()

# 显示找到的最佳解决方案
print("Global Best Position:", pso_instance.gbest_position)
print("Global Best Value:", pso_instance.gbest_value)

# 打印出 PSO 找到的最优解的详细信息
optimal_solution_details = f"""
Optimal Solution Details:
- Wall Insulation Level: {round(pso_instance.gbest_position[0])}
- Heating System Level: {round(pso_instance.gbest_position[1])}
- Glazing Level: {round(pso_instance.gbest_position[2])}
- Number of PV Panels: {round(pso_instance.gbest_position[3])}
- Minimum Cost (Global Best Value): {pso_instance.gbest_value:.2f}
"""
print(optimal_solution_details)