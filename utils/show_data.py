import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#
# NODES = [2, 10, 11, 12, 13, 21, 22, 23, 31, 32]  # 节点编号
# LINKS = [9, 10, 11, 12, 21, 22, 31, 110, 111, 112, 113, 121, 122]  # 节点编号


NODES = [i for i in range(1, 15)]  # 节点编号
LINKS = [i for i in range(1, 15)]  # 管道编号

#
#
# def plot_data(data, i, root_path="F:\\渗漏数据集\\Net1_CMH\\Node"):
#     # 定义颜色列表
#     colors = [
#         "blue",
#         "orange",
#         "green",
#         "red",
#         "purple",
#         "brown",
#         "pink",
#         "gray",
#         "olive",
#         "cyan",
#         "magenta",
#     ]
#
#     # 绘制压力图像
#     plt.figure(figsize=(15, 15))
#     plt.rcParams["font.size"] = 10
#     for j in range(len(nodes)):
#
#         plt.subplot(10, 1, j + 1)
#         plt.plot(
#             data[nodes[j]][0],
#             label=f"Node-{nodes[j]}-Press",
#             color=colors[j % len(colors)],
#         )  # 指定线的颜色
#         plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
#     plt.suptitle(f"Scenario-{i+1}")
#     plt.savefig(os.path.join(root_path, f"Scenario-{i+1}-Press.png"))
#     plt.close()
#
#     # 绘制水需求图像
#     plt.figure(figsize=(15, 15))
#     plt.rcParams["font.size"] = 10
#     for j in range(len(nodes)):
#         plt.subplot(10, 1, j + 1)
#         plt.plot(
#             data[nodes[j]][1],
#             label=f"Node-{nodes[j]}-Demand",
#             color=colors[j % len(colors)],
#         )  # 指定线的颜色
#         plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
#     plt.suptitle(f"Scenario-{i+1}")
#     plt.savefig(os.path.join(root_path, f"Scenario-{i+1}-Demand.png"))
#     plt.close()
#
#
# # 绘制压力和水需求图像
# def draw_node_value():
#
#     for i in range(1000):  # 1000个场景
#         print(f"Scenario-{i+1}")
#         # 获取每个节点的数据
#         data = {}
#         for node in nodes:
#             file = f"Node_{str(node)}.csv"
#             fp = []
#             for ty in [
#                 "Pressures",
#                 "Demands",
#             ]:
#                 dir = os.path.join(
#                     f"F:\\渗漏数据集\\Net1_CMH\\Scenario-{str(i+1)}\\", ty
#                 )  # 流量 & 压力 数据路径
#                 file_path = os.path.join(dir, file)
#                 df = pd.read_csv(file_path, header=None)
#                 # 转换为numpy数组
#                 fp.append(df.values[1:, 1].astype(np.float32))
#             # 添加到data字典中
#             data[node] = fp
#         # 绘制图像
#         plot_data(data, i)
#
#
# def plot_edge(data, i, root_path="F:\\渗漏数据集\\Net1_CMH\\"):
#     # 定义颜色列表
#     colors = [
#         "blue",
#         "orange",
#         "green",
#         "red",
#         "purple",
#         "brown",
#         "pink",
#         "gray",
#         "olive",
#         "cyan",
#         "magenta",
#         "yellow",
#         "black",
#         "silver",
#     ]
#     # 绘制图像
#     plt.figure(figsize=(15, 15))
#     plt.rcParams["font.size"] = 10
#     for j in range(1, len(links)):
#         plt.subplot(13, 1, j + 1)
#         plt.plot(
#             data[links[j]][0], label=f"link-{links[j]}", color=colors[j % len(colors)]
#         )  # 指定线的颜色
#         plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
#     plt.suptitle(f"Scenario-{i+1}")
#     plt.savefig(os.path.join(root_path, f"Scenario-{i+1}-Flow.png"))
#     plt.close()
#
#
# # 绘制流量图像
# def draw_edge_value():
#     for i in range(1000):  # 1000个场景
#         print(f"Scenario-{i+1}")
#         # 获取每个节点的数据
#         data = {}
#         for link in links:
#             file = f"Link_{str(link)}.csv"
#             fp = []
#             for ty in [
#                 "Flows",
#             ]:
#                 dir = os.path.join(
#                     f"F:\\渗漏数据集\\Net1_CMH\\Scenario-{str(i+1)}\\", ty
#                 )  # 流量 & 压力 数据路径
#                 file_path = os.path.join(dir, file)
#                 df = pd.read_csv(file_path, header=None)
#                 # 转换为numpy数组
#                 fp.append(df.values[1:, 1].astype(np.float32))
#             # 添加到data字典中
#             data[link] = fp
#         # 绘制图像
#         plot_edge(data, i, root_path="F:\\渗漏数据集\\Net1_CMH\\Flows\\")
#

DATA_DIR = "F:\\渗漏数据集\\Hanoi_CMH\\"
SCENARIO = "Scenario-"
PRESSURE_DIR = "Pressures"
DEMAND_DIR = "Demands"
LEAK_DIR = "Leaks"
FLOWS_DIR = "Flows"


class Leak:
    def __init__(self, scenario_id, leak_id):
        self.leak_id = leak_id

        self._leak_demand = []
        self._leak_info = {}
        self._initialize(scenario_id)

    def _initialize(self, scenario_id):
        scenario_id = str(scenario_id)
        leaks_dir = os.path.join(DATA_DIR, SCENARIO + scenario_id, LEAK_DIR)
        #  读取leak_demand
        leak_demand_file = os.path.join(leaks_dir, f"Leak_{self.leak_id}_demand.csv")
        with open(leak_demand_file, "r") as f:
            f.readline()
            for line in f.readlines():
                time, demand = line.strip().split(",")
                self._leak_demand.append(float(demand))
        # 读取leak_info
        leak_info_file = os.path.join(leaks_dir, f"Leak_{self.leak_id}_info.csv")
        with open(leak_info_file, "r") as f:
            f.readline()
            f.readline()
            for line in f.readlines():
                info, val = line.strip().split(",")
                info = info.strip()
                val = val.strip()
                self._leak_info[info] = val
        pass


class Node:
    def __init__(self, scenario_id, node_id):
        self.node_id = node_id
        self._press = []
        self._demand = []
        self._initialize(scenario_id)

    def get_press(self):
        return tuple(self._press)

    def get_demand(self):
        return tuple(self._demand)

    def _initialize(self, scenario_id: str):
        scenario_id = str(scenario_id)
        # 读取压力文件
        press_file = os.path.join(
            DATA_DIR,
            SCENARIO + scenario_id,
            PRESSURE_DIR,
            f"Node_{str(self.node_id)}.csv",
        )
        with open(press_file, "r") as f:
            f.readline()
            for line in f.readlines():
                self._press.append(float(line.strip().split(",")[1]))
        # 读取水需求文件
        demand_file = os.path.join(
            DATA_DIR,
            SCENARIO + scenario_id,
            DEMAND_DIR,
            f"Node_{str(self.node_id)}.csv",
        )
        with open(demand_file, "r") as f:
            f.readline()
            for line in f.readlines():
                self._demand.append(float(line.strip().split(",")[1]))
        pass


def draw(data, scope, cls, scenario_id, leak_id=None):
    # 在子图中绘制图像
    plt.figure(figsize=(30, 30))
    plt.rcParams["font.size"] = 8
    for j in range(len(data)):
        plt.subplot(len(data), 1, j + 1)
        plt.plot(
            range(scope[2], scope[3]),
            data[j][scope[2] : scope[3]],
            label=f"Node-{NODES[j]}-{cls}",
            color="orange" if leak_id == NODES[j] else "blue",
        )
        # 设置x轴范围
        plt.xlim(scope[2], scope[3])
        # 绘制起始和结束点
        plt.axvline(x=scope[0], color="red", linestyle="--", label="Leak Scope")
        plt.axvline(x=scope[1], color="red", linestyle="--")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.suptitle(f"Scenario-{scenario_id}")
    plt.subplots_adjust(hspace=1)
    plt.show()
    pass


if __name__ == "__main__":
    nodes = []
    scenario_id = 15
    leak_id = 3
    leak_s3_13 = Leak(scenario_id, leak_id)
    for i in NODES:
        node = Node(scenario_id, i)
        nodes.append(node)
    # press data
    press_data = []
    for node in nodes:
        press_data.append(node.get_press())
    # demand data
    demand_data = []
    for node in nodes:
        demand_data.append(node.get_demand())
    # scope of data
    start = int(leak_s3_13._leak_info["Leak Start"])
    end = int(leak_s3_13._leak_info["Leak End"])
    range_value = 1000
    scope = [
        start,
        end,
        start - range_value if start - range_value > 0 else 0,
        start + range_value if start + range_value < 17520 else 17519,
        # 0, 17520
    ]
    # draw
    draw(np.array(press_data), scope, "press", scenario_id, leak_id=leak_id)
    draw(np.array(demand_data), scope, "demand", scenario_id, leak_id=leak_id)
