import os
import pytest

DATA_DIR = "F:\\渗漏数据集\\Hanoi_CMH\\"
SCENARIO = "Scenario-"
PRESSURE_DIR = "Pressures"
DEMAND_DIR = "Demands"
LEAK_DIR = "Leaks"
FLOWS_DIR = "Flows"


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


class Link:
    def __init__(self, scenario_id, link_id):
        self.link_id = link_id
        self._flow = []
        self._initialize(scenario_id)

    def get_flow(self):
        return tuple(self._flow)

    def _initialize(self, scenario_id):
        # 读取流量文件
        flow_file = os.path.join(
            DATA_DIR, SCENARIO + scenario_id, FLOWS_DIR, f"Link_{str(self.link_id)}.csv"
        )
        with open(flow_file, "r") as f:
            f.readline()
            for line in f.readlines():
                self._flow.append(float(line.strip().split(",")[1]))
        pass


class Leak:
    def __init__(self, scenario_id, leak_id):
        self.leak_id = leak_id

        self._leak_demand = []
        self._leak_info = {}
        self._initialize(scenario_id)

    def _initialize(self, scenario_id):
        leaks_dir = os.path.join(DATA_DIR, SCENARIO + scenario_id, LEAK_DIR)
        # # 获取leak_id
        # file_list = os.listdir(leaks_dir)
        # if len(file_list):
        #     self.leak_id = set()
        # for file_name in file_list:
        #     leak_id = file_name.split("_")[1]  # 获取leak_id
        #     self.leak_id.add(leak_id)
        # self.leak_id = list(self.leak_id)  # 转换为list
        # self.leak_id = sorted(self.leak_id, key=int)  # 排序

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


class Scenario:
    def __init__(self, name):
        self.name = name
        self.labels = []
        self.scenario_info = {}
        self.nodes = []
        self.links = []
        self.leaks = []

    def get_nodes(self):
        pass

    def get_links(self):
        pass

    def get_pressures(self):
        pass

    def get_demands(self):
        pass

    def get_flows(self):
        pass

    def get_labels(self):
        pass

    def get_scenario_info(self):
        pass

    def get_leaks(self):
        pass


class HanoiCMH:
    def __init__(self, dataset_path):
        self.path = dataset_path
        self.labels = self.get_labels(label_name="Labels.csv")
        self.scenarios = self.get_scenarios()

    def get_labels(self, label_name=None):
        labels = []
        # 读取标签文件
        with open(os.path.join(self.path, label_name), "r") as f:
            # 跳过第一行
            f.readline()
            for line in f.readlines():  # line格式：'i,1 or 0/n'
                # 获取标签
                label = float(line.strip().split(",")[1])
                # 添加到labels中
                labels.append(label)
        return labels

    def get_scenarios(self):
        # 获取场景文件夹
        scenarios = []
        for scenario in os.listdir(self.path):
            if scenario.startswith("Scenario"):
                scenarios.append(scenario)

        # 对场景文件夹进行排序
        def sort_by_last_i(folder_name):
            last_i = folder_name.split("-")[-1]
            return int(last_i)

        sorted_folder_names = sorted(scenarios, key=sort_by_last_i)
        return sorted_folder_names


if __name__ == "__main__":
    leak = Leak(str(13), str(21))
