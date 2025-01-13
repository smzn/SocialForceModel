from CVRP_Setup_3d_v1 import CVRP_Setup_3d
#from CVRP_Calculation_v1 import CVRP_Calculation
import time

# 初期設定
num_clients=300
num_shelters=20
num_vehicles=10
demand_options=(1, 2)
vehicle_capacity=4
area_size=20
min_distance=0.5
speed = 40 #車両の移動速度

# 時間計測開始
start_time = time.time()

setup = CVRP_Setup_3d(num_clients, num_shelters, num_vehicles, demand_options, vehicle_capacity, area_size, min_distance, speed)
nodes, vehicles = setup.generate_nodes_and_vehicles()
cost_matrix = setup.calculate_cost_matrix()
cost_matrix = setup.calculate_cost_matrix()
setup.plot_nodes()
setup.plot_nodes_3d()
setup.plot_contour_2d()
setup.plot_contour_3d()