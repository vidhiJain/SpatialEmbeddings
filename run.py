import os, sys
import math
import random
import matplotlib.pyplot as plt

import pandas as pd
import networkx as nx
from pathlib import Path
import logging, sys
from tqdm import tqdm

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
seed = 42
random.seed(seed)
from ai2thor.controller import Controller
from ai2thor.util import metrics

import utils
import embeddings



def vector_distance_2D(v0, v1):
    dx = v0['x'] - v1['x']
    dz = v0['z'] - v1['z']
    return math.sqrt(dx * dx + dz * dz)

def within_distance_check(query_pos, agent_pos, threshold=5):
    _distance = ((query_pos['x'] - agent_pos['x'])**2 + (query_pos['z'] - agent_pos['z'])**2)
    if _distance > threshold:
        return False
    return True


def plot_graph(g, shortest_path, agent_pos_node, query_pos_node):
    pos = {i:i for i in g.nodes}
    nx.draw_networkx(g, pos=pos, node_list=shortest_path, with_labels=False, node_size=8)
    plt.plot(agent_pos_node[0], agent_pos_node[1], marker='*', color='red')
    plt.plot(query_pos_node[0], query_pos_node[1], marker='*', color='yellow')
    plt.show()
    

def get_closest_reachable_point(object_position, reaching_points):
    """
    Finding closest reaching point for robot to the object
    @args:
        object_pos : dict with keys for x and z coordinates 
        reaching_points : 2D coordinates that are reachable by the robot

    @returns:
        target_pos : 2D tuple with x and z coordinates from reaching point 
            that is closest to the object_pos
    """
    distances = [vector_distance_2D(object_position, reachable_point) for reachable_point in reaching_points]
    min_dist = min(distances)
    index = [idx for idx in range(len(distances)) if distances[idx] == min_dist]
    idx = random.choice(index)
    target_pos_node = (reaching_points[idx]['x'], reaching_points[idx]['z'])
    return target_pos_node


def get_shortest_path(g, event, query_metadata, reaching_points):
    """
    Finding shortest path
    @args:
        g: graph, 
        event: Ai2THOR's event metadata (returned on controller's step/reset), 
        query_metadata: Metadata of the query object, 
        reaching_points: the points that can be traversed by the robot in current env
    @returns:
        shortest path, agent_pos_node, query_pos_node
    """
    assert (len(g.nodes) != 0)
    agent_metadata = event.metadata['agent']
    agent_pos_node = (agent_metadata['position']['x'], agent_metadata['position']['z'])
    query_pos = {'x': float(query_metadata['position']['x']), 'z':float(query_metadata['position']['z'])}

    query_pos_node = get_closest_reachable_point(query_pos, reaching_points)
    
    shortest_path = nx.astar_path(g, agent_pos_node, query_pos_node)
    shortest_path_length = len(shortest_path)
    return shortest_path_length, agent_pos_node, query_pos_node


def initialize_scene(controller, scene, seed=seed):
    """
    returns: controller after reset, graph of reachable positions
    """
    controller.reset(scene=scene)
    event = controller.step(action='GetReachablePositions')
    g = nx.Graph()
    reaching_points = event.metadata['actionReturn']
    for coord in reaching_points:
        for d in [-1, 1]:
            node = (coord['x'], coord['z'])
            if {'x': coord['x'] + d*GRIDSIZE, 'y': coord['y'], 'z': coord['z']} in reaching_points:
                g.add_edge(node, (coord['x'] + d*GRIDSIZE, coord['z']), weight=1)
            if {'x': coord['x'], 'y': coord['y'], 'z': coord['z'] + d*GRIDSIZE} in reaching_points:
                g.add_edge(node, (coord['x'], coord['z'] + d*GRIDSIZE), weight=1)
            if {'x': coord['x'] + d*GRIDSIZE, 'y': coord['y'], 'z': coord['z'] + d*GRIDSIZE} in reaching_points:
                g.add_edge(node, (coord['x'] + d*GRIDSIZE, coord['z'] + d*GRIDSIZE), weight=2)
            if {'x': coord['x'] - d*GRIDSIZE,'y': coord['y'], 'z': coord['z'] + d*GRIDSIZE} in reaching_points:
                g.add_edge(node, (coord['x'] - d*GRIDSIZE, coord['z'] + d*GRIDSIZE), weight=2)
    
    # Agent Initialize
    if seed is not None:
        random.seed(seed)
    idx = random.randint(0,len(reaching_points)-1)
    event = controller.step(action='Teleport',
        x=reaching_points[idx]['x'], 
        y=reaching_points[idx]['y'], 
        z=reaching_points[idx]['z']
    )
    return controller, g, reaching_points, event


def initialize_query_object(query, event, g, reaching_points):
    FOUND_OBJECT_FLAG = False
    query_metadata = None
    min_pathlength = 0
    # agent_metadata = event.metadata['agent']
    query_objects = []
    for obj in event.metadata['objects']:
        if obj['objectType'] == query:
            query_objects.append(obj)
            FOUND_OBJECT_FLAG = True  #### Termination 1: Query object found and returned
    #  Computing the shortest path to each instance 
    if FOUND_OBJECT_FLAG:
        pathlengths = [get_shortest_path(g, event, instance, reaching_points)[0] for instance in query_objects]
        min_pathlength = min(pathlengths)
        index = [idx for idx in range(len(pathlengths)) if pathlengths[idx] == min_pathlength][0]
        query_metadata = query_objects[index]
    # if not FOUND_OBJECT_FLAG: # Query object not present in scene
        # controller.stop()
    return  FOUND_OBJECT_FLAG, query_metadata, min_pathlength
    
    
def runExperiment(controller, graph, query, reaching_points, emb,
    STEP_MULTIPLE=1, VISIBLE_AREA_THRESHOLD=0.01):
    """
    @args:
        controller: controller for a respective scene/room
        event: event on initializing agent
        graph: graph constructed for a scene
        query: query object should be in query object
        reaching_points: dict of x,y,z coordinates reachable by robot

        EMB: Choose an embedding from embedding.py
        STEP_MULTIPLE: How many "Steps * 0.25" to move each time
        VISIBLE_AREA_THRESHOLD: Objects that are lower than this ratio, are discarded


    @returns:
        int path_length if object found. (Steps, *.25 to find area)
        -1: Query object not in scene
        -2: Finished naviation and object not found

    """
    print("Starting experiment with for query:", query)
    event = controller.step(action='Pass')
    
    ### Termination 1: Query object not found in scene
    FOUND_OBJECT_FLAG, query_metadata, shortest_path = initialize_query_object(query, event, graph, reaching_points)
    if not FOUND_OBJECT_FLAG:
        print('Object not found in scene', query)
        return -1
    
    # Find the groundtruth shortest path to the query
    # agent_pos_node = event.metadata['agent']
    __shortest_path, agent_pos_node, query_pos_node = get_shortest_path(graph, event, query_metadata, reaching_points)
    print("Shortest path:", __shortest_path, "; Agent pos:", agent_pos_node, "; Query pos:", query_pos_node)
    if (shortest_path != __shortest_path):
        breakpoint()

    # Start our navigation path
    path_length = 0
    object_found = False
    visited_objects = []

    # print ("Embeddings loaded Successfully")
    while not object_found:
        filtered_objects = []
        for _step in range(4):
            instances = utils.filter_objects_v2(event.instance_detections2D, VISIBLE_AREA_THRESHOLD)
            filtered_objects += instances
            event = controller.step(action='RotateRight', degrees=90)
        # Filter objects that are smaller than size VISIBLE_AREA_THRESHOLD
        filtered_objects_set = list(set(filtered_objects) - set(visited_objects))
        agent_metadata = event.metadata['agent']
        
        # If no more objects left to navigate
        if len(filtered_objects_set) == 0:
            print("No object to navigate to. filtered_objects_set is empty")
            # controller.stop()
            return -2
        if path_length >= 10*shortest_path:
            # print("Posible Oscilation")
            # controller.stop()
            return -3

        # Find most similar object 
        sim_scores = []  
        for _object in filtered_objects_set:
            # If reached query end
            objtype = _object.split('|')[0]
            if objtype == query: ### Query object found in field of view
                # print("Object found in filtered set")
                target_pos_node = get_closest_reachable_point(query_metadata['position'], reaching_points)  
                agent_pos_node = (agent_metadata['position']['x'], agent_metadata['position']['z'])
                path_node_list = nx.astar_path(graph, agent_pos_node, target_pos_node)
                path_length += len(path_node_list)
                object_found = True                 #### Termination 2: Query object found and returned
                # print("=>")
                # print("Current Path length:", path_length)
            # Else find embeddings:
            else:
                sim_scores.append(emb.get_similarity(query, objtype))
    
        # Sorting filtered list
        _val = [(x,y) for x, y in zip(filtered_objects_set, sim_scores)]
        _val.sort(key=lambda x:x[1], reverse=True)
        # print(_val)
    
        if object_found:
            break

        obj_to_navigate = _val[0][0]
        # print("In this iteration, navigating to:", obj_to_navigate)

        pos = [float(x) for x in obj_to_navigate.split('|')[1:4]]
        object_position = {'x': pos[0], 'y': pos[1], 'z': pos[2]}

        target_pos_node = get_closest_reachable_point(object_position, reaching_points)      
        agent_pos_node = (agent_metadata['position']['x'], agent_metadata['position']['z'])
        path_node_list = nx.astar_path(graph, agent_pos_node, target_pos_node)
        # print('path_node_list:', path_node_list)

        if STEP_MULTIPLE >= len(path_node_list)-1:
            event = controller.step(action='Teleport', x=target_pos_node[0], y=pos[1], z=target_pos_node[1])
            visited_objects.append(obj_to_navigate)
            path_length += (STEP_MULTIPLE-1)
        else:
            _step  = int(len(path_node_list)/STEP_MULTIPLE)
            event = controller.step(action='Teleport', x=path_node_list[_step][0], y=pos[1], z=path_node_list[_step][1])
            path_length += _step
        # ### DEBUG
        # print ("Agent location:", agent_pos_node)
        # print("Current Path length:", path_length)
        # print("Visited:", visited_objects)
        # print("Filtered:", _val)
        if within_distance_check(query_metadata['position'], agent_metadata['position'], threshold=5):
            print("Object Found at distance ")
            target_pos_node = get_closest_reachable_point(query_metadata['position'], reaching_points)  
            agent_pos_node = (agent_metadata['position']['x'], agent_metadata['position']['z'])
            path_node_list = nx.astar_path(graph, agent_pos_node, target_pos_node)
            path_length += len(path_node_list)
            object_found = True
        if query_pos_node in visited_objects:
            # print("Object Found")
            object_found = True
    # controller.stop() 
    return [shortest_path, path_length]


def main(controller, floors, name):
    total_runs = len(bedroom_floors)*len(query_list)
    oscillation_count = 0
    object_not_present_inscene = 0
    no_object_to_navigate = 0
    embeddings_list = ["GraphEmbedding", "RobocseEmbedding", "FastTextEmbedding", "Word2VecEmbedding"]

    records = []
    for emb_name in embeddings_list:
        _embedding_to_call = getattr(embeddings, emb_name)
        emb = _embedding_to_call()
        for _room in floors:
            
            _controller, _graph, _reaching_points, _event = initialize_scene(controller, _room)  

            for i, _query in enumerate(tqdm(query_list)):
                
                assert (_graph.nodes() is not None)
                _val = runExperiment(_controller, _graph, _query[0], _reaching_points, emb, STEP_MULTIPLE = 5, VISIBLE_AREA_THRESHOLD=0.01)

                print('Embedding: ', emb_name, 'Scene: ',  _room, 'Query: ', _query[0], 'seed: ', seed, '_val: ', str(_val))

                records.append([emb_name, _room, _query[0], seed, str(_val)])
                pd.DataFrame(records, columns = ["Embedding", "Scene", "Query", "Seed", "_val"]).to_csv('results_'+name+'_queries.csv', index = False)                

    breakpoint()
    print("Done")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test the object embeddings for search and navigation')
    parser.add_argument('--floors', default='livingroom', help='room type to evaluate on')
    parser.add_argument('--gridsize', default=0.25, help='gridsize of the rooms for robot to navigate on')
    parser.add_argument('--step_multiple', default=3, help='step size to move with towards sub-goal object')
    parser.add_argument('--visible_area_threshold', default=0.01, help='visible area threshold to detect objects')
    args = parser.parse_args()
    print(args)

    # Parameters
    GRIDSIZE = args.gridsize
    STEP_MULTIPLE = args.step_multiple
    VISIBLE_AREA_THRESHOLD = args.visible_area_threshold

    # Generate list to navigate over
    kitchens_floors = [f'FloorPlan{num}' for num in range(1,31)]
    livingroom_floors = [f'FloorPlan{num}' for num in range(201,231)]
    bedroom_floors = [f'FloorPlan{num}' for num in range(301,331)]
    bathroom_floors = [f'FloorPlan{num}' for num in range(401,431)]
    query_list =  pd.read_csv("data/query.csv", header = None).values # Object list to sample from 

    # floors = [kitchens_floors, livingroom_floors, bedroom_floors, bathroom_floors]
    if args.floors == 'kitchen':
        floors = kitchen_floors
    elif args.floors == 'livingroom':
        floors = livingroom_floors
    elif args.floors == 'bedroom':
        floors = bedroom_floors
    elif args.floors == 'bathroom':
        floors = bathroom_floors


    controller = Controller(scene="FloorPlan1", gridSize=GRIDSIZE, fieldOfView=120, 
        renderObjectImage=True, renderClassImage=True)

    main(controller, floors, name=args.floors)