
import re
import numpy as np

def main(su2_mesh_filepath,root_leading_edge_ID,tip_leading_edge_ID,root_trailing_edge_ID,tip_trailing_edge_ID,wing_marker_tag):

    node_connectivity = initialise_node_connectivity(su2_mesh_filepath,wing_marker_tag)

    #Find leading/trailing edges
    leading_edge = a_star_search(root_leading_edge_ID,tip_leading_edge_ID,node_connectivity)
    print('Found leading edge')
    trailing_edge = a_star_search(root_trailing_edge_ID,tip_trailing_edge_ID,node_connectivity)
    print('Found trailing edge')

    #Generate nodal position data
    position_data = initialise_position_data(su2_mesh_filepath)



    return relative_chord, which_region

def initialise_node_connectivity(su2_mesh_filepath,wing_marker_tag):

    mesh_file = open(su2_mesh_filepath,'r')
    mesh_data = mesh_file.read()
    mesh_file.close()

    mesh_data = re.split('MARKER_TAG= '+wing_marker_tag,mesh_data)
    mesh_data = re.split('\n',mesh_data[1])
    number_of_elements = re.split('MARKER_ELEMS= ',mesh_data[1])
    number_of_elements = int(number_of_elements[1])

    #Parse "edge list", i.e. list of elements by node
    edge_list = np.zeros((number_of_elements,3))
    for a in range(2,number_of_elements+2):
        current_element = re.split('\t',mesh_data[a])
        for b in range(1,4):
            edge_list[a-2,b-1] = int(current_element[b])

    #Convert "edge list" to a connectivity list
    max_node_num = np.max(edge_list)
    node_connectivity = [[] for _ in range(int(max_node_num+1))]
    for a in range(number_of_elements):
        for b in range(3):
            for c in range(3):
                if c!=b:
                    if edge_list[a,c] not in node_connectivity[int(edge_list[a,b])]:
                        node_connectivity[int(edge_list[a,b])].append(int(edge_list[a,c]))

    return node_connectivity

def a_star_search(start_node,target_node,node_connectivity):

    node_score = [1E10]*len(node_connectivity)
    node_score[start_node] = 0

    def breadth_search(node_list,node_connectivity,node_score,blacklist,target_node):

        if node_list == []:
            return node_score

        whitelist = set()

        for which_node in node_list:
            for element in node_connectivity[which_node]:
                if element not in whitelist and element not in blacklist:
                    whitelist.add(element)
                if node_score[element] > node_score[which_node]+1:
                    node_score[element] = node_score[which_node]+1
            blacklist.add(which_node)

        new_node_list = []

        for element in whitelist:
            if element not in blacklist:
                new_node_list.append(element)

        node_score = breadth_search(new_node_list,node_connectivity,node_score,blacklist,target_node)

        return node_score

    node_score = breadth_search([start_node],node_connectivity,node_score,set(),target_node)

    current_score = node_score[target_node]

    def get_optimal_path(current_score,current_node,node_connectivity,node_score,optimal_path,exit_flag=False):

        for element in node_connectivity[current_node]:
            if not exit_flag:
                if node_score[element] == 0:
                    return [element], True
                elif node_score[element] == current_score-1:
                    optimal_path, exit_flag = get_optimal_path(current_score-1,element,node_connectivity,node_score,optimal_path,exit_flag)
                    if exit_flag:
                        optimal_path.append(element)

        return optimal_path, True

    optimal_path = get_optimal_path(current_score,target_node,node_connectivity,node_score,[target_node])[0]
    optimal_path.append(target_node)

    return optimal_path

def initialise_position_data(su2_mesh_filepath):

        mesh_file = open(su2_mesh_filepath,'r')
        mesh_data = mesh_file.read()
        mesh_file.close()

        mesh_data = re.split('NPOIN= ',mesh_data)
        mesh_data = re.split('\n',mesh_data[1])
        number_of_elements = int(mesh_data[0])

        #Parse "edge list", i.e. list of elements by node
        pos_list = np.zeros((number_of_elements,3))
        for a in range(1,number_of_elements+1):
            current_element = re.split('\t',mesh_data[a])
            for b in range(3):
                pos_list[a-1,b] = current_element[b]

        return pos_list
