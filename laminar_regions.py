
import re
import numpy as np

def main(su2_mesh_filepath,root_leading_edge_ID,tip_leading_edge_ID,root_lower_trailing_edge_ID,tip_lower_trailing_edge_ID,root_upper_trailing_edge_ID,tip_upper_trailing_edge_ID,wing_marker_tag):

    node_connectivity, max_node_num = initialise_node_connectivity(su2_mesh_filepath,wing_marker_tag)

    #Generate nodal position data
    position_data = initialise_position_data(su2_mesh_filepath,max_node_num)
    print('Got position data')

    mean_trailing_edge_root = np.mean(position_data[(root_lower_trailing_edge_ID,root_upper_trailing_edge_ID),:],axis=0)
    mean_trailing_edge_tip = np.mean(position_data[(tip_lower_trailing_edge_ID,tip_upper_trailing_edge_ID),:],axis=0)

    upper_lower_direction, upper_lower_boundary = rect_norm([mean_trailing_edge_root,mean_trailing_edge_tip,position_data[tip_leading_edge_ID,:],position_data[root_leading_edge_ID,:]])

    lower_surface_direction = np.dot(upper_lower_direction,position_data[root_lower_trailing_edge_ID]) - upper_lower_boundary
    lower_surface_direction = lower_surface_direction / abs(lower_surface_direction)
    print('Separated upper and lower surfaces')

    #Find leading/trailing edges
    leading_edge = a_star_search(root_leading_edge_ID,tip_leading_edge_ID,node_connectivity,position_data)
    print('Found leading edge')
    trailing_edge = a_star_search(root_lower_trailing_edge_ID,tip_lower_trailing_edge_ID,node_connectivity,position_data)
    print('Found trailing edge')

    print('leading edge length: '+str(len(leading_edge)))
    print('trailing edge length: '+str(len(trailing_edge)))

    np.savetxt('leading_edge.csv', position_data[leading_edge,:], delimiter=',')
    np.savetxt('trailing_edge.csv', position_data[trailing_edge,:], delimiter=',')

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
        current_element = re.split(' ',mesh_data[a])
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

    return node_connectivity, max_node_num

def a_star_search(start_node,target_node,node_connectivity,position_data):

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

    def get_optimal_path(current_score,current_node,node_connectivity,node_score,optimal_path,position_data,target_node,exit_flag=False):

        whitelist = []

        for element in node_connectivity[current_node]:
            if not exit_flag:
                if node_score[element] == 0:
                    return [element], True
                elif node_score[element] == current_score-1:
                    whitelist.append(element)

        if len(whitelist) == 1:

            optimal_path, exit_flag = get_optimal_path(current_score-1,whitelist[0],node_connectivity,node_score,optimal_path,position_data,target_node,exit_flag)
            if exit_flag:
                optimal_path.append(whitelist[0])

        elif len(whitelist) > 1:

            distances = []
            for element in whitelist:
                dist = np.linalg.norm(position_data[target_node]-position_data[current_node])
                distances.append(dist)
            min_whitelist = np.argmin(distances)

            optimal_path, exit_flag = get_optimal_path(current_score-1,whitelist[min_whitelist],node_connectivity,node_score,optimal_path,position_data,target_node,exit_flag)
            if exit_flag:
                optimal_path.append(whitelist[min_whitelist])

        return optimal_path, True

    optimal_path = get_optimal_path(current_score,target_node,node_connectivity,node_score,[target_node],position_data,start_node)[0]
    optimal_path.append(target_node)

    return optimal_path

def initialise_position_data(su2_mesh_filepath,max_node_num=1E10):

        mesh_file = open(su2_mesh_filepath,'r')
        mesh_data = mesh_file.read()
        mesh_file.close()

        mesh_data = re.split('NPOIN= ',mesh_data)
        mesh_data = re.split('\n',mesh_data[1])
        number_of_elements = int(mesh_data[0])
        if max_node_num < number_of_elements:
            number_of_elements = int(max_node_num)

        #Parse "edge list", i.e. list of elements by node
        pos_list = np.zeros((number_of_elements,3))
        for a in range(1,number_of_elements+1):
            current_element = re.split(' ',mesh_data[a])
            for b in range(3):
                pos_list[a-1,b] = current_element[b]

        return pos_list

def rect_norm(input_points):

    norm_list = np.zeros((4,3))

    for a in range(4):
        current_norm = np.zeros((3,3))
        counter = 0
        clockwiseness = a
        for b in range(4):
            if a != b:
                clockwiseness += 1
                if clockwiseness == 4:
                    clockwiseness = 0
                current_norm[counter,:] = input_points[clockwiseness]
                counter += 1

        norm_list[a,:] = np.cross((current_norm[1]-current_norm[0]),(current_norm[2]-current_norm[0]))
        norm_list[a,:] = norm_list[a,:] / np.linalg.norm(norm_list[a,:])

    mean_norm = np.mean(norm_list,axis=0)
    plane_origin = np.dot(mean_norm,np.mean(input_points,axis=0))

    return mean_norm, plane_origin
