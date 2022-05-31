
import re
import numpy as np
from scipy.interpolate import CubicSpline

def main(su2_mesh_filepath,root_leading_edge_ID,tip_leading_edge_ID,root_lower_trailing_edge_ID,tip_lower_trailing_edge_ID,root_upper_trailing_edge_ID,tip_upper_trailing_edge_ID,wing_marker_tag,span_direction,chord_direction,max_nlf,efficacy,banned_span):

    node_connectivity, max_node_num = initialise_node_connectivity(su2_mesh_filepath,wing_marker_tag)

    #Generate nodal position data
    position_data = initialise_position_data(su2_mesh_filepath,max_node_num)
    print('got position data')

    mean_trailing_edge_root = np.mean(position_data[(root_lower_trailing_edge_ID,root_upper_trailing_edge_ID),:],axis=0)
    mean_trailing_edge_tip = np.mean(position_data[(tip_lower_trailing_edge_ID,tip_upper_trailing_edge_ID),:],axis=0)

    upper_lower_direction, upper_lower_boundary = rect_norm([mean_trailing_edge_root,mean_trailing_edge_tip,position_data[tip_leading_edge_ID,:],position_data[root_leading_edge_ID,:]])

    lower_surface_direction = np.dot(upper_lower_direction,position_data[root_lower_trailing_edge_ID]) - upper_lower_boundary
    lower_surface_direction = lower_surface_direction / abs(lower_surface_direction)
    print('separated upper and lower surfaces')

    #Find leading/trailing edges
    leading_edge = a_star_search(root_leading_edge_ID,tip_leading_edge_ID,node_connectivity,position_data,upper_lower_direction,-lower_surface_direction,upper_lower_boundary,set())
    print('found leading edge')
    trailing_edge_lower = a_star_search(root_lower_trailing_edge_ID,tip_lower_trailing_edge_ID,node_connectivity,position_data,upper_lower_direction,lower_surface_direction,upper_lower_boundary,set())
    print('found lower trailing edge')
    trailing_edge_upper = a_star_search(root_upper_trailing_edge_ID,tip_upper_trailing_edge_ID,node_connectivity,position_data,upper_lower_direction,-lower_surface_direction,upper_lower_boundary,set(trailing_edge_lower))
    print('found upper trailing edge')

    print('leading edge length: '+str(len(leading_edge)))
    print('lower trailing edge length: '+str(len(trailing_edge_lower)))
    print('upper trailing edge length: '+str(len(trailing_edge_upper)))

    np.savetxt('leading_edge.csv', position_data[leading_edge,:], delimiter=',')
    np.savetxt('trailing_edge_lower.csv', position_data[trailing_edge_lower,:], delimiter=',')
    np.savetxt('trailing_edge_upper.csv', position_data[trailing_edge_upper,:], delimiter=',')

    leading_edge_span_length = path_spatial_length(leading_edge,position_data,span_direction)
    trailing_edge_lower_span_length = path_spatial_length(trailing_edge_lower,position_data,span_direction)
    trailing_edge_upper_span_length = path_spatial_length(trailing_edge_upper,position_data,span_direction)

    leading_span = leading_edge_span_length[-1]-leading_edge_span_length[0]
    trailing_lower_span = trailing_edge_lower_span_length[-1]-trailing_edge_lower_span_length[0]
    trailing_upper_span = trailing_edge_upper_span_length[-1]-trailing_edge_upper_span_length[0]

    print('leading edge span length: '+str(leading_span))
    print('trailing edge lower span length: '+str(trailing_lower_span))
    print('trailing edge upper span length: '+str(trailing_upper_span))

    leading_edge_chord_length = path_spatial_length(leading_edge,position_data,chord_direction)
    trailing_edge_lower_chord_length = path_spatial_length(trailing_edge_lower,position_data,chord_direction)
    trailing_edge_upper_chord_length = path_spatial_length(trailing_edge_upper,position_data,chord_direction)

    print('leading edge chord length: '+str(leading_edge_chord_length[-1]-leading_edge_chord_length[0]))
    print('trailing edge lower chord length: '+str(trailing_edge_lower_chord_length[-1]-trailing_edge_lower_chord_length[0]))
    print('trailing edge upper chord length: '+str(trailing_edge_upper_chord_length[-1]-trailing_edge_upper_chord_length[0]))

    leading_edge_spline = CubicSpline(leading_edge_span_length, leading_edge_chord_length, axis=0)
    trailing_edge_lower_spline = CubicSpline(trailing_edge_lower_span_length, trailing_edge_lower_chord_length, axis=0)
    trailing_edge_upper_spline = CubicSpline(trailing_edge_upper_span_length, trailing_edge_upper_chord_length, axis=0)

    print('converted leading/trailing edges to splines')

    # leading_edge_percent_chord = leading_edge_span_length/leading_edge_span_length[-1]
    # trailing_edge_lower_percent_chord = trailing_edge_lower_span_length/trailing_edge_lower_span_length[-1]
    # trailing_edge_upper_percent_chord = trailing_edge_upper_span_length/trailing_edge_upper_span_length[-1]

    lower_surface = process_laminarity(leading_edge,trailing_edge_lower,node_connectivity,position_data,upper_lower_direction,lower_surface_direction,upper_lower_boundary,span_direction,chord_direction,leading_span,max_nlf,banned_span,efficacy,max_node_num,leading_edge_span_length[0],leading_edge_spline,trailing_edge_lower_spline)
    upper_surface = process_laminarity(leading_edge,trailing_edge_upper,node_connectivity,position_data,upper_lower_direction,-lower_surface_direction,upper_lower_boundary,span_direction,chord_direction,leading_span,max_nlf,banned_span,efficacy,max_node_num,leading_edge_span_length[0],leading_edge_spline,trailing_edge_upper_spline)

    print('generated all laminarity metadata')

    np.savetxt('lower_surface.csv',lower_surface,delimiter=',')
    np.savetxt('upper_surface.csv',upper_surface,delimiter=',')

    print('wrote data to csv files')

    return

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

def a_star_search(start_node,target_node,node_connectivity,position_data,upper_lower_direction,correct_direction,upper_lower_boundary,blacklist):

    node_score = [1E10]*len(node_connectivity)
    node_score[start_node] = 0

    def breadth_search(node_list,node_connectivity,node_score,blacklist,target_node):

        if node_list == []:
            return node_score

        whitelist = set()

        for which_node in node_list:
            if node_score[which_node] < node_score[target_node]:
                for element in node_connectivity[which_node]:
                    which_direction = (np.dot(upper_lower_direction,position_data[element,:])-upper_lower_boundary)*correct_direction
                    if element not in whitelist and element not in blacklist and which_direction > -0.01:
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

    node_score = breadth_search([start_node],node_connectivity,node_score,blacklist,target_node)

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
            number_of_elements = int(max_node_num)+1

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

def path_spatial_length(path,position_data,direction):

    path_length = np.zeros((len(path)))
    dir = np.asarray(direction,dtype=float)

    for a in range(len(path)):
        path_length[a] = np.dot(position_data[path[a],:],dir)

    return path_length

def process_laminarity(leading_edge,trailing_edge,node_connectivity,position_data,upper_lower_direction,lower_surface_direction,upper_lower_boundary,span_direction,chord_direction,leading_span,max_nlf,banned_span,efficacy,max_node_num,root_span,leading_edge_spline,trailing_edge_spline):
    laminar_data = [[] for _ in range(len(leading_edge))]

    output_data = np.zeros((int(max_node_num),8))
    counter = 0

    for a in range(np.min([len(leading_edge),len(trailing_edge)])):

        laminar_data[a] = a_star_search(leading_edge[a],trailing_edge[a],node_connectivity,position_data,upper_lower_direction,-lower_surface_direction,upper_lower_boundary,set())

        for b in range(len(laminar_data[a])):

            current_pos = position_data[laminar_data[a][b]]
            current_span = np.dot(current_pos,span_direction)
            current_chord = np.dot(current_pos,chord_direction)

            leading_chord = leading_edge_spline(current_span)
            trailing_chord = trailing_edge_spline(current_span)

            percent_span = (current_span - root_span)/leading_span
            percent_chord = (current_chord - leading_chord) / (trailing_chord - leading_chord)
            current_efficacy = percent_chord / max_nlf

            current_banned_span = False
            for c in range(len(banned_span[:,0])):
                if percent_span >= banned_span[c,0] and percent_span <= banned_span[c,1]:
                    current_banned_span = True
                    break

            if current_efficacy < efficacy and not current_banned_span:
                laminar_or_not = 1
            else:
                laminar_or_not = 0

            output_data[counter,0] = laminar_data[a][b]
            output_data[counter,1] = percent_span
            output_data[counter,2] = percent_chord
            output_data[counter,3] = current_efficacy
            output_data[counter,4] = laminar_or_not
            output_data[counter,5:8] = position_data[laminar_data[a][b]]
            counter += 1

    output_data = output_data[~np.all(output_data == 0, axis=1)]

    return output_data

banned_span = np.zeros((4,2))
banned_span[0,:] = [0.,0.03]
banned_span[1,:] = [0.2,0.25]
banned_span[2,:] = [0.455,0.48]
banned_span[3,:] = [0.96,1]

main('C:/[redacted]/Laminar_Regions/mrsbw-V-BASE-newBL.su2',1,136218,4,136222,22500,158565,'wing',[0,1,0],[1,0,0],0.6,0.5,banned_span)
