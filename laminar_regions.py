
import re
import numpy as np
from scipy.interpolate import CubicSpline
from paraview.simple import *
import sys

def main(su2_mesh_filepath,root_leading_edge_ID,tip_leading_edge_ID,root_lower_trailing_edge_ID,tip_lower_trailing_edge_ID,root_upper_trailing_edge_ID,tip_upper_trailing_edge_ID,wing_marker_tag,span_direction,chord_direction,max_nlf_lower,max_nlf_upper,efficiency_lower,efficiency_upper,banned_span,get_paths_from_csv,get_pos_from_csv,su2_filename,root_filepath,laminar_lower_filename,laminar_upper_filename,turbulent_lower_filename,turbulent_upper_filename,razor_out_filename,output_vtk_instead_of_csv):

    node_connectivity, max_node_num, edge_list, number_of_elements = initialise_node_connectivity(root_filepath+su2_mesh_filepath,wing_marker_tag)

    su2_data = np.genfromtxt(root_filepath+su2_filename, delimiter=',')
    if all(np.isnan(su2_data[0,:])):
        su2_data = np.delete(su2_data,0,axis=0)

    if get_pos_from_csv:
        position_data = su2_data[:,1:4]
        #position_data = np.genfromtxt('position_data.csv', delimiter=',')
    else:
        #Generate nodal position data
        position_data = initialise_position_data(su2_mesh_filepath,max_node_num)
        np.savetxt(root_filepath+'position_data.csv', position_data, delimiter=',')

    print('got position data')

    if get_paths_from_csv:
        lower_pathlist = read_csv_as_list(root_filepath+'lower_nodes.csv')
        upper_pathlist = read_csv_as_list(root_filepath+'upper_nodes.csv')

        leading_edge = np.genfromtxt(root_filepath+'leading_edge.csv', delimiter=',', dtype=int)
        trailing_edge_lower = np.genfromtxt(root_filepath+'trailing_edge_lower.csv', delimiter=',', dtype=int)
        trailing_edge_upper = np.genfromtxt(root_filepath+'trailing_edge_upper.csv', delimiter=',', dtype=int)

        print('loaded paths from file')
    else:
        mean_trailing_edge_root = np.mean(position_data[(root_lower_trailing_edge_ID,root_upper_trailing_edge_ID),:],axis=0)
        mean_trailing_edge_tip = np.mean(position_data[(tip_lower_trailing_edge_ID,tip_upper_trailing_edge_ID),:],axis=0)

        upper_lower_direction, upper_lower_boundary = rect_norm([mean_trailing_edge_root,mean_trailing_edge_tip,position_data[tip_leading_edge_ID,:],position_data[root_leading_edge_ID,:]])

        lower_surface_direction = np.dot(upper_lower_direction,position_data[root_lower_trailing_edge_ID]) - upper_lower_boundary
        lower_surface_direction = lower_surface_direction / abs(lower_surface_direction)
        print('separated upper and lower surfaces')

        #Find leading/trailing edges
        leading_edge = a_star_search(root_leading_edge_ID,tip_leading_edge_ID,node_connectivity,position_data,upper_lower_direction,None,upper_lower_boundary,set())
        print('found leading edge')
        trailing_edge_lower = a_star_search(root_lower_trailing_edge_ID,tip_lower_trailing_edge_ID,node_connectivity,position_data,upper_lower_direction,None,upper_lower_boundary,set())
        print('found lower trailing edge')
        trailing_edge_upper = a_star_search(root_upper_trailing_edge_ID,tip_upper_trailing_edge_ID,node_connectivity,position_data,upper_lower_direction,None,upper_lower_boundary,set(trailing_edge_lower))
        print('found upper trailing edge')

        print('leading edge length: '+str(len(leading_edge)))
        print('lower trailing edge length: '+str(len(trailing_edge_lower)))
        print('upper trailing edge length: '+str(len(trailing_edge_upper)))

        np.savetxt(root_filepath+'leading_edge.csv', leading_edge, delimiter=',')
        np.savetxt(root_filepath+'trailing_edge_lower.csv', trailing_edge_lower, delimiter=',')
        np.savetxt(root_filepath+'trailing_edge_upper.csv', trailing_edge_upper, delimiter=',')

        lower_pathlist = get_surface_path_list(leading_edge,trailing_edge_lower,node_connectivity,position_data,upper_lower_direction,lower_surface_direction,upper_lower_boundary)
        upper_pathlist = get_surface_path_list(leading_edge,trailing_edge_upper,node_connectivity,position_data,upper_lower_direction,-lower_surface_direction,upper_lower_boundary)

        print('got lower/upper surface full node list')

        save_list_as_csv(lower_pathlist,root_filepath+'lower_nodes.csv')
        save_list_as_csv(upper_pathlist,root_filepath+'upper_nodes.csv')

    # flattened_list = []
    # for row in lower_pathlist:
    #     for element in row:
    #         flattened_list.append(element)
    # np.savetxt('lower_test.csv',su2_data[flattened_list,:],delimiter=',')
    # flattened_list = []
    # for row in upper_pathlist:
    #     for element in row:
    #         flattened_list.append(element)
    # np.savetxt('upper_test.csv',su2_data[flattened_list,:],delimiter=',')
    # raise ValueError

    lower_set = set()
    upper_set = set()

    for row in lower_pathlist:
        for element in row:
            lower_set.add(int(element))

    for row in upper_pathlist:
        for element in row:
            upper_set.add(int(element))

    leftovers = np.zeros((max_node_num+1),dtype=int)
    for a in range(max_node_num+1):
        if a in lower_set or a in upper_set:
            leftovers[a] = 1

    element_tree = generate_element_tree(edge_list,position_data,lower_set,upper_set)

    print('generated element tree')

    # print('Lower pathlist lengths')
    # for a in range(len(lower_pathlist)):
    #     print(len(lower_pathlist[a]))
    #
    # print('Upper pathlist lengths')
    # for a in range(len(upper_pathlist)):
    #     print(len(upper_pathlist[a]))

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

    print(leading_edge_span_length)
    print(leading_edge_chord_length)
    raise ValueError

    leading_edge_spline = CubicSpline(leading_edge_span_length, leading_edge_chord_length, axis=0)
    trailing_edge_lower_spline = CubicSpline(trailing_edge_lower_span_length, trailing_edge_lower_chord_length, axis=0)
    trailing_edge_upper_spline = CubicSpline(trailing_edge_upper_span_length, trailing_edge_upper_chord_length, axis=0)

    print('converted leading/trailing edges to splines')

    # leading_edge_percent_chord = leading_edge_span_length/leading_edge_span_length[-1]
    # trailing_edge_lower_percent_chord = trailing_edge_lower_span_length/trailing_edge_lower_span_length[-1]
    # trailing_edge_upper_percent_chord = trailing_edge_upper_span_length/trailing_edge_upper_span_length[-1]

    #correction_spline_list = read_chordwise_corrections('C:/[redacted]/Laminar_Regions/laminar_distribution.csv')

    lower_laminar_spline_cf, upper_laminar_spline_cf, lower_turbulent_spline_cf, upper_turbulent_spline_cf, lower_laminar_spline_cp, upper_laminar_spline_cp, lower_turbulent_spline_cp, upper_turbulent_spline_cp = load_CF_corrections(max_nlf_lower,max_nlf_upper,root_filepath,laminar_lower_filename,laminar_upper_filename,turbulent_lower_filename,turbulent_upper_filename)

    print('created correction splines')

    lower_surface = process_laminarity(leading_edge,trailing_edge_lower,su2_data,span_direction,chord_direction,leading_span,max_nlf_lower,banned_span,efficiency_lower,max_node_num,leading_edge_span_length[0],leading_edge_spline,trailing_edge_lower_spline,lower_pathlist,lower_laminar_spline_cf,lower_turbulent_spline_cf,lower_laminar_spline_cp,lower_turbulent_spline_cp,element_tree,edge_list,-1)
    upper_surface = process_laminarity(leading_edge,trailing_edge_upper,su2_data,span_direction,chord_direction,leading_span,max_nlf_upper,banned_span,efficiency_upper,max_node_num,leading_edge_span_length[0],leading_edge_spline,trailing_edge_upper_spline,upper_pathlist,upper_laminar_spline_cf,upper_turbulent_spline_cf,upper_laminar_spline_cp,upper_turbulent_spline_cp,element_tree,edge_list,1)
    leftover_surface = process_leftovers(leftovers,su2_data,max_node_num)

    print('generated all laminarity metadata')

    if not output_vtk_instead_of_csv:
        output_file = np.concatenate((lower_surface,upper_surface,leftover_surface),axis=0)
        np.savetxt(root_filepath+razor_out_filename,output_file,delimiter=',')
        # np.savetxt('lower_surface.csv',lower_surface,delimiter=',')
        # np.savetxt('upper_surface.csv',upper_surface,delimiter=',')
        # np.savetxt('leftover_surface.csv',leftover_surface,delimiter=',')

        print('wrote data to csv files')

    else:

        write_vtk_file(root_filepath+razor_out_filename,position_data,max_node_num,edge_list,number_of_elements,[lower_surface,upper_surface,leftover_surface])

        print('wrote data to vtk files')

    # calculate_coefficients()

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
    max_node_num = int(np.max(edge_list))
    node_connectivity = [[] for _ in range(max_node_num+1)]
    for a in range(number_of_elements):
        for b in range(3):
            for c in range(3):
                if c!=b:
                    if edge_list[a,c] not in node_connectivity[int(edge_list[a,b])]:
                        node_connectivity[int(edge_list[a,b])].append(int(edge_list[a,c]))

    return node_connectivity, max_node_num, edge_list, number_of_elements

def a_star_search(start_node,target_node,node_connectivity,position_data,upper_lower_direction,correct_direction,upper_lower_boundary,blacklist):

    node_score = [1.E50]*len(node_connectivity)
    node_score[start_node] = 0

    def breadth_search(node_list,node_connectivity,node_score,blacklist,target_node):

        if node_list == []:
            return node_score

        whitelist = set()

        for which_node in node_list:
            if node_score[which_node] < node_score[target_node]:
                for element in node_connectivity[which_node]:

                    if element not in whitelist and element not in blacklist:# and which_direction > -0.01:
                        whitelist.add(element)

                    if correct_direction is not None:
                        which_direction = (np.dot(upper_lower_direction,position_data[element,:])-upper_lower_boundary)*correct_direction
                    else:
                        which_direction = 1.

                    node_difference = position_data[element,:]-position_data[which_node,:]
                    if which_direction < 0.:
                        node_difference *= 3.

                    node_distance = np.linalg.norm(node_difference)

                    if node_score[element] > node_score[which_node] + node_distance:
                        node_score[element] = node_score[which_node] + node_distance

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

        score_list = []

        for element in node_connectivity[current_node]:
            score_list.append(node_score[element])

        best_score = np.amin(score_list)

        if best_score == 0.0:
            return [element], True

        if best_score >= current_score:
            raise ValueError

        whitelist = []
        best_node = node_connectivity[current_node][np.argmin(score_list)]
        whitelist.append(best_node)

        # for element in node_connectivity[current_node]:
        #     if not exit_flag:
        #         if node_score[element] == 0:
        #             return [element], True
        #         elif node_score[element] == current_score-1:
        #             whitelist.append(element)

        if len(whitelist) == 1:

            optimal_path, exit_flag = get_optimal_path(best_score,whitelist[0],node_connectivity,node_score,optimal_path,position_data,target_node,exit_flag)
            if exit_flag:
                optimal_path.append(whitelist[0])

        elif len(whitelist) > 1:

            distances = []
            for element in whitelist:
                dist = np.linalg.norm(position_data[target_node]-position_data[current_node])
                distances.append(dist)
            min_whitelist = np.argmin(distances)

            optimal_path, exit_flag = get_optimal_path(best_score,whitelist[min_whitelist],node_connectivity,node_score,optimal_path,position_data,target_node,exit_flag)
            if exit_flag:
                optimal_path.append(whitelist[min_whitelist])

        return optimal_path, True

    optimal_path = get_optimal_path(current_score,target_node,node_connectivity,node_score,[target_node],position_data,start_node)[0]
    optimal_path.append(target_node)

    if optimal_path[0] == optimal_path[1]:
        raise ValueError

    return optimal_path

def initialise_position_data(su2_mesh_filepath,max_node_num=1E10):

        mesh_file = open(su2_mesh_filepath,'r')
        mesh_data = mesh_file.read()
        mesh_file.close()

        mesh_data = re.split('NPOIN= ',mesh_data)
        mesh_data = re.split('\n',mesh_data[1])
        number_of_elements = int(mesh_data[0])
        if max_node_num < number_of_elements:
            number_of_elements = max_node_num+1

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

def process_laminarity(leading_edge,trailing_edge,su2_data,span_direction,chord_direction,leading_span,max_nlf,banned_span,efficiency,max_node_num,root_span,leading_edge_spline,trailing_edge_spline,node_list,laminar_spline_cf,turbulent_spline_cf,laminar_spline_cp,turbulent_spline_cp,element_tree,edge_list,lower_or_upper):

    output_data = np.zeros((max_node_num,12))
    counter = 0

    for a in range(len(node_list)):
        for b in range(len(node_list[a])):

            current_pos = su2_data[int(node_list[a][b]),1:4]
            current_span = np.dot(current_pos,span_direction)
            current_chord = np.dot(current_pos,chord_direction)

            leading_chord = leading_edge_spline(current_span)
            trailing_chord = trailing_edge_spline(current_span)

            percent_span = (current_span - root_span)/leading_span
            percent_chord = (current_chord - leading_chord) / (trailing_chord - leading_chord)
            current_efficiency = percent_chord / max_nlf

            current_pressure = su2_data[int(node_list[a][b]),13]
            current_friction = su2_data[int(node_list[a][b]),15:18]

            current_banned_span = False
            for c in range(len(banned_span[:,0])):
                if percent_span >= banned_span[c,0] and percent_span <= banned_span[c,1]:
                    current_banned_span = True
                    break

            percent_transition = max_nlf*efficiency

            if percent_chord < percent_transition+0.1 and efficiency > 0.1:

                if percent_chord < percent_transition:

                    corrected_friction = current_friction * laminar_spline_cf(percent_chord) / turbulent_spline_cf(percent_chord)
                    corrected_pressure = current_pressure * laminar_spline_cp(percent_chord) / turbulent_spline_cp(percent_chord)

                else:

                    transition_chord = percent_transition*(trailing_chord-leading_chord) + leading_chord
                    laminar_transition_cf = interpolate_field_values(su2_data,[15,16,17],element_tree,edge_list,[transition_chord,current_span],True,lower_or_upper) * laminar_spline_cf(percent_transition) / turbulent_spline_cf(percent_transition)
                    turbulent_transition_cf = interpolate_field_values(su2_data,[15,16,17],element_tree,edge_list,[transition_chord+0.1,current_span],True,lower_or_upper)
                    corrected_friction = turbulent_transition_cf*shifted_sigmoid(percent_chord-percent_transition,0.1) +  laminar_transition_cf*(1-shifted_sigmoid(percent_chord-percent_transition,0.1))
                    corrected_friction[np.abs(corrected_friction)>np.abs(current_friction)] = current_friction[np.abs(corrected_friction)>np.abs(current_friction)]

                    laminar_transition_cp = interpolate_field_values(su2_data,[13],element_tree,edge_list,[transition_chord,current_span],True,lower_or_upper) * laminar_spline_cp(percent_transition) / turbulent_spline_cp(percent_transition)
                    turbulent_transition_cp = interpolate_field_values(su2_data,[13],element_tree,edge_list,[transition_chord+0.1,current_span],True,lower_or_upper)
                    corrected_pressure = turbulent_transition_cp*shifted_sigmoid(percent_chord-percent_transition,0.1) +  laminar_transition_cp*(1-shifted_sigmoid(percent_chord-percent_transition,0.1))

                    if np.abs(corrected_pressure) > np.abs(current_pressure):
                        corrected_pressure = current_pressure

            else:
                corrected_friction = current_friction
                corrected_pressure = current_pressure

            corrected_pressure = current_pressure*mirrored_shifted_sigmoid(percent_span,0.02,banned_span) + (1-mirrored_shifted_sigmoid(percent_span,0.02,banned_span))*corrected_pressure
            corrected_friction = current_friction*mirrored_shifted_sigmoid(percent_span,0.02,banned_span) + (1-mirrored_shifted_sigmoid(percent_span,0.02,banned_span))*corrected_friction

            # if corrected_friction[0] < 0 and percent_chord > 0.1 and percent_chord < 0.6:
            #     print(str(percent_chord)+' : '+str(laminar_spline_cf(percent_transition) / turbulent_spline_cf(percent_transition))+' : '+str(corrected_friction[0])+' : '+str(current_friction[0]))

            if current_efficiency < efficiency and not current_banned_span:
                laminar_or_not = 1
            else:
                laminar_or_not = 0

            output_data[counter,0] = node_list[a][b]
            output_data[counter,1:4] = current_pos
            output_data[counter,4] = percent_span
            output_data[counter,5] = percent_chord
            output_data[counter,6] = current_efficiency
            output_data[counter,7] = laminar_or_not
            output_data[counter,8] = corrected_pressure
            output_data[counter,9:12] = corrected_friction
            counter += 1

    output_data = output_data[~np.all(output_data == 0, axis=1)]

    return output_data

def process_leftovers(leftovers,su2_data,max_node_num):

    output_data = np.zeros((max_node_num+1,12))

    for a in range(len(leftovers)):
        if leftovers[a] == 0:
            output_data[a,0] = a
            output_data[a,1:4] = su2_data[a,1:4]
            output_data[a,8] = su2_data[a,13]
            output_data[a,9:12] = su2_data[a,15:18]

    output_data = output_data[~np.all(output_data == 0, axis=1)]

    return output_data

def get_surface_path_list(leading_edge,trailing_edge,node_connectivity,position_data,upper_lower_direction,lower_surface_direction,upper_lower_boundary):
    surface_paths = [[] for _ in range(len(leading_edge))]

    for a in range(np.min([len(leading_edge),len(trailing_edge)])):
        surface_paths[a] = a_star_search(leading_edge[a],trailing_edge[a],node_connectivity,position_data,upper_lower_direction,lower_surface_direction,upper_lower_boundary,set())

    return surface_paths

def save_list_as_csv(the_list,the_filepath):

    the_file = open(the_filepath,'w')

    for a in range(len(the_list)):
        for b in range(len(the_list[a])):
            the_file.write(str(the_list[a][b]))

            if b < (len(the_list[a])-1):
                the_file.write(',')
            elif a < (len(the_list)-1):
                the_file.write('\n')

    the_file.close()

    return

def read_csv_as_list(the_filepath):

    the_file = open(the_filepath,'r')
    file_data = the_file.read()
    the_file.close()

    file_data = re.split('\n',file_data)

    the_list = [[] for _ in range(len(file_data))]

    for a in range(len(file_data)):
        the_list[a] = re.split(',',file_data[a])
        for b in range(len(the_list[a])):
            the_list[a][b] = int(the_list[a][b])

    return the_list

def write_vtk_file(the_filepath,position_data,max_node_num,edge_list,number_of_elements,field_data):

    the_file = open(the_filepath,'w')
    the_file.write('# vtk DataFile Version 3.0\n')
    the_file.write('vtk output\n')
    the_file.write('ASCII\n')
    the_file.write('DATASET UNSTRUCTURED_GRID\n')
    the_file.write('POINTS '+str(max_node_num+1)+' float\n')

    for a in range(max_node_num+1):
        for b in range(len(position_data[a,:])):
            the_file.write(str(position_data[a,b]))

            if b < (len(position_data[a,:])-1):
                the_file.write(' ')
            else:
                the_file.write('\n')

    the_file.write('CELLS '+str(number_of_elements)+' '+str(number_of_elements*4)+'\n')

    for a in range(number_of_elements):
        the_file.write('3 ')
        for b in range(len(edge_list[a,:])):
            the_file.write(str(int(edge_list[a,b])))

            if b < (len(edge_list[a,:])-1):
                the_file.write(' ')
            else:
                the_file.write('\n')

    the_file.write('CELL_TYPES '+str(number_of_elements)+'\n')

    for a in range(number_of_elements):
        the_file.write('5\n')

    the_file.write('POINT_DATA '+str(max_node_num+1)+'\n')
    the_file.write('FIELD FieldData 2\n')
    the_file.write('Pressure 1 '+str(max_node_num+1)+' float\n')

    pressure_list = np.zeros((max_node_num+1))

    for a in range(len(field_data)):
        for b in range(len(field_data[a][:,0])):
            pressure_list[int(field_data[a][b,0])] = field_data[a][b,8]

    for a in range(max_node_num+1):
        the_file.write(str(pressure_list[a])+'\n')

    the_file.write('Skin_Friction_Coefficient 3 '+str(max_node_num+1)+' float\n')

    friction_list = np.zeros((max_node_num+1,3))

    for a in range(len(field_data)):
        for b in range(len(field_data[a][:,0])):
            friction_list[int(field_data[a][b,0]),:] = field_data[a][b,9:12]

    for a in range(max_node_num+1):
        for b in range(3):
            the_file.write(str(friction_list[a,b]))
            if b<3:
                the_file.write(' ')
            else:
                the_file.write('\n')

    return

def read_chordwise_corrections(the_filepath):

    corrections_data = np.genfromtxt(the_filepath, delimiter=',')
    percent_correction = np.ones((len(corrections_data[:,0]),len(corrections_data[0,:])-1))

    for a in range(len(corrections_data[0,:])-2):
        percent_correction[:,a+1] = corrections_data[:,a+2]/corrections_data[:,1]

    percent_correction[np.isnan(percent_correction)] = 1.

    correction_spline_list = []

    for a in range(len(percent_correction[0,:])):
        correction_spline_list.append(CubicSpline(corrections_data[:,0], percent_correction[:,a], axis=0))

    return correction_spline_list

def shifted_sigmoid(x,width):

    #Put x through the sigmoid function, but have x=0 be at the y=0 part of the sigmoid
    #x must be a normalised value from 0 to 1

    initial_boundary = 3.
    width_multiplier = 2.*initial_boundary/width
    scaled_x = x*width_multiplier
    scaled_x -= initial_boundary

    output = 1. / (1. + np.exp(-scaled_x))

    return output

def mirrored_shifted_sigmoid(x,width,ban_list):

    #Betwixt the ban_list, the output of this function shall tend to 0

    if x > ban_list[-1,1]:
        output = 0.
        return output

    for a in range(len(ban_list[:,0])):
        if x < ban_list[a,0]:
            ban_ID = a
            before_ID = True
            break
        elif x <= ban_list[a,1]:
            ban_ID = a
            before_ID = False
            break

    if before_ID:
        if ban_ID == 0:
            x1 = -x
        else:
            x1 = -x+ban_list[ban_ID-1,1]

        x2 = x-ban_list[ban_ID,0]
        sigmoid_1 = shifted_sigmoid(x1,width)
        sigmoid_2 = shifted_sigmoid(x2,width)
        output = (x - ban_list[ban_ID-1,1]) / (ban_list[ban_ID,0]-ban_list[ban_ID-1,1])
        if ban_ID == len(ban_list[:,0])-1:
            output = sigmoid_1
        else:
            output = sigmoid_1*(1.-output) + sigmoid_2*output

    else:
        x1 = x-ban_list[ban_ID,0]
        x2 = -x+ban_list[ban_ID,1]
        sigmoid_1 = shifted_sigmoid(x1,width)
        sigmoid_2 = shifted_sigmoid(x2,width)
        output = (x - ban_list[ban_ID,0]) / (ban_list[ban_ID,1] - ban_list[ban_ID,0])
        if ban_ID == 0:
            output = sigmoid_2
        elif ban_ID == len(ban_list[:,0])-1:
            output = sigmoid_1
        else:
            output = sigmoid_1*(1.-output) + sigmoid_2*output

    return output

def load_CF_corrections(max_nlf_lower,max_nlf_upper,root_filepath,laminar_lower_filename,laminar_upper_filename,turbulent_lower_filename,turbulent_upper_filename):

    laminar_upper_filepath = root_filepath+laminar_lower_filename
    laminar_lower_filepath = root_filepath+laminar_upper_filename
    turbulent_lower_filepath = root_filepath+turbulent_lower_filename
    turbulent_upper_filepath = root_filepath+turbulent_upper_filename

    #Remove double-spaces so numpy can read it easily
    for filename in [laminar_lower_filename,laminar_upper_filename,turbulent_lower_filename,turbulent_upper_filename]:
        the_file = open(filename,'r')
        file_data = the_file.read()
        the_file.close()
        while '  ' in file_data:
            file_data = file_data.replace('  ',' ')
            the_file = open(filename,'w')
            the_file.write(file_data)
            the_file.close()

    lower_laminar_data = np.genfromtxt(laminar_lower_filename, delimiter=' ')
    upper_laminar_data = np.genfromtxt(laminar_upper_filename, delimiter=' ')
    lower_turbulent_data = np.genfromtxt(turbulent_lower_filename, delimiter=' ')
    upper_turbulent_data = np.genfromtxt(turbulent_upper_filename, delimiter=' ')

    lower_laminar_spline_cf = CubicSpline(lower_laminar_data[:,0], lower_laminar_data[:,14], axis=0)
    upper_laminar_spline_cf = CubicSpline(upper_laminar_data[:,0], upper_laminar_data[:,14], axis=0)
    lower_turbulent_spline_cf = CubicSpline(lower_turbulent_data[:,0], lower_turbulent_data[:,14], axis=0)
    upper_turbulent_spline_cf = CubicSpline(upper_turbulent_data[:,0], upper_turbulent_data[:,14], axis=0)

    lower_laminar_spline_cp = CubicSpline(lower_laminar_data[:,0], lower_laminar_data[:,14], axis=0)
    upper_laminar_spline_cp = CubicSpline(upper_laminar_data[:,0], upper_laminar_data[:,14], axis=0)
    lower_turbulent_spline_cp = CubicSpline(lower_turbulent_data[:,0], lower_turbulent_data[:,14], axis=0)
    upper_turbulent_spline_cp = CubicSpline(upper_turbulent_data[:,0], upper_turbulent_data[:,14], axis=0)

    # for a in range(1000):
    #     print([str(a/(1000/max_nlf_lower)),lower_laminar_spline_cf(a/(1000/max_nlf_lower))/lower_turbulent_spline_cf(a/(1000/max_nlf_lower))])
    #     #print([str(a/(20/max_nlf_lower)),lower_laminar_spline(a/(20/max_nlf_lower))])
    # raise ValueError

    return lower_laminar_spline_cf, upper_laminar_spline_cf, lower_turbulent_spline_cf, upper_turbulent_spline_cf, lower_laminar_spline_cp, upper_laminar_spline_cp, lower_turbulent_spline_cp, upper_turbulent_spline_cp

def bool_array_to_decimal(the_array):

    cumsum = 0
    for a in range(len(the_array)):
        if the_array[-a-1]:
            cumsum += 2**a

    return cumsum

def generate_element_tree(edge_list,position_data,lower_set,upper_set):

    element_position_list = np.zeros((len(edge_list[:,0]),3))

    for a in range(len(edge_list[:,0])):

        cumulative_position = np.zeros((3))

        for b in range(len(edge_list[a,:])):
            cumulative_position += position_data[int(edge_list[a,b]),:]

        element_position_list[a,:] = cumulative_position / (b+1)

    class Point_tree:
        def __init__(self,the_points,point_list,depth,lower_set,upper_set,edge_list):
            self.point_list = point_list
            self.point_data = the_points
            self.lower_set = lower_set
            self.upper_set = upper_set
            self.children = []
            self.centroid_median = np.median(the_points[point_list,:],axis=0)
            #self.centroid_mean = np.mean(the_points[point_list,:],axis=0)

            if len(point_list) > 10:
                child_assignment = np.zeros((len(point_list)),dtype=int)
                for a in range(len(point_list)):
                    child_assignment[a] = bool_array_to_decimal(the_points[point_list[a],:] < self.centroid_median)

                assignment_list = [[] for _ in range(8)]
                for a in range(len(child_assignment)):
                    assignment_list[child_assignment[a]].append(point_list[a])

                for a in range(8):
                    self.children.append(Point_tree(the_points,assignment_list[a],depth+1,self.lower_set,self.upper_set,edge_list))

            else:

                self.which_set = []
                for element in point_list:
                    cumsum = 0
                    for node in edge_list[element,:]:
                        if int(node) in lower_set:
                            cumsum -= 1
                        elif int(node) in upper_set:
                            cumsum += 1

                    if cumsum > 0:
                        self.which_set.append(1)
                    elif cumsum < 0:
                        self.which_set.append(-1)
                    else:
                        self.which_set.append(0)

        def get_nearest_element(self,the_position,ignore_z,which_set):

            if len(self.children) > 0:

                search_truth_array = the_position < self.centroid_median
                nearest_element, nearest_distance = self.children[bool_array_to_decimal(search_truth_array)].get_nearest_element(the_position,ignore_z,which_set)

                if ignore_z:
                    search_truth_array[2] = not search_truth_array[2]
                    nearest_element_z, nearest_distance_z = self.children[bool_array_to_decimal(search_truth_array)].get_nearest_element(the_position,ignore_z,which_set)
                    if nearest_distance_z < nearest_distance:
                        nearest_element = nearest_element_z.copy()
                        nearest_distance = nearest_distance_z.copy()

            else:

                current_min = 1e10
                nearest_element = None
                for a in range(len(self.point_list)):
                    if which_set == 0 or self.which_set[a] == which_set:
                        if ignore_z:
                            current_dist = np.linalg.norm(the_position[0:2] - self.point_data[self.point_list[a],0:2])
                        else:
                            current_dist = np.linalg.norm(the_position - self.point_data[self.point_list[a],:])
                        if current_dist < current_min:
                            current_min = current_dist
                            nearest_element = self.point_list[a]

                nearest_distance = current_min

            return nearest_element, nearest_distance

    element_tree = Point_tree(element_position_list,np.arange(0,len(element_position_list[:,0])),0,lower_set,upper_set,edge_list)

    return element_tree

def interpolate_field_values(su2_data,which_fields,element_tree,edge_list,the_point,correct_Z,which_set):

    corrected_point = np.zeros((3))

    if len(the_point) == 2:
        extended_point = np.zeros((3))
        extended_point[0:2] = the_point.copy()
        extended_point[2] = 0.
    else:
        extended_point = the_point.copy()

    closest_element, _ = element_tree.get_nearest_element(extended_point,correct_Z,which_set)


    if correct_Z:
        for element in edge_list[closest_element,:]:
            corrected_point[2] += su2_data[int(element),3]

        corrected_point[2] = corrected_point[2] / len(edge_list[closest_element,:])

    else:
        corrected_point = the_point.copy()

    distances = []

    for element in edge_list[closest_element,:]:
        distances.append(np.linalg.norm(corrected_point-su2_data[int(element),1:4]))

    distances = distances/np.sum(distances)

    interpolated_field_values = np.zeros((len(which_fields)))

    for a in range(len(which_fields)):
        cumsum = 0.
        for b in range(len(edge_list[closest_element,:])):
            cumsum += su2_data[int(edge_list[closest_element,a]),which_fields[a]]*distances[b]
        interpolated_field_values[a] = cumsum

    return interpolated_field_values

def calculate_coefficients():
    # trace generated using paraview version 5.10.0-RC1

    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'Legacy VTK Reader'
    wingvtk = LegacyVTKReader(registrationName='wing.vtk', FileNames=['C:\\[redacted]\\Laminar_Regions\\HDMR\\wing.vtk'])

    UpdatePipeline(time=0.0, proxy=wingvtk)

    # create a new 'Calculator'
    calculator1 = Calculator(registrationName='Calculator1', Input=wingvtk)
    calculator1.AttributeType = 'Point Data'
    calculator1.CoordinateResults = 0
    calculator1.ResultNormals = 0
    calculator1.ResultTCoords = 0
    calculator1.ResultArrayName = 'Result'
    calculator1.Function = ''
    calculator1.ReplaceInvalidResults = 1
    calculator1.ReplacementValue = 0.0
    calculator1.ResultArrayType = 'Double'

    # Properties modified on calculator1
    calculator1.Function = ''

    UpdatePipeline(time=0.0, proxy=calculator1)

    # Properties modified on calculator1
    calculator1.Function = 'coords = Coords'

    # Properties modified on calculator1
    calculator1.Function = 'Coords = coords'

    # Properties modified on calculator1
    calculator1.CoordinateResults = 1
    calculator1.Function = 'coords'

    # set active source
    SetActiveSource(wingvtk)

    # create a new 'Extract Surface'
    extractSurface1 = ExtractSurface(registrationName='ExtractSurface1', Input=wingvtk)
    extractSurface1.PieceInvariant = 1
    extractSurface1.NonlinearSubdivisionLevel = 1
    extractSurface1.FastMode = 0
    extractSurface1.UseGeometryFilter = 0

    UpdatePipeline(time=0.0, proxy=extractSurface1)

    # create a new 'Generate Surface Normals'
    generateSurfaceNormals1 = GenerateSurfaceNormals(registrationName='GenerateSurfaceNormals1', Input=extractSurface1)
    generateSurfaceNormals1.FeatureAngle = 30.0
    generateSurfaceNormals1.Splitting = 1
    generateSurfaceNormals1.Consistency = 1
    generateSurfaceNormals1.FlipNormals = 0
    generateSurfaceNormals1.NonManifoldTraversal = 1
    generateSurfaceNormals1.ComputeCellNormals = 0
    generateSurfaceNormals1.PieceInvariant = 1

    # Properties modified on generateSurfaceNormals1
    generateSurfaceNormals1.FlipNormals = 1

    UpdatePipeline(time=0.0, proxy=generateSurfaceNormals1)

    # create a new 'Calculator'
    calculator2 = Calculator(registrationName='Calculator2', Input=generateSurfaceNormals1)
    calculator2.AttributeType = 'Point Data'
    calculator2.CoordinateResults = 0
    calculator2.ResultNormals = 0
    calculator2.ResultTCoords = 0
    calculator2.ResultArrayName = 'Result'
    calculator2.Function = ''
    calculator2.ReplaceInvalidResults = 1
    calculator2.ReplacementValue = 0.0
    calculator2.ResultArrayType = 'Double'

    # Properties modified on calculator2
    calculator2.ResultArrayName = 'Pressure_Coefficient'
    calculator2.Function = '(Pressure-26201.6)/(0.5*0.409727*219.921^2)'

    UpdatePipeline(time=0.0, proxy=calculator2)

    # create a new 'Calculator'
    calculator3 = Calculator(registrationName='Calculator3', Input=calculator2)
    calculator3.AttributeType = 'Point Data'
    calculator3.CoordinateResults = 0
    calculator3.ResultNormals = 0
    calculator3.ResultTCoords = 0
    calculator3.ResultArrayName = 'Result'
    calculator3.Function = ''
    calculator3.ReplaceInvalidResults = 1
    calculator3.ReplacementValue = 0.0
    calculator3.ResultArrayType = 'Double'

    # Properties modified on calculator3
    calculator3.ResultArrayName = 'CpForce'
    calculator3.Function = 'Pressure_Coefficient*Normals'

    UpdatePipeline(time=0.0, proxy=calculator3)

    # create a new 'Calculator'
    calculator4 = Calculator(registrationName='Calculator4', Input=calculator3)
    calculator4.AttributeType = 'Point Data'
    calculator4.CoordinateResults = 0
    calculator4.ResultNormals = 0
    calculator4.ResultTCoords = 0
    calculator4.ResultArrayName = 'Result'
    calculator4.Function = ''
    calculator4.ReplaceInvalidResults = 1
    calculator4.ReplacementValue = 0.0
    calculator4.ResultArrayType = 'Double'

    # Properties modified on calculator4
    calculator4.ResultArrayName = 'Total_Force'
    calculator4.Function = 'CpForce+Skin_Friction_Coefficient'

    UpdatePipeline(time=0.0, proxy=calculator4)

    # create a new 'Calculator'
    calculator5 = Calculator(registrationName='Calculator5', Input=calculator4)
    calculator5.AttributeType = 'Point Data'
    calculator5.CoordinateResults = 0
    calculator5.ResultNormals = 0
    calculator5.ResultTCoords = 0
    calculator5.ResultArrayName = 'Result'
    calculator5.Function = ''
    calculator5.ReplaceInvalidResults = 1
    calculator5.ReplacementValue = 0.0
    calculator5.ResultArrayType = 'Double'

    # Properties modified on calculator5
    calculator5.ResultArrayName = 'Drag'
    calculator5.Function = 'Total_Force_X*cos(0.0)+Total_Force_Z*sin(0.0)'

    UpdatePipeline(time=0.0, proxy=calculator5)

    # create a new 'Calculator'
    calculator6 = Calculator(registrationName='Calculator6', Input=calculator5)
    calculator6.AttributeType = 'Point Data'
    calculator6.CoordinateResults = 0
    calculator6.ResultNormals = 0
    calculator6.ResultTCoords = 0
    calculator6.ResultArrayName = 'Lift'
    calculator6.Function = ''
    calculator6.ReplaceInvalidResults = 1
    calculator6.ReplacementValue = 0.0
    calculator6.ResultArrayType = 'Double'

    # Properties modified on calculator6
    calculator6.Function = '-Total_Force_X*sin(0.0)+Total_Force_Z*cos(0.0)'

    UpdatePipeline(time=0.0, proxy=calculator6)

    # create a new 'Integrate Variables'
    integrateVariables1 = IntegrateVariables(registrationName='IntegrateVariables1', Input=calculator6)
    integrateVariables1.DivideCellDataByVolume = 0

    UpdatePipeline(time=0.0, proxy=integrateVariables1)

    # create a new 'Calculator'
    calculator7 = Calculator(registrationName='Calculator7', Input=integrateVariables1)
    calculator7.AttributeType = 'Point Data'
    calculator7.CoordinateResults = 0
    calculator7.ResultNormals = 0
    calculator7.ResultTCoords = 0
    calculator7.ResultArrayName = 'CL'
    calculator7.Function = ''
    calculator7.ReplaceInvalidResults = 1
    calculator7.ReplacementValue = 0.0
    calculator7.ResultArrayType = 'Double'

    # Properties modified on calculator7
    calculator7.Function = 'Lift/64.83938389994147'

    UpdatePipeline(time=0.0, proxy=calculator7)

    # create a new 'Calculator'
    calculator8 = Calculator(registrationName='Calculator8', Input=calculator7)
    calculator8.AttributeType = 'Point Data'
    calculator8.CoordinateResults = 0
    calculator8.ResultNormals = 0
    calculator8.ResultTCoords = 0
    calculator8.ResultArrayName = 'CD'
    calculator8.Function = ''
    calculator8.ReplaceInvalidResults = 1
    calculator8.ReplacementValue = 0.0
    calculator8.ResultArrayType = 'Double'

    # Properties modified on calculator8
    calculator8.Function = 'Drag/64.83938389994147'

    UpdatePipeline(time=0.0, proxy=calculator8)

    result = servermanager.Fetch(calculator8).GetPointData()
    CD = result.GetArray("CD").GetValue(0)
    CL = result.GetArray("CL").GetValue(0)

    return CL, CD

def laminar_sensitivity_function(x):

    #Defunct as of this commit, however only the variables put into main() need to be changed

    banned_span = np.zeros((4,2))
    banned_span[0,:] = [0.,(0.02+x[0])]
    banned_span[1,:] = [0.2+x[1],0.25+x[2]]
    banned_span[2,:] = [0.45+x[3],0.48+x[4]]
    banned_span[3,:] = [0.98+x[5],1]

    efficiency_lower = x[6]
    efficiency_upper = x[7]

    span_direction = [0,1,0]
    chord_direction = [1,0,0]

    su2_filename = 'C:/[redacted]/Laminar_Regions/example_su2_input.csv'

    main('C:/[redacted]/Laminar_Regions/mrsbw-V-BASE-newBL.su2',1,136218,4,136222,22500,158565,'wing',span_direction,chord_direction,0.6321,0.6739,efficiency_lower,efficiency_upper,banned_span,True,True,su2_filename)
    CL, CD = calculate_coefficients()

    return CL

def razor_corrections_interface(root_filepath,snapshot_filename,MSES_transition_lower,MSES_transition_upper,laminar_lower_filename,laminar_upper_filename,turbulent_lower_filename,turbulent_upper_filename,razor_out_filename,output_vtk_instead_of_csv):

    #root folder filepath: all additional filepaths will append to the root filepath

    #snapshot filename: path relative to root_folder_filepath that points to the snapshot

    #MSES_transition_lower/upper: These are the transition points identified in the MSES solutions
    #For 0 deg the values should be: 0.6508,0.1637
    #5 deg: 0.3835, 0.7077
    #2.5 deg: 0.472, 0.678
    #-5 deg: 0.7385, 0.1806

    #laminar/turbulent lower/upper filename: These are the paths relative to the root filepath for the MSES solutions for a given AOA

    #razor_out_filename: the path relative to the root filepath for the (unsorted) output csv

    #output_vtk_instead_of_csv: self-explanatory variable name. WARNING: this will use razor_out_filename

    #Areas of the span with fully turbulent flow
    banned_span = np.zeros((4,2))
    banned_span[0,:] = [0.,0.03]
    banned_span[1,:] = [0.2,0.25]
    banned_span[2,:] = [0.45,0.48]
    banned_span[3,:] = [0.96,1]

    #Always 1
    efficiency_lower = 1.0
    efficiency_upper = 1.0

    #Used to transform coordinates into the corresponding directions
    span_direction = [0,1,0]
    chord_direction = [1,0,0]

    #If the node IDs need to be regenerated, use this su2 file...
    su2_mesh_file = 'mrsbw-V-0_WT.su2'
    #...and check the following node IDs are correct...
    root_leading_edge_ID = 1
    tip_leading_edge_ID = 136218
    root_lower_trailing_edge_ID = 4
    tip_lower_trailing_edge_ID = 136222
    root_upper_trailing_edge_ID = 22500
    tip_upper_trailing_edge_ID = 158565
    #...and that this wing marker is correct...
    wing_marker_tag = 'wing'
    #...and set this to false so it knows to regenerate them
    get_paths_from_csv = True

    #If for some reason, you want the node positions to come from the su2 mesh rather than the csv, set this to false
    get_pos_from_csv = True

    main(su2_mesh_file,root_leading_edge_ID,tip_leading_edge_ID,root_lower_trailing_edge_ID,tip_lower_trailing_edge_ID,root_upper_trailing_edge_ID,tip_upper_trailing_edge_ID,wing_marker_tag,span_direction,chord_direction,MSES_transition_lower,MSES_transition_upper,efficiency_lower,efficiency_upper,banned_span,get_paths_from_csv,get_pos_from_csv,snapshot_filename,root_filepath,laminar_lower_filename,laminar_upper_filename,turbulent_lower_filename,turbulent_upper_filename,razor_out_filename,output_vtk_instead_of_csv)

    return

# sys.setrecursionlimit(10**6)

root_folder_filepath = 'C:/[redacted]/Laminar_Regions/'
snapshot_filename = 'sol_wing_00000_0.0.csv'
MSES_transition_lower = 0.6321
MSES_transition_upper = 0.6739
laminar_lower_filename = '0_laminar_lower.mses'
laminar_upper_filename = '0_laminar_upper.mses'
turbulent_lower_filename = '0_turbulent_lower.mses'
turbulent_upper_filename = '0_turbulent_upper.mses'
razor_out_filename = 'good_naming_convention.vtk'
output_vtk_instead_of_csv = True

razor_corrections_interface(root_folder_filepath,snapshot_filename,MSES_transition_lower,MSES_transition_upper,laminar_lower_filename,laminar_upper_filename,turbulent_lower_filename,turbulent_upper_filename,razor_out_filename,output_vtk_instead_of_csv)
