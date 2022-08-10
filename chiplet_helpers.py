
import qiskit


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
from math import ceil
from itertools import combinations, permutations
from qiskit import IBMQ
from math import floor

IBMQ.load_account()

############################################# VARIABLES #########################################


#machine information
device_names = {'falcon':['ibmq_toronto','ibmq_montreal','ibmq_mumbai','ibm_cairo','ibm_hanoi','ibm_auckland'],
                'falcon_5L':['ibmq_santiago','ibmq_bogota','ibmq_manila'],
                'falcon_5T':['ibmq_lima','ibmq_belem','ibmq_quito'],
                'falcon_7':['ibm_lagos','ibmq_jakarta','ibm_perth'],
                'falcon_16':['ibmq_guadalupe'],
                'humm': ['ibmq_brooklyn'],
                'eag': ['ibmq_washington']
               }

provider=IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='phy147')
coupling_map_f_5L = provider.get_backend('ibmq_manila').configuration().coupling_map
coupling_map_f_5T = provider.get_backend('ibmq_lima').configuration().coupling_map
coupling_map_f_7 = provider.get_backend('ibmq_jakarta').configuration().coupling_map
coupling_map_f_16 = provider.get_backend('ibmq_guadalupe').configuration().coupling_map
coupling_map_f = provider.get_backend('ibmq_toronto').configuration().coupling_map
#coupling_map_h = provider.get_backend('ibmq_brooklyn').configuration().coupling_map
coupling_map_e = provider.get_backend('ibm_washington').configuration().coupling_map


n_qubits_f_5L = provider.get_backend('ibmq_manila').configuration().n_qubits
n_qubits_f_5T = provider.get_backend('ibmq_lima').configuration().n_qubits
n_qubits_f_7 = provider.get_backend('ibmq_jakarta').configuration().n_qubits
n_qubits_f_16 = provider.get_backend('ibmq_guadalupe').configuration().n_qubits
n_qubits_f = provider.get_backend('ibmq_toronto').configuration().n_qubits
#n_qubits_h = provider.get_backend('ibmq_brooklyn').configuration().n_qubits
n_qubits_e = provider.get_backend('ibm_washington').configuration().n_qubits

#used inspiration from the heavy-hex pattern/real devices
freq_pattern_f_5L = {0:1, 1:2, 2:0, 3:2, 4:1}
freq_pattern_f_5T = {0:2, 1:1, 2:2, 3:2, 4:0}

freq_pattern_f_7 = {0:2, 1:1, 2:2, 3:2, 4:2, 5:0, 6:2 }
freq_pattern_f_16 ={0:2,1:1,2:2,3:0,4:2,5:2,6:2,7:0,8:1,9:2,10:2,11:2,12:1,13:2,14:0,15:2}
freq_pattern_f ={0:2,1:1,2:2,3:0,4:2,5:2,6:2,7:0,8:1,9:2,10:2,11:2,12:1,13:2,14:0,15:2,16:2,17:2,18:0,19:1,20:2,21:2,22:2,23:1,24:2,25:0,26:2}
#freq_pattern_h 
#freq_pattern_e 

############################################# CLASS: create_backend #########################################


from qiskit.providers import BaseBackend
from qiskit.test.mock import FakeBackend
from qiskit.providers.models import BackendProperties, BackendConfiguration
from qiskit.providers.models.backendconfiguration import GateConfig

class create_backend(FakeBackend):
    """create a backend with custom 2q gate error values.
    
        gate_error by default {}. add error with keys as 
        '[0, 1]', '[1, 0]', etc.
    
    """

    def __init__(self,coupling_map,name='custom_backend',gate_errors={}):
        
        self.gate_errors=gate_errors
        unique_qubits = list(set().union(*coupling_map))
        n_qubits = len(unique_qubits)

        configuration = BackendConfiguration(
            backend_name=name,
            backend_version='0.0.0',
            n_qubits=n_qubits,
            #basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
            basis_gates=['x', 'sx', 'rz', 'cx', 'id'],
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=65536,
            #gates=[GateConfig(name='TODO', parameters=[], qasm_def='TODO')],
            gates = self.set_up_gates(n_qubits,coupling_map),
            coupling_map=coupling_map,
        )

        super().__init__(configuration)
        

    def set_up_gates(self,n_qubits,coupling_map):
        gate_names = ['id','rz','sx','x','cx','reset']
        gate_details = []
        
        for i in gate_names:
            #set name
            name = i
            
            #set parameters
            if i == 'id' or i == 'sx' or i == 'x' or i == 'cx' or i == 'reset':
                parameters = []
            if i == 'rz':
                parameters = ['theta']
                
            #set qasm definition
            if i == 'id':
                qasm_def = 'gate id q { U(0, 0, 0) q; }'
            elif i == 'rz':
                qasm_def = 'gate rz(theta) q { U(0, 0, theta) q; }'
            elif i == 'sx':
                qasm_def = 'gate sx q { U(pi/2, 3*pi/2, pi/2) q; }'
            elif i == 'x':
                qasm_def = 'gate x q { U(pi, 0, pi) q; }'
            elif i == 'cx':
                qasm_def = 'gate cx q0, q1 { CX q0, q1; }'
            elif i == 'reset':
                qasm_def = 'gate reset q {RESET q;}'
                
            #set coupling map
            if i == 'id' or i == 'sx' or i == 'x' or i == 'rz' or i == 'reset':
                gate_map = []
                for j in range(n_qubits):
                    gate_map.append([j])
            if i == 'cx':
                gate_map = coupling_map
                
            gate_details.append(GateConfig(name=name,
                                           parameters=parameters,
                                           qasm_def=qasm_def,
                                           coupling_map=gate_map))
        return gate_details
            
        
    
    def properties(self):
        """Return backend properties"""

        """Return backend properties"""

        coupling_map = self.configuration().coupling_map
        unique_qubits = list(set().union(*coupling_map))
        

        for pair in coupling_map:
            if str(pair) not in self.gate_errors:
                self.gate_errors[str(pair)] = 0.0

        properties = {
            #TODO: Add real values from machine when building up
            'backend_name': self.name(),
            'backend_version': self.configuration().backend_version,
            'last_update_date': '2000-01-01 00:00:00Z',
            'qubits': [
                [
                    {
                        "date": "2000-01-01 00:00:00Z",
                        "name": "T1",
                        "unit": "\u00b5s",
                        "value": 0.0
                    },
                    {
                        "date": "2000-01-01 00:00:00Z",
                        "name": "T2",
                        "unit": "\u00b5s",
                        "value": 0.0
                    },
                    {
                        "date": "2000-01-01 00:00:00Z",
                        "name": "frequency",
                        "unit": "GHz",
                        "value": 0.0
                    },
                    {
                        "date": "2000-01-01 00:00:00Z",
                        "name": "readout_error",
                        "unit": "",
                        "value": 0.0
                    }
                ] for _ in range(len(unique_qubits))
            ],
            'gates': [{
                "gate": "cx",
                "name": "CX" + str(pair[0]) + "_" + str(pair[1]),
                "parameters": [
                    {
                        "date": "2000-01-01 00:00:00Z",
                        "name": "gate_error",
                        "unit": "",
                        "value":  self.gate_errors[str(pair)]
                    }
                ],
                "qubits": [
                    pair[0],
                    pair[1]
                ]
            } for pair in coupling_map],
            'general': []
        }

        return BackendProperties.from_dict(properties)

############################################# FUNCTIONS: create_backend helpers #########################

    
def create_full_error_dict_cpu(coupling_map,avg_error,gate_errors={}):
    
    for pair in coupling_map:
        if str(pair) not in gate_errors:
            gate_errors[str(pair)] = avg_error
    return gate_errors



############################################# CLASS: simple_qc #########################################


class simple_qc:  
      
    '''
    layout_type - str: 'heavy_hex', 'heavy_hex_chiplet' 'grid' , 'heavy_square'
    
    can create simple_qc from a dictionary if from_dict == True and input_dict is provided.
    '''
    def __init__(self,layout_type=None, n_qubits=None, anharmonicity=None, build_from_dict=False, input_dict = {}):  
        
        if build_from_dict == True:
            self.from_dict(input_dict)
        
        else:
        
            if layout_type not in ['heavy_hex','heavy_hex_chiplet', 'grid' , 'heavy_square']:
                print('invalid layout type.')
                
            if n_qubits == None or anharmonicity == None:
                print('please specify required qc information (n_qubits, anharmonicity, etc.)')
            
            self.layout_type = layout_type
            self.n_qubits = n_qubits
            self.anharmonicity = anharmonicity
            
            #things to be set
            self.coupling_map = None
            self.n_freq = None
            self.graph = None
            self.freq_pattern = None
            self.freq_assignment_ideal = None
            self.freq_assignment_actual = None
            self.collision_free = None
            self.error_dict = None
            self.link_list = None
 
    def from_dict(self,input_dict):
        self.layout_type = input_dict['layout_type']
        self.n_qubits = input_dict['n_qubits']
        self.anharmonicity = input_dict['anharmonicity']
        self.coupling_map = input_dict['coupling_map']
        self.n_freq = input_dict['n_freq']
        self.graph = None
        self.freq_pattern = input_dict['freq_pattern']
        self.freq_assignment_ideal = input_dict['freq_assignment_ideal']
        self.freq_assignment_actual = input_dict['freq_assignment_actual']
        self.collision_free = input_dict['collision_free']
        self.error_dict = input_dict['error_dict']
        self.graph = graph_from_machine(self.n_qubits,self.coupling_map)
        if 'link_list' in input_dict.keys():
            self.link_list = input_dict['link_list']
        
    
    
    def to_dict(self):
        new_dict = {'layout_type': self.layout_type,
                    'n_qubits': self.n_qubits,
                    'anharmonicity': self.anharmonicity,
                    'coupling_map':self.coupling_map,
                    'n_freq':self.n_freq,
                    'freq_pattern':self.freq_pattern,
                    'freq_assignment_ideal': self.freq_assignment_ideal,
                    'freq_assignment_actual':self.freq_assignment_actual,
                    'collision_free': self.collision_free,
                    'error_dict': self.error_dict
                    }
        
        return new_dict
    
    
    def set_coupling_map(self):
        if self.layout_type == 'grid':
            self.n_freq = 5
            self.coupling_map = create_grid_map(self.n_qubits)
            self.graph = graph_from_machine(self.n_qubits,self.coupling_map)
        
        elif self.layout_type == 'heavy_hex':
            self.n_freq = 3
            self.coupling_map = create_heavyhex_map(self.n_qubits)
            self.graph = graph_from_machine(self.n_qubits,self.coupling_map)
            
        elif self.layout_type == 'heavy_hex_chiplet':
            self.n_freq = 3
            self.coupling_map = create_heavyhex_map_for_chiplets(self.n_qubits)
            self.graph = graph_from_machine(self.n_qubits,self.coupling_map)
        
        elif self.layout_type == 'heavy_square':
            self.n_freq = 3
            self.coupling_map = create_heavysquare_map(self.n_qubits)
            self.graph = graph_from_machine(self.n_qubits,self.coupling_map)
            
    def set_freq_pattern(self):
        #higher number qubit should be control of lower number qubit
        
        if self.freq_pattern != None:
            return
        
        elif self.layout_type == 'grid':
            #self.freq_pattern = nx.equitable_color(self.graph,num_colors=self.n_freq)
            self.freq_pattern = freq_5_pattern_grid(self)
            
        elif self.layout_type == 'heavy_hex':
            self.freq_pattern = freq_3_pattern_heavy_hex(self)
            
        elif self.layout_type == 'heavy_hex_chiplet':
            self.freq_pattern = freq_3_pattern_heavy_hex_for_chiplets(self)
            
        elif self.layout_type == 'heavy_square':
            self.freq_pattern = freq_3_pattern_heavy_square(self)
        
    
    
    def set_ideal_freq_assignment(self,freq_assignment_list):
        '''
        These are the target frequencies on-chip
        Index in list corresponds to integers in frequency pattern. 
        List should have length equal to n_freq.
        '''
        if self.freq_assignment_ideal != None:
            return
        
        if len(freq_assignment_list) != self.n_freq:
            print('length of frequency assignment list does not correspond with n_freq.')
            return
            
        self.freq_assignment_ideal = {}
        for i in self.freq_pattern:
            self.freq_assignment_ideal[i] = freq_assignment_list[self.freq_pattern[i]]
    
    def set_actual_freq_assignment(self,sigma_f):
        '''
        Sets actual qubit frequency. The fabrication process can be modeled by adding a Gaussian noise. 
        sigma_f - fabrication precision parameter (MHz)
        '''
        
        if self.freq_assignment_ideal == None:
            print('ideal/target frequencies need assignment.')
        
        self.freq_assignment_actual = {}
        for i in self.freq_assignment_ideal:
            self.freq_assignment_actual[i] = np.random.normal(loc=self.freq_assignment_ideal[i], 
                                                              scale=sigma_f)
            
    def return_error_stat(self,error_stat='avg'):
        '''
        return avg, std, median of gate errors.
        '''
        temp_list = []
        if self.error_dict == None:
            return
        
        for i in self.error_dict:
            temp_list.append(self.error_dict[i])
        
        if error_stat == 'avg':
            return np.average(temp_list)
        if error_stat == 'std':
            return np.std(temp_list)
        if error_stat == 'median':
            return np.median(temp_list)
            

############################################# FUNCTIONS: FIDELITY ASSIGN. ######################################


def return_detune_dist_to_infid_dict(cx_infid_detuning):
    '''
    input: cx_infid_detuning that is gathered from historical data
    output: detune_dist_to_infid - creates detuning buckets (described below) 
    and sorts assocated cx infidelity values.
    
    Mean and std of these grouped datasets will be used to assign fidelities 
    according to frequency values resulting from simulation.
    
    groups of detuning as follows (GHz):
        0: [:-0.2)
        1: [-0.2:-0.1)
        2: [-0.1:0)
        3: [0:0.1)
        4: [0.1:0.2)
        5: [0.2:)
    '''
    detune_dist_to_infid = {0:[],
                           1:[],
                           2:[],
                           3:[],
                           4:[],
                           5:[]}
    
    
    for i in cx_infid_detuning:
        detuning_val = i[0]
        error_val = i[1]
        if detuning_val < -0.2:
            detune_dist_to_infid[0].append(error_val)
            
        elif (detuning_val >= -0.2 and detuning_val < -0.1):
            detune_dist_to_infid[1].append(error_val)
            
        elif (detuning_val >= -0.1 and detuning_val < 0):
            detune_dist_to_infid[2].append(error_val)
        
        elif (detuning_val >= 0 and detuning_val < 0.1):
            detune_dist_to_infid[3].append(error_val)
            
        elif (detuning_val >= 0.1 and detuning_val < 0.2):
            detune_dist_to_infid[4].append(error_val)
            
        elif detuning_val >= 0.2 :
            detune_dist_to_infid[5].append(error_val)
        
        
    return detune_dist_to_infid

def return_error_dictionary(device,cx_infid_detuning):
    '''
    input - 
    device
    cx_infid_detuning - list of fidelity detuning & corresponding fidelity pairing. This information is 
    created using averages from historical data.
    
    output-
    error_dict - this will be assiigned to a device. Every 2q pairing (for a device
    in the collision-free yield) should have a CX infidelity.
    '''
    
    detune_dist_to_infid = return_detune_dist_to_infid_dict(cx_infid_detuning)
    c_map = device.coupling_map
    freq_assignment = device.freq_assignment_actual
    error_dict = {}
    
    for i in c_map:
        if str(i) not in error_dict:
            q1 = i[0]
            q2 = i[1]
            detuning_val = q1-q2
            
            if detuning_val < -0.2:
                error_dict[str(i)] = np.random.choice(detune_dist_to_infid[0])
                error_dict[str([q2,q1])] = error_dict[str(i)]
                
            elif (detuning_val >= -0.2 and detuning_val < -0.1):
                error_dict[str(i)] = np.random.choice(detune_dist_to_infid[1])
                error_dict[str([q2,q1])] = error_dict[str(i)]

            elif (detuning_val >= -0.1 and detuning_val < 0):
                error_dict[str(i)] = np.random.choice(detune_dist_to_infid[2])
                error_dict[str([q2,q1])] = error_dict[str(i)]
            
            elif (detuning_val >= 0 and detuning_val < 0.1):
                error_dict[str(i)] = np.random.choice(detune_dist_to_infid[3])
                error_dict[str([q2,q1])] = error_dict[str(i)]
                
            elif (detuning_val >= 0.1 and detuning_val < 0.2):
                error_dict[str(i)] = np.random.choice(detune_dist_to_infid[4])
                error_dict[str([q2,q1])] = error_dict[str(i)]
                
            elif detuning_val >= 0.2 :
                error_dict[str(i)] = np.random.choice(detune_dist_to_infid[5])
                error_dict[str([q2,q1])] = error_dict[str(i)]
                
    return error_dict
    

############################################# FUNCTIONS: GRAPH HELPERS #########################################

def graph_from_machine(n_qubits,c_map,add_weights = False, weight_value=1):
    G = nx.Graph()
    G.add_nodes_from(np.arange(n_qubits))
    G.add_edges_from(c_map)
    if add_weights == True:
        nx.set_edge_attributes(G, values = weight_value, name = 'weight')
    return G

def draw_graph(G):
    pos = nx.spring_layout(G)
    fig = plt.figure(figsize=(14, 14))
    nx.draw_networkx_nodes(G, pos,node_color='red', node_size=300)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)
    plt.show()



def get_furthest_nodes(G):
    sp_length = {} # dict containing shortest path distances for each pair of nodes
    diameter = None # will contain the graphs diameter (length of longest shortest path)
    furthest_node_list = [] # will contain list of tuple of nodes with shortest path equal to diameter
    
    for node in G.nodes:
        # Get the shortest path from node to all other nodes
        sp_length[node] = nx.single_source_shortest_path_length(G,node)
        longest_path = max(sp_length[node].values()) # get length of furthest node from node
        
        # Update diameter when necessary (on first iteration and when we find a longer one)
        if diameter == None:
            diameter = longest_path # set the first diameter
            
        # update the list of tuples of furthest nodes if we have a best diameter
        if longest_path >= diameter:
            diameter = longest_path
            
            # a list of tuples containing
            # the current node and the nodes furthest from it
            node_longest_paths = [(node,other_node)
                                      for other_node in sp_length[node].keys()
                                      if sp_length[node][other_node] == longest_path]
            if longest_path > diameter:
                # This is better than the previous diameter
                # so replace the list of tuples of diameter nodes with this nodes
                # tuple of furthest nodes
                furthest_node_list = node_longest_paths
            else: # this is equal to the current diameter
                # add this nodes tuple of furthest nodes to the current list    
                furthest_node_list = furthest_node_list + node_longest_paths
                
    # return the diameter,
        # all pairs of nodes with shortest path length equal to the diameter
        # the dict of all-node shortest paths
    return({'diameter':diameter,
            'furthest_node_list':furthest_node_list,
            'node_shortest_path_dicts':sp_length})

def determine_diameter_path(G):
    node_info = get_furthest_nodes(G)
    diameter = node_info['diameter']
    for item in node_info['furthest_node_list']:
        if (len(nx.dijkstra_path(G,item[0],item[1]))-1) == diameter:
            return nx.dijkstra_path(G,item[0],item[1])
        

def sum_diam_path_weights(G,path):
    total = 0 
    for i in range(1,len(path)):
        total = total + G.get_edge_data(path[i],path[i-1])['weight']
        
    return total
    
############################################# FUNCTIONS: CHECK FREQ. COLLISION ##################################

def check_collision_conditions_two_qubits(freq_a, freq_b, anharm,thresholds = []):
    '''
    check if two frequencies satisfy frequency collision criteria.
    needs two frequencies and anharmicity
    control qubit will have higher frequency. set that first.
    
    this function examines conditions 1-4 (indexed 0-3) of laser tuning paper.
    '''
    
    q_c = max([freq_a,freq_b])
    q_t = min([freq_a,freq_b])
    
    collision = False
    
    if thresholds == []:
        thresholds = [0.017,
                     0.004,
                     0.030,
                     None]
    
    #condition 0
    if abs(q_c - q_t) < thresholds[0]:
        collision = True
        return collision 
    
    #condition 1
    if abs(q_c - q_t + (anharm/2)) < thresholds[1]:
        collision = True
        return collision 
    
    #condition 2
    if abs(q_c - q_t - anharm) < thresholds[2]:
        collision = True
        return collision 
    
    #condition 3
    if abs(q_t - q_c) < anharm or (q_c < q_t):
        collision = True
        return collision
    
    return collision

def count_collision_conditions_two_qubits(freq_a, freq_b, anharm,thresholds = []):
    '''
    check if two frequencies satisfy frequency collision criteria.
    needs two frequencies and anharmicity
    control qubit will have higher frequency. set that first.
    
    this function examines conditions 1-4 (indexed 0-3) of laser tuning paper.
    '''
    
    q_c = max([freq_a,freq_b])
    q_t = min([freq_a,freq_b])
    
    collisions = 0
    
    if thresholds == []:
        thresholds = [0.017,
                     0.004,
                     0.030,
                     None]
    
    #condition 0
    if abs(q_c - q_t) < thresholds[0]:
        collisions = collisions + 1
    
    #condition 1
    if abs(q_c - q_t + (anharm/2)) < thresholds[1]:
        collisions = collisions + 1
    
    #condition 2
    if abs(q_c - q_t - anharm) < thresholds[2]:
        collisions = collisions + 1 
    
    #condition 3
    if abs(q_t - q_c) < anharm or (q_c < q_t):
        collisions = collisions + 1
    
    return collisions
    
def check_collision_conditions_three_qubits(freq_a, freq_b, freq_c, anharm,thresholds = []):
    '''
    check if three connected qubits satisfy frequency collision criteria.
    needs three frequencies and anharmicity.
    
    this function examines conditions 5-7 (indexed 0-2) of laser tuning paper.
    
    qubit_a <-> qubit_b <-> qubit_c
    '''
    
    if thresholds == []:
        thresholds = [0.017,
                     0.025,
                     0.017]
    collision = False
    
    #condition 0
    if abs(freq_a - freq_c) < thresholds[0]:
        collision = True
        return collision
    
    #condition 1
    if (abs(freq_a - freq_c - anharm) < thresholds[1]) or (abs(freq_a - freq_c + anharm) < thresholds[1]):
        collision = True
        return collision
    
    #condition 2
    if abs(2*freq_b + anharm - freq_a - freq_c) < thresholds[2]:
        collision = True
        return collision
    
    
    return collision

def count_collision_conditions_three_qubits(freq_a, freq_b, freq_c, anharm, thresholds=[]):
    '''
    check if three connected qubits satisfy frequency collision criteria.
    needs three frequencies and anharmicity.
    
    this function examines conditions 5-7 (indexed 0-2) of laser tuning paper.
    
    qubit_a <-> qubit_b <-> qubit_c
    '''
    if thresholds == []:
        thresholds = [0.017,
                     0.025,
                     0.017]
    collisions = 0
    
    #condition 0
    if abs(freq_a - freq_c) < thresholds[0]:
        collisions = collisions + 1
    
    #condition 1
    if (abs(freq_a - freq_c - anharm) < thresholds[1]) or (abs(freq_a - freq_c + anharm) < thresholds[1]):
        collisions = collisions + 1
    
    #condition 2
    if abs(2*freq_b + anharm - freq_a - freq_c) < thresholds[2]:
        collisions = collisions + 1
    
    
    return collisions

def check_device_freq_collisions(device, values_used_ideal = False, two_q_thresholds=[],three_q_thresholds=[]):
    '''-
    if returns true, there's a frequency collision
    '''
    
    anharm = device.anharmonicity
    n_qubits = device.n_qubits
    
    if values_used_ideal:
        freq_assignment = device.freq_assignment_ideal
    else:
        freq_assignment = device.freq_assignment_actual
    
    
    collision_found = False
    for i in range(n_qubits):
        near_neighbors = list(nx.neighbors(device.graph,i))
        for j in near_neighbors:
            freq_a = freq_assignment[i]
            freq_b = freq_assignment[j]
            
            collision_found = check_collision_conditions_two_qubits(freq_a, freq_b, anharm,thresholds=two_q_thresholds)
            
            if collision_found == True:
                return collision_found
            
        if (device.layout_type == 'heavy_hex' and device.freq_pattern[i]!=2):
            continue
        if (device.layout_type == 'heavy_hex_chiplet' and device.freq_pattern[i]!=2):
            continue
        elif (device.layout_type == 'heavy_square' and device.freq_pattern[i]!=2):
            continue
        
        neighbor_pairs = list(permutations(near_neighbors,2))
        for j in neighbor_pairs:
            freq_a = freq_assignment[j[0]]
            freq_b = freq_assignment[i]
            freq_c = freq_assignment[j[1]]
            
            if device.layout_type == 'grid':
                if freq_b < freq_a or freq_b < freq_c:
                    continue
            
            collision_found = check_collision_conditions_three_qubits(freq_a,freq_b,freq_c,anharm,thresholds=three_q_thresholds)
            
            if collision_found == True:
                return collision_found
            
    return collision_found

def count_device_freq_collisions(device, values_used_ideal = False, two_q_thresholds=[],three_q_thresholds=[]):
    '''-
    counts te number of frequency collisions
    '''
    
    anharm = device.anharmonicity
    n_qubits = device.n_qubits
    
    if values_used_ideal:
        freq_assignment = device.freq_assignment_ideal
    else:
        freq_assignment = device.freq_assignment_actual
    
    
    collisions_found = 0
    for i in range(n_qubits):
        near_neighbors = list(nx.neighbors(device.graph,i))
        for j in near_neighbors:
            freq_a = freq_assignment[i]
            freq_b = freq_assignment[j]
            
            collisions_found = collisions_found + count_collision_conditions_two_qubits(freq_a, freq_b, anharm,thresholds=two_q_thresholds)
            

            
        if (device.layout_type == 'heavy_hex' and device.freq_pattern[i]!=2):
            continue
        if (device.layout_type == 'heavy_hex_chiplet' and device.freq_pattern[i]!=2):
            continue
        elif (device.layout_type == 'heavy_square' and device.freq_pattern[i]!=2):
            continue
        
        neighbor_pairs = list(permutations(near_neighbors,2))
        for j in neighbor_pairs:
            freq_a = freq_assignment[j[0]]
            freq_b = freq_assignment[i]
            freq_c = freq_assignment[j[1]]
            
            if device.layout_type == 'grid':
                if freq_b < freq_a or freq_b < freq_c:
                    continue
            
            collisions_found = collisions_found + count_collision_conditions_three_qubits(freq_a,freq_b,freq_c,anharm,thresholds=three_q_thresholds)
            
            
    return collisions_found

############################################# FUNCTIONS: FREQ. PATTERN/ASSIGN ###################################

def calculate_f_12(f_01,anharmonicity):
    return f_01 + anharmonicity

def freq_5_pattern_grid(device):
    '''
    assign 5 freq. pattern for grid device according to arXiv: 2009.00781
    '''
    
    starting_freq_assignment = [0,2,4,1,3]
    n_qubits = device.n_qubits
    edge_length = ceil(np.sqrt(n_qubits))
    freq_pattern = {}
    row_number = 0
    qubit_assignment = starting_freq_assignment[row_number]
    
    for i in range(n_qubits):
        freq_pattern[i]=qubit_assignment%5
        qubit_assignment = qubit_assignment + 1
        if (i+1)%edge_length == 0:
            
            row_number = row_number +1
            qubit_assignment = starting_freq_assignment[row_number%5]
            
    return freq_pattern
    

def freq_3_pattern_heavy_square(device):
    '''
    assign using pattern in arXiv: 2009.00781
    '''
        #fix initial values for smallest structure
    n_qubits = device.n_qubits
    
    q_per_row = 3
    q_connect = 2
    n_rows = 2
    n_connect_rows = n_rows - 1
    dim_found = False
    
    while dim_found == False:
    
        if n_qubits > q_per_row*n_rows + q_connect*n_connect_rows:
            q_per_row = q_per_row + 2
            q_connect = q_connect + 1
            n_rows = n_rows + 1
            n_connect_rows = n_rows - 1
        
        else: 
            dim_found = True
    
                
    #all connecting row qubits will be assigned '2'
    #other rows alternate:
    #      1,2,0,2....
    #      0,2,1,2....
    
    freq_row_patterns = [[1,2,0,2],
                        [0,2,1,2]]
    
    freq_pattern = {}
    
    qubit_counter = 0
    
    while qubit_counter < n_qubits:
        for i in range(q_per_row):
            freq_pattern[qubit_counter]= freq_row_patterns[0][i%len(freq_row_patterns[0])]
            qubit_counter = qubit_counter + 1
            
            if qubit_counter >= n_qubits:
                return freq_pattern
            
        for i in range(q_connect):
            freq_pattern[qubit_counter] = 2
            qubit_counter = qubit_counter + 1
            
            if qubit_counter >= n_qubits:
                return freq_pattern
            
        for i in range(q_per_row):
            freq_pattern[qubit_counter]= freq_row_patterns[1][i%len(freq_row_patterns[1])]
            qubit_counter = qubit_counter + 1
            
            if qubit_counter >= n_qubits:
                return freq_pattern
            
        for i in range(q_connect):
            freq_pattern[qubit_counter] = 2
            qubit_counter = qubit_counter + 1
            
            if qubit_counter >= n_qubits:
                return freq_pattern



def freq_3_pattern_heavy_hex(device):
    '''
    ibm devices will use fixed coloring inspired by real devices.
    
    other maps will be be assigned using below algorithm.
    
    '''
    
    n_qubits = device.n_qubits
    
    if n_qubits == n_qubits_f_5L:
        return freq_pattern_f_5L 
    #    #return freq_pattern_f_5T #returns 'T' pattern
    #
    if n_qubits == n_qubits_f_7:
        return freq_pattern_f_7 
    #
    if n_qubits == n_qubits_f_16:
        return freq_pattern_f_16 
    #
    if n_qubits == n_qubits_f:
        return freq_pattern_f 
    #
    #if n_qubits == n_qubits_h:
    #    return freq_pattern_h 
    #
    #if n_qubits == n_qubits_e:
    #    return freq_pattern_e 

    
    #fix initial values for smallest structure
    q_per_row = 7
    q_connect = 2
    n_rows = 3
    n_connect_rows = n_rows - 1
    dim_found = False
    
    while dim_found == False:
    
        if n_qubits > q_per_row*n_rows + q_connect*n_connect_rows:
            q_per_row = q_per_row + 4
            q_connect = q_connect + 1
            n_rows = n_rows + 2
            n_connect_rows = n_rows - 1
        
        else: 
            dim_found = True
            
            
    #all connecting row qubits will be assigned '2'
    #other rows alternate:
    #      1,2,0,2....
    #      0,2,1,2....
    
    freq_row_patterns = [[1,2,0,2],
                        [0,2,1,2]]
    
    freq_pattern = {}
    
    qubit_counter = 0
    
    while qubit_counter < n_qubits:
        for i in range(q_per_row):
            freq_pattern[qubit_counter]= freq_row_patterns[0][i%len(freq_row_patterns[0])]
            qubit_counter = qubit_counter + 1
            
            if qubit_counter >= n_qubits:
                return freq_pattern
            
        for i in range(q_connect):
            freq_pattern[qubit_counter] = 2
            qubit_counter = qubit_counter + 1
            
            if qubit_counter >= n_qubits:
                return freq_pattern
            
        for i in range(q_per_row):
            freq_pattern[qubit_counter]= freq_row_patterns[1][i%len(freq_row_patterns[1])]
            qubit_counter = qubit_counter + 1
            
            if qubit_counter >= n_qubits:
                return freq_pattern
            
        for i in range(q_connect):
            freq_pattern[qubit_counter] = 2
            qubit_counter = qubit_counter + 1
            
            if qubit_counter >= n_qubits:
                return freq_pattern
    
def freq_3_pattern_heavy_hex_for_chiplets(device):
    '''
    this is for the chiplet devices. must use the optimal qubit counts
    or will get unexpected results!!!!!!
    
    '''
    
    n_qubits = device.n_qubits
    


    
    #fix initial values for smallest structure
    
    q_per_row = 3 #this is actually total qubits per row minus one since the extra in the row allocated for in n_right_connect_q
    q_connect = 1
    n_rows = 2
    n_connect_rows = n_rows 
    n_right_connect_q = n_rows # right connection qubits will be placed last
    dim_found = False
    
    
    while dim_found == False:
    
        if n_qubits > q_per_row*n_rows + q_connect*n_connect_rows + n_right_connect_q:
            q_per_row = q_per_row + 4
            q_connect = q_connect + 1
            n_rows = n_rows + 2
            n_connect_rows = n_rows 
            n_right_connect_q = n_rows
        
        else: 
            dim_found = True
            
            
    #all connecting row qubits will be assigned '2'
    #other rows alternate:
    #      1,2,0,2....
    #      0,2,1,2....
    
    freq_row_patterns = [[1,2,0,2],
                        [0,2,1,2]]
    
    freq_pattern = {}
    
    qubit_counter = 0
    
    while qubit_counter < n_qubits-n_right_connect_q:
        for i in range(q_per_row):
            freq_pattern[qubit_counter]= freq_row_patterns[0][i%len(freq_row_patterns[0])]
            qubit_counter = qubit_counter + 1
            
            if qubit_counter >= n_qubits-n_right_connect_q:
                break
            
        for i in range(q_connect):
            freq_pattern[qubit_counter] = 2
            qubit_counter = qubit_counter + 1
            
            if qubit_counter >= n_qubits-n_right_connect_q:
                break
            
        for i in range(q_per_row):
            freq_pattern[qubit_counter]= freq_row_patterns[1][i%len(freq_row_patterns[1])]
            qubit_counter = qubit_counter + 1
            
            if qubit_counter >= n_qubits-n_right_connect_q:
                break
            
        for i in range(q_connect):
            freq_pattern[qubit_counter] = 2
            qubit_counter = qubit_counter + 1
            
            if qubit_counter >= n_qubits-n_right_connect_q:
                break
                
    for i in  range(qubit_counter,n_qubits):
        freq_pattern[i] = 2
    
    return freq_pattern


############################################# FUNCTIONS: MCMs and LINKS #########################################

import ast

def return_link_infid():
    '''
    using coherence-limited fidelity from arXiv:2102.13293
    '''
    fidelity_vals = np.array([98.58,92.07,88.84,96.74,88.17,97.4,90.81,96.92,77.25,98.2])/100
    return 1-(fidelity_vals)
    

def create_mcms(rows,colums,chips):
    '''
    This takes in chiplets and puts them into multi-chip module (MCM) configurations.
    
    rows - number of rows in MCM
    columns - number of columns in MCM
    chips - the provided chiplets sent as a list. all elligible chips part of the no-collision yield.
    
    multiple configurations will be sent back (up to chips! configurations).
    '''
    
    return None

    
def assemble_single_mcm(rows,columns,chips,use_input_link_vals=False,input_link_vals=[]):
    '''
    This takes in chiplets and puts them together into a single multi-chip module.
    They will be placed according to their order in chips i.e.
    
    0 - 1 - 2
    3 - 4 - 5
    6 - 7 - 8
    
    Use in combination of create_mcms to return all possible configurations
    
    input:
    rows - number of rows in MCM
    columns - number of columns in MCM
    chips - the provided chiplets sent as a list. there should be the exact amount to fit on the MCM.
    
    returns:
    mcm_dict - dictionary form of simple_qc. Must use simple_qc.from_dict() to use for things.
    '''
    total_chips = rows*columns
    
    if len(chips) != total_chips:
        print('incorrect number of chiplets provided')
        return
    
    
    n_qubits_chip = chips[0].n_qubits
    c_map_chips = chips[0].coupling_map
    
    #find structure of chips
    q_per_row = 3 #this is actually total qubits per row minus one since the extra in the row allocated for in n_right_connect_q
    q_connect = 1
    n_rows = 2
    n_connect_rows = n_rows 
    n_right_connect_q = n_rows # right connection qubits will be placed last
    dim_found = False
    

    while dim_found == False:
    
        if n_qubits_chip > q_per_row*n_rows + q_connect*n_connect_rows + n_right_connect_q:
            q_per_row = q_per_row + 4
            q_connect = q_connect + 1
            n_rows = n_rows + 2
            n_connect_rows = n_rows 
            n_right_connect_q = n_rows
        
        else: 
            dim_found = True
    
    #start setting features for MCM
    mcm_dict = {}
    mcm_dict['layout_type'] = chips[0].layout_type
    mcm_dict['anharmonicity'] = chips[0].anharmonicity
    mcm_dict['n_qubits'] = chips[0].n_qubits*total_chips
    mcm_dict['n_freq'] = chips[0].n_freq
    mcm_dict['collision_free'] = None
    
    
    mcm_dict['freq_pattern'] = {}
    mcm_dict['freq_assignment_ideal'] = {}
    mcm_dict['freq_assignment_actual'] = {}
    
    mcm_dict['coupling_map'] = []
    
    mcm_dict['error_dict'] = {}
    
    #set ideal/actual freq. assignment
    #add intra-chip links to map
    for i in range(total_chips):
        for j in range(n_qubits_chip):
            mcm_dict['freq_pattern'][i*n_qubits_chip+j] = chips[i].freq_pattern[j]
            mcm_dict['freq_assignment_ideal'][i*n_qubits_chip+j] = chips[i].freq_assignment_ideal[j]
            mcm_dict['freq_assignment_actual'][i*n_qubits_chip+j] = chips[i].freq_assignment_actual[j]
            
        for j in chips[i].coupling_map:
            q0 = j[0] + i*n_qubits_chip
            q1 = j[1] + i*n_qubits_chip
            mcm_dict['coupling_map'].append([q0,q1])
            
        for j in chips[i].error_dict:
            q0_q1 = ast.literal_eval(j)
            q0 = q0_q1[0] + i*n_qubits_chip
            q1 = q0_q1[1] + i*n_qubits_chip
            mcm_dict['error_dict'][str([q0,q1])] = chips[i].error_dict[j]
            
    link_list = []  
    
    left_link_qubits = []
    for i in range(n_rows):
        left_link_qubits.append(i*(q_per_row+q_connect))
        
    right_link_qubits = list(range(n_qubits_chip-n_rows,n_qubits_chip))
    
    top_connect_qubits = []
    for i in range(q_connect):
        top_connect_qubits.append(i*4+2)
    bottom_connect_qubits = list(range(n_qubits_chip-n_rows-q_connect,n_qubits_chip-n_rows))
        
    #create add links to coupling map
    for i in range(1,total_chips):
        if i%columns!=0:
            #connect horizontally 
            for j in range(len(left_link_qubits)):
                q0 = left_link_qubits[j] + i*n_qubits_chip
                q1 = right_link_qubits[j] + (i-1)*n_qubits_chip
                mcm_dict['coupling_map'].append([q0,q1])
                mcm_dict['coupling_map'].append([q1,q0])
                
                link_list.append([q0,q1])
                link_list.append([q1,q0])
        
        if i>= columns:
            #connect vertically
            for j in range(len(top_connect_qubits)):
                q0 = top_connect_qubits[j] + i*n_qubits_chip
                q1 = bottom_connect_qubits[j] + (i-columns)*n_qubits_chip
                
                mcm_dict['coupling_map'].append([q0,q1])
                mcm_dict['coupling_map'].append([q1,q0])
                
                link_list.append([q0,q1])
                link_list.append([q1,q0])
    
    
    #add links to error dict
    if use_input_link_vals == False:
        link_errors = return_link_infid()
    else:
        link_errors = np.array(input_link_vals)
    for i in link_list:
        if str(i) not in mcm_dict['error_dict']:
            temp_error = np.random.choice(link_errors)
            mcm_dict['error_dict'][str(i)] = temp_error
            mcm_dict['error_dict'][str([i[1],i[0]])] = temp_error
            
    mcm_dict['link_list'] = link_list
    
    return mcm_dict
            

def create_mcm_dict(max_qubits,chip_sizes):
    #send in upper bound for no. qubits and chip sizes. 
    #this function finds all configurations using max_qubits and chip_sizes.
    #mcm_dict is returned
    mcm_dict = {}
    
    for i in chip_sizes:
        already_in_dict = []
        max_1d = floor(max_qubits/i)
        
        mcm_dict[i] = []
        
        for j in range(2,max_1d+1):
            for k in range(1,j+1):
                temp_qubits = j*k*i
                if temp_qubits > max_qubits:
                    continue
                
                elif temp_qubits not in already_in_dict:
                    already_in_dict.append(temp_qubits)
                    mcm_dict[i].append({'dim':[j,k],'tot_qubits':temp_qubits})
                    
                
    return mcm_dict    

############################################# FUNCTIONS: MAP GENERATORS #########################################
def create_grid_map(n_qubits):
    '''
    attempts to make a device that's a square as possible
    input - n_qubits:number of qubits for a device
    output - new_coupling_map: new coupling map
    '''
    edge_length = ceil(np.sqrt(n_qubits))
    new_coupling_map = []
    
    
    for i in range(1,n_qubits):
        #connect to left qubit
        
        if i % edge_length != 0 and i != 0:
            new_coupling_map.append([i,i-1])
            new_coupling_map.append([i-1,i])
            
        #connect to top qubit
        if i+1 > edge_length:
            new_coupling_map.append([i,i-edge_length])
            new_coupling_map.append([i-edge_length,i])
            
            
    return new_coupling_map

def create_heavysquare_map(n_qubits):
    '''
    tries to fit number of qubits to best square structure

        
    input - n_qubits:number of qubits for device
    output - new_coupling_map: new coupling map for device
    heavy_square_optimal_qubit_counts  = [8,21,40,65,96,133,176,225,280,341,408,481,560,645,
    736,833,936,1045,1160,1281,1408,1541,1680,1825,1976,2133,2296,2465,2640,2821,3008,3201,3400,
    3605,3816,4033,4256,4485,4720,4961,5208]
    '''

    #fix initial values for smallest structure
    q_per_row = 3
    q_connect = 2
    n_rows = 2
    n_connect_rows = n_rows - 1
    dim_found = False
    
    while dim_found == False:
    
        if n_qubits > q_per_row*n_rows + q_connect*n_connect_rows:
            q_per_row = q_per_row + 2
            q_connect = q_connect + 1
            n_rows = n_rows + 1
            n_connect_rows = n_rows - 1
        
        else: 
            dim_found = True
    
    new_coupling_map = []
    left_align_vals_back = []
    left_align_vals_fwd = []
    
    for i in range(0,q_connect):
        temp = q_connect + i*1
        left_align_vals_fwd.append(temp)
        left_align_vals_back.append(q_per_row+q_connect-temp)
        
    
    count_q_row = 0
    count_q_connect = 0
    link_back_qubits = []
    for i in range(q_connect):
        link_back_qubits.append([None,None])
    
    
    for i in range(0,n_qubits):
        
        for item in link_back_qubits:
            if i == item[0]:
                new_coupling_map.append([i,item[1]])
                new_coupling_map.append([item[1],i])    
        
        
        if count_q_row < (q_per_row-1) and i%(q_per_row+q_connect) !=0:
            new_coupling_map.append([i,i-1])
            new_coupling_map.append([i-1,i])
            
            count_q_row = count_q_row + 1 
            
            
        elif count_q_row >= (q_per_row-1):
            new_coupling_map.append([i,i-left_align_vals_back[count_q_connect]])
            new_coupling_map.append([i-left_align_vals_back[count_q_connect],i])
            
            link_back_qubits[count_q_connect] = [left_align_vals_fwd[count_q_connect]+i,i]
            count_q_connect = count_q_connect + 1
                
                
            if count_q_connect == q_connect:
                count_q_row = 0
                count_q_connect = 0
    
    G = graph_from_machine(n_qubits,new_coupling_map)
    
    if nx.is_connected(G) == False:
        S = []
        for c in nx.connected_components(G):
            S.append(c.copy())
        
        if len(S[0]) >= len(S[1]):
            new_coupling_map.append([max(S[0]),min(S[1])])
            new_coupling_map.append([min(S[1]),max(S[0])])
            
        else:
            new_coupling_map.append([min(S[0]),max(S[1])])
            new_coupling_map.append([max(S[1]),min(S[0])])    
    
    return new_coupling_map
        



def create_heavyhex_map(n_qubits):
    '''
    tries to be as 'square' as possible to minimize graph diameter...semi-optimal could be optimized.
    below are the max device qubit counts before maps increase in feature size (i.e. q_per_row++, n_rows++, etc.)
    heavy_hex_optimal_qubit_counts = [25,67,129,211,313,435,577,739,921,1123,1345,1587,1849,2131,2433,2755,3097,3459,3841,4243,4665,5107]

    
    Returns map that matches device if n_qubits == that of a IBMQ machine.
    
    input - n_qubits:number of qubits for device
    output - new_coupling_map: new coupling map for device
    '''
    
    if n_qubits == n_qubits_f_5L:
        return coupling_map_f_5L #returns 'L' structure
        #return coupling_map_f_5T #returns 'T' structure
    
    if n_qubits == n_qubits_f_7:
        return coupling_map_f_7 
    
    if n_qubits == n_qubits_f_16:
        return coupling_map_f_16 
    
    if n_qubits == n_qubits_f:
        return coupling_map_f 
    
    #if n_qubits == n_qubits_h:
    #    return coupling_map_h 
    
    #if n_qubits == n_qubits_e:
    #    return coupling_map_e 

    
    #fix initial values for smallest structure
    q_per_row = 7
    q_connect = 2
    n_rows = 3
    n_connect_rows = n_rows - 1
    dim_found = False
    
    while dim_found == False:
    
        if n_qubits > q_per_row*n_rows + q_connect*n_connect_rows:
            q_per_row = q_per_row + 4
            q_connect = q_connect + 1
            n_rows = n_rows + 2
            n_connect_rows = n_rows - 1
        
        else: 
            dim_found = True
        
    
    
    new_coupling_map = []
    left_align = True
    left_align_vals_back = []
    left_align_vals_fwd = []
    
    for i in range(0,q_connect):
        temp = q_connect + i*3
        left_align_vals_fwd.append(temp)
        left_align_vals_back.append(q_per_row+q_connect-temp)
        
    right_align_vals_back = left_align_vals_fwd.copy()
    right_align_vals_back.reverse()
    
    right_align_vals_fwd = left_align_vals_back.copy()
    right_align_vals_fwd.reverse()
    
    count_q_row = 0
    count_q_connect = 0
    link_back_qubits = []
    for i in range(q_connect):
        link_back_qubits.append([None,None])
    
    
    
    for i in range(0,n_qubits):
        
        for item in link_back_qubits:
            if i == item[0]:
                new_coupling_map.append([i,item[1]])
                new_coupling_map.append([item[1],i])    
        
        
        if count_q_row < (q_per_row-1) and i%(q_per_row+q_connect) !=0:
            new_coupling_map.append([i,i-1])
            new_coupling_map.append([i-1,i])
            
            count_q_row = count_q_row + 1 
            
            
        elif count_q_row >= (q_per_row-1):
            if left_align == True:
                new_coupling_map.append([i,i-left_align_vals_back[count_q_connect]])
                new_coupling_map.append([i-left_align_vals_back[count_q_connect],i])
                
                link_back_qubits[count_q_connect] = [left_align_vals_fwd[count_q_connect]+i,i]
                count_q_connect = count_q_connect + 1
                
            else:
                #align to right
                new_coupling_map.append([i,i-right_align_vals_back[count_q_connect]])
                new_coupling_map.append([i-right_align_vals_back[count_q_connect],i])
                
                link_back_qubits[count_q_connect] = [right_align_vals_fwd[count_q_connect]+i,i]
                count_q_connect = count_q_connect + 1
                
            if count_q_connect == q_connect:
                count_q_row = 0
                count_q_connect = 0
                left_align = not left_align
    
    G = graph_from_machine(n_qubits,new_coupling_map)
    
    if nx.is_connected(G) == False:
        S = []
        for c in nx.connected_components(G):
            S.append(c.copy())
        
        if len(S[0]) >= len(S[1]):
            new_coupling_map.append([max(S[0]),min(S[1])])
            new_coupling_map.append([min(S[1]),max(S[0])])
            
        else:
            new_coupling_map.append([min(S[0]),max(S[1])])
            new_coupling_map.append([max(S[1]),min(S[0])])    
    
    return new_coupling_map
        
def create_heavyhex_map_for_chiplets(n_qubits):
    '''
    produces maps with edges on right and bottom that can be linked to create a chiplet architecture
    that is heavy-hex preserving.
    
    Use ONLY with the optimal counts or else bottom/left connections might be missing!!!
    
    heavy_hex_chiplets_optimal_qubit_counts = [10,40,90,160,250,360,490,640,810,1000,1210,1440,1690,1960,
    2250,2560,2890,3240,3610,4000,4410,4840,5290,5760,6250,6760,7290,7840,8410,9000,9610,10240]


        
    input - n_qubits:number of qubits for device
    output - new_coupling_map: new coupling map for device
    '''

    
    #fix initial values for smallest structure
    q_per_row = 3 #this is actually total qubits per row minus one since the extra in the row allocated for in n_right_connect_q
    q_connect = 1
    n_rows = 2
    n_connect_rows = n_rows 
    n_right_connect_q = n_rows # right connection qubits will be placed last
    dim_found = False
    
    while dim_found == False:
    
        if n_qubits > q_per_row*n_rows + q_connect*n_connect_rows + n_right_connect_q:
            q_per_row = q_per_row + 4
            q_connect = q_connect + 1
            n_rows = n_rows + 2
            n_connect_rows = n_rows 
            n_right_connect_q = n_rows
        
        else: 
            dim_found = True
        
    
    
    new_coupling_map = []
    left_align = True
    left_align_vals_back = []
    left_align_vals_fwd = []
    
    for i in range(0,q_connect):
        temp = q_connect + i*3
        left_align_vals_fwd.append(temp)
        left_align_vals_back.append(q_per_row+q_connect-temp)
        
    right_align_vals_back = left_align_vals_fwd.copy()
    right_align_vals_back.reverse()
    
    right_align_vals_fwd = left_align_vals_back.copy()
    right_align_vals_fwd.reverse()
    
    count_q_row = 0
    count_q_connect = 0
    link_back_qubits = []
    for i in range(q_connect):
        link_back_qubits.append([None,None])
    
    
    
    for i in range(0,n_qubits-n_right_connect_q):
        
        for item in link_back_qubits:
            if i == item[0]:
                new_coupling_map.append([i,item[1]])
                new_coupling_map.append([item[1],i])    
        
        
        if count_q_row < (q_per_row-1) and i%(q_per_row+q_connect) !=0:
            new_coupling_map.append([i,i-1])
            new_coupling_map.append([i-1,i])
            
            count_q_row = count_q_row + 1 
            
            
        elif count_q_row >= (q_per_row-1):
            if left_align == True:
                new_coupling_map.append([i,i-left_align_vals_back[count_q_connect]])
                new_coupling_map.append([i-left_align_vals_back[count_q_connect],i])
                
                link_back_qubits[count_q_connect] = [left_align_vals_fwd[count_q_connect]+i,i]
                count_q_connect = count_q_connect + 1
                
            else:
                #align to right
                new_coupling_map.append([i,i-right_align_vals_back[count_q_connect]])
                new_coupling_map.append([i-right_align_vals_back[count_q_connect],i])
                
                link_back_qubits[count_q_connect] = [right_align_vals_fwd[count_q_connect]+i,i]
                count_q_connect = count_q_connect + 1
                
            if count_q_connect == q_connect:
                count_q_row = 0
                count_q_connect = 0
                left_align = not left_align
    
    #add q_coupled_right_connect
    for i in range(n_rows):
        temp = i*(q_connect+q_per_row) + (q_per_row - 1)
        temp2 = n_qubits-n_right_connect_q+i
        new_coupling_map.append([temp,temp2])
        new_coupling_map.append([temp2,temp])

    
    # makes full graph if incorrect # qubits used
    G = graph_from_machine(n_qubits,new_coupling_map)
    
    if nx.is_connected(G) == False:
        S = []
        for c in nx.connected_components(G):
            S.append(c.copy())
        
        if len(S[0]) >= len(S[1]):
            new_coupling_map.append([max(S[0]),min(S[1])])
            new_coupling_map.append([min(S[1]),max(S[0])])
            
        else:
            new_coupling_map.append([min(S[0]),max(S[1])])
            new_coupling_map.append([max(S[1]),min(S[0])])    
    
    return new_coupling_map

def create_heavyhex_map_for_chiplets_old(n_qubits):
    '''
    produces maps with edges on right and bottom that can be linked to create a chiplet architecture
    that is heavy-hex preserving.
    
    base chiplet is 20 qubits in size
    
    Use ONLY with the optimal counts or else bottom/left connections might be missing!!!
    
    heavy_hex_chiplets_optimal_qubit_counts = [20,60,120,200,300,420,560,720,900,1100,1320,1560,
    1820,2100,2400,2720,3060,3420,3800,4200,4620,5060,5520,6000,6500,7020,7560,8120,8700,9300,9920]

        
    input - n_qubits:number of qubits for device
    output - new_coupling_map: new coupling map for device
    '''

    
    #fix initial values for smallest structure
    q_per_row = 7 #this is actually total qubits per row minus one since the extra in the row allocated for in n_right_connect_q
    q_connect = 2
    n_rows = 2
    n_connect_rows = n_rows 
    n_right_connect_q = n_rows # right connection qubits will be placed last
    dim_found = False
    
    while dim_found == False:
    
        if n_qubits > q_per_row*n_rows + q_connect*n_connect_rows + n_right_connect_q:
            q_per_row = q_per_row + 4
            q_connect = q_connect + 1
            n_rows = n_rows + 2
            n_connect_rows = n_rows 
            n_right_connect_q = n_rows
        
        else: 
            dim_found = True
        
    
    
    new_coupling_map = []
    left_align = True
    left_align_vals_back = []
    left_align_vals_fwd = []
    
    for i in range(0,q_connect):
        temp = q_connect + i*3
        left_align_vals_fwd.append(temp)
        left_align_vals_back.append(q_per_row+q_connect-temp)
        
    right_align_vals_back = left_align_vals_fwd.copy()
    right_align_vals_back.reverse()
    
    right_align_vals_fwd = left_align_vals_back.copy()
    right_align_vals_fwd.reverse()
    
    count_q_row = 0
    count_q_connect = 0
    link_back_qubits = []
    for i in range(q_connect):
        link_back_qubits.append([None,None])
    
    
    
    for i in range(0,n_qubits-n_right_connect_q):
        
        for item in link_back_qubits:
            if i == item[0]:
                new_coupling_map.append([i,item[1]])
                new_coupling_map.append([item[1],i])    
        
        
        if count_q_row < (q_per_row-1) and i%(q_per_row+q_connect) !=0:
            new_coupling_map.append([i,i-1])
            new_coupling_map.append([i-1,i])
            
            count_q_row = count_q_row + 1 
            
            
        elif count_q_row >= (q_per_row-1):
            if left_align == True:
                new_coupling_map.append([i,i-left_align_vals_back[count_q_connect]])
                new_coupling_map.append([i-left_align_vals_back[count_q_connect],i])
                
                link_back_qubits[count_q_connect] = [left_align_vals_fwd[count_q_connect]+i,i]
                count_q_connect = count_q_connect + 1
                
            else:
                #align to right
                new_coupling_map.append([i,i-right_align_vals_back[count_q_connect]])
                new_coupling_map.append([i-right_align_vals_back[count_q_connect],i])
                
                link_back_qubits[count_q_connect] = [right_align_vals_fwd[count_q_connect]+i,i]
                count_q_connect = count_q_connect + 1
                
            if count_q_connect == q_connect:
                count_q_row = 0
                count_q_connect = 0
                left_align = not left_align
    
    #add q_coupled_right_connect
    for i in range(n_rows):
        temp = i*(q_connect+q_per_row) + (q_per_row - 1)
        temp2 = n_qubits-n_right_connect_q+i
        new_coupling_map.append([temp,temp2])
        new_coupling_map.append([temp2,temp])

    
    # makes full graph if incorrect # qubits used
    G = graph_from_machine(n_qubits,new_coupling_map)
    
    if nx.is_connected(G) == False:
        S = []
        for c in nx.connected_components(G):
            S.append(c.copy())
        
        if len(S[0]) >= len(S[1]):
            new_coupling_map.append([max(S[0]),min(S[1])])
            new_coupling_map.append([min(S[1]),max(S[0])])
            
        else:
            new_coupling_map.append([min(S[0]),max(S[1])])
            new_coupling_map.append([max(S[1]),min(S[0])])    
    
    return new_coupling_map
         