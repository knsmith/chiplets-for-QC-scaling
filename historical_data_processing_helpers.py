
import numpy as np
import matplotlib.pyplot as plt 
from math import floor

import qiskit
from qiskit import Aer, IBMQ
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, transpile, visualization

from qiskit.quantum_info import Operator

from qiskit.quantum_info import Kraus, SuperOp, Operator, average_gate_fidelity
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.ibmq.job import job_monitor
from qiskit.quantum_info.analysis import hellinger_fidelity
from qiskit.converters import circuit_to_dagdependency
from qiskit.converters import circuit_to_dag


sim = Aer.get_backend('qasm_simulator')
sim_statevect = Aer.get_backend('statevector_simulator')
from qiskit import IBMQ
IBMQ.load_account()

provider=IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='phy147')


import datetime 
import pickle
from copy import deepcopy


def load_historical_data(processors,gather_date):
    for item in processors:
        file_name = "./" + gather_date + "_machine_props/" + item + "_" + gather_date + "_to_release_properties"
        file = open(file_name, "rb")
        processors[item] = pickle.load(file)
    return processors
        

def remove_duplicate_records(processor_dict):
    found_records = []
    filtered_processor_dict = {}
    for i in processor_dict:
        if processor_dict[i]['last_update_date'] not in found_records:
            found_records.append(processor_dict[i]['last_update_date'])
            filtered_processor_dict[i] = processor_dict[i]
        
    
    return filtered_processor_dict


def find_longest_measurement_gate_pair(record_in):
    '''
    in ns but convert to us
    '''
    
    qubit_tracker = {}
    for i in range(len(record_in['qubits'])):
        qubit_tracker[i] = {}
        for j in record_in['qubits'][i]:
            if j['name'] == 'readout_length':
                qubit_tracker[i]['readout_length'] = j['value']
    
    for i in qubit_tracker:
        longest_gate = 0
        for j in record_in['gates']:
            if i in j['qubits']:
                for k in j['parameters']:
                    if k['name'] == 'gate_length':
                        if k['value'] > longest_gate:
                            longest_gate =  k['value']
        qubit_tracker[i]['longest_gate'] = longest_gate
    
    longest_measurement_gate = 0
    
    for i in qubit_tracker:
        if qubit_tracker[i]['longest_gate'] + qubit_tracker[i]['readout_length'] > longest_measurement_gate:
            longest_measurement_gate = qubit_tracker[i]['longest_gate'] + qubit_tracker[i]['readout_length']
    
    return longest_measurement_gate/1000


def isolate_1q_property(n_qubits,processors,what_to_analyze,verbose = True):
    '''
    returns dictionary q_prop, q_prop_mean, and unit
    
    look for 'T1', 'T2', 'frequency', 'anharmonicity', 'readout_error', 'prob_meas0_prep1', 'prob_meas1_prep0' , 'readout_length'
    
    NOTE: not all records contain all these features (earlier records don't include as much....)
    
    duplicate and incomplete records are removed from returned set 
    '''
    #n_qubits = 27
    q_prop = {}
    q_prop_mean = {}
    unit = None
    labels = {}
    


    for i in processors:
        
        new_processor = remove_duplicate_records(processors[i][i])
        
        q_prop[i] = {}
        q_prop_mean[i] = []
        labels[i] = []
        for j in range(n_qubits):
            q_prop[i][j] = []
        for j in new_processor.keys():
            #print(f'hello {j} {i}')
            
            #try:
            
            if len(new_processor[j]['qubits']) != n_qubits:
                q_w_data = len(new_processor[j]['qubits'])
                if verbose == True:
                    print(f'Only {q_w_data} qubits have {what_to_analyze} on {i} on {j}.')
                continue
            
            #only record property if every qubit has it
            
            temp_prop = []
            incomplete_record = False
            for k in range(len(new_processor[j]['qubits'])):
                prop_index = None
                for m in range(len(new_processor[j]['qubits'][k])):
                    if what_to_analyze == new_processor[j]['qubits'][k][m]['name']:
                        prop_index = m
    
                if prop_index == None:
                    if verbose == True:
                        print(f'{what_to_analyze} unknown on {i} qubit {k} on {j}.')
                    incomplete_record = True
                    break
                
                #q_prop[i][k].append(new_processor[j]['qubits'][k][prop_index]['value'])
                temp_prop.append(new_processor[j]['qubits'][k][prop_index]['value'])
                if unit == None:
                    unit = new_processor[j]['qubits'][k][prop_index]['unit']
                
            if incomplete_record == False:
                for k in range(len(temp_prop)):
                    q_prop[i][k].append(temp_prop[k])
                
                labels[i].append(j)
            #except:
                #continue
        for j in q_prop[i]:
            q_prop_mean[i].append(np.mean(np.array(q_prop[i][j])))
            
    return q_prop, q_prop_mean, unit, labels

def isolate_cx_error(processors, coupling_map, threshold=1, verbose=True):
    '''
    returns dictionary cx_error, cx_error_mean
    
    duplicate, at/above threshold, and incomplete records are removed from returned set 
    
    '''
    cx_error = {}
    cx_error_mean = {}
    
    labels = {}
    
    for i in processors:
        
        new_processor = remove_duplicate_records(processors[i][i])
        
        cx_error[i] = {}
        cx_error_mean[i] = {}
        labels[i] = []
        
        for j in coupling_map:
            cx_error[i][str(j)] = []

        
        for j in new_processor.keys():
            
            #remove all incomplete samples!!
            temp = []
            for k in new_processor[j]['gates']:

                #determine if CX gate
                if len(k['qubits'])==2:
                    
                    temp.append(k['qubits'])
            str_c_map = set(str(k) for k in coupling_map)
            str_temp = set(str(k) for k in temp)
            if str_c_map == str_temp:
                
                #remove all records that contain a 1 
                temp_error_vals = {}
                threshold_found = False
                
                for k in new_processor[j]['gates']:
                    if len(k['qubits'])==2:
                    
                        for m in k['parameters']:
                            if m['name'] == 'gate_error':
                                
                                temp_error_vals[str(k['qubits'])] = m['value']
                                if m['value'] >= threshold:
                                    threshold_found = True
                                    if verbose == True:
                                        print(f'Threshold exceeded {i} on {j} for '+str(k['qubits'])+' : '+str(m['value']))
                
                if threshold_found == False:
                    for k in temp_error_vals.keys():
                        
                        cx_error[i][k].append(temp_error_vals[k])
                    
                    labels[i].append(j)
                    
            else:
                if verbose==True:
                    print(f'Record mismatch with coupling map for {i} on {j}. {len(coupling_map)} items in map, {len(temp)} found.')
    
                    unique_temp = str_temp-str_c_map
                    unique_map = str_c_map-str_temp
                    print(f'    not connection: {unique_temp}')
                    print(f'    missing: {unique_map}')
                
                            
        for j in cx_error[i]:
            cx_error_mean[i][j] = np.mean(np.array(cx_error[i][j]))
            
    return cx_error, cx_error_mean, labels


def clean_data(processors,prop_setting='cx',coupling_map = None , threshold = 1, one_q_prop = None, n_qubits=None, verbose = True):
    '''
    Returns dict that is the same as what's input (how the historical data is initially loaded). 
    
    All duplicates, incorrect, and partial records are removed.
    
    for T1/T2, records with coherence times too short for real use are removed
    
    Call EACH TIME you want to include a feature.
    
    If prop_setting == "1q_prop", one_q_prop must be a str in the set of:
    {'T1', 'T2', 'frequency', 'anharmonicity', 'readout_error', 
    'prob_meas0_prep1', 'prob_meas1_prep0' , 'readout_length'}
    
    coupling_map and threshold are used for 2q gate properties
    
    n_qubits used for 1q properties
    
    
    '''
    
    if prop_setting == 'cx' and coupling_map != None :
        #do the cx filtering
        
        new_data_dict = {}
        
        for i in processors:
            
            new_processor = remove_duplicate_records(processors[i][i])
            new_data_dict[i] = {i:{}}
            
            
            for j in new_processor.keys():
                temp = []
                for k in new_processor[j]['gates']:
                    
                    #determine if cx gate
                    if len(k['qubits']) == 2:
                        temp.append(k['qubits'])
                        
                str_c_map = set(str(k) for k in coupling_map)
                str_temp = set(str(k) for k in temp)
                if str_c_map == str_temp:
                
                    #remove all records that contain a 1 
                    #temp_error_vals = {}
                    threshold_found = False
                    
                    for k in new_processor[j]['gates']:
                        if len(k['qubits'])==2:
                        
                            for m in k['parameters']:
                                if m['name'] == 'gate_error':
                                    
                                    #temp_error_vals[str(k['qubits'])] = m['value']
                                    if m['value'] >= threshold:
                                        threshold_found = True
                                        if verbose == True:
                                            print(f'Threshold exceeded {i} on {j} for '+str(k['qubits'])+' : '+str(m['value']))
                    
                    if threshold_found == False:   
                        new_data_dict[i][i][j] = deepcopy(new_processor[j])
                        
                else:
                    if verbose==True:
                        print(f'Record mismatch with coupling map for {i} on {j}. {len(coupling_map)} items in map, {len(temp)} found.')
        
                        unique_temp = str_temp-str_c_map
                        unique_map = str_c_map-str_temp
                        print(f'    not connection: {unique_temp}')
                        print(f'    missing: {unique_map}')
                    
                                
            
                
        return new_data_dict
                        
                    
        
    elif prop_setting == '1q_prop' and one_q_prop != None and n_qubits!= None:
        #do the 1q properties filtering
        
        new_data_dict = {}
        
        for i in processors:
            
            new_processor = remove_duplicate_records(processors[i][i])
            new_data_dict[i] = {i:{}}
            

            for j in new_processor.keys():
                                
                if len(new_processor[j]['qubits']) != n_qubits:
                    q_w_data = len(new_processor[j]['qubits'])
                    if verbose == True:
                        print(f'Only {q_w_data} qubits have {one_q_prop} on {i} on {j}.')
                    continue
                
                #only record property if every qubit has it
                

                if one_q_prop == 'T1' or one_q_prop == 'T2':
                    longest_measurement_gate = find_longest_measurement_gate_pair(new_processor[j])
                
                incomplete_record = False
                for k in range(len(new_processor[j]['qubits'])):
                    prop_index = None
                    for m in range(len(new_processor[j]['qubits'][k])):
                        if one_q_prop == new_processor[j]['qubits'][k][m]['name']:
                            prop_index = m
        
                    if prop_index == None:
                        if verbose == True:
                            print(f'{one_q_prop} unknown on {i} qubit {k} on {j}.')
                        incomplete_record = True
                        break
                    
                    # only include T1/T2 records if they are greater than longest operation+measurement    
                    if one_q_prop == 'T1' or one_q_prop == 'T2':
                        if new_processor[j]['qubits'][k][prop_index]['value'] < longest_measurement_gate: 
                            if verbose == True:
                                print(f'{one_q_prop} smaller than longest gate/measurement pair on {i} qubit {k} on {j}.')
                            incomplete_record = True
                            break

                if incomplete_record == False:

                    
                    new_data_dict[i][i][j] = deepcopy(new_processor[j])

                
        return new_data_dict

        
        
        
        
    else:
        print('ERROR: Cannot filter data. Check your input parameters.')
        return {}