#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings; warnings.simplefilter('ignore')


# **TODO**:
# 
# 1. H gate on oracle ws -> DONE
# 2. All invalid fitness values are negative and the valid qubit is 0 -> DONE
# 3. Check how subtractor/adder are working in a separate environment as oracles -> WORKS PERFECT
# 4. Review the entire program and compare it with grover_oracle_adder.ipynb which works perfect!
# 
# **PROBLEM**: Daca pentru fiecare individ calculez Ufit folosint CNOT ( Tofolli) rezulta ca atunci cand controlul e 1 toate cnot-urile vor fi active.
# Exemplu:
# Se presupune ca daca:
# individ = 0 => fit = 0011
# individ = 1 => fit = 0100
# 
# Insa, atunci cand individul e 1, in fit va fi valoarea 0111
# 
# **RESULT**: Problema e rezolvata daca imediat atunci cand bitul e 0, se pune x gate si apoi se clear-uie.

# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, execute, Aer, IBMQ,QuantumRegister,ClassicalRegister
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
import networkx as nx
#from iqx import *

from numpy import flip,array,binary_repr,insert
# Loading your IBM Q account(s)
#provider = IBMQ.load_account()


# In[3]:


NO_OF_NODES = 4
NO_OF_QUBITS_FITNESS = 8
NO_OF_COLORS = 3# 2 bits needed
INVALID_COLORS_LIST = [(1,1)]
NO_OF_QUBITS_PER_COLOR = 2
NO_OF_QUBITS_INDIVIDUAL = 2*NO_OF_NODES
POPULATION_SIZE = 2**NO_OF_QUBITS_INDIVIDUAL #2*NO_OF_NODES#


# In[4]:


#Helper function used for calculating Ufit
def to_binary(value, number_of_bits, lsb=False):
    """
    Function return binary in MSB
    :param value: value that will be converted
    :param number_of_bits:
    :returns: np.array that represents the binary configuration
    >>> to_binary(10,4)
    array([1, 0, 1, 0])
    >>> to_binary(10,4,True)
    array([0, 1, 0, 1])
    """
    if lsb == True:
        return flip(array(list(binary_repr(value, number_of_bits)), dtype=int))
    return array(list(binary_repr(value, number_of_bits)), dtype=int)


# In[5]:


def init_graph(no_of_nodes,edge_probability):
    graph = nx.fast_gnp_random_graph(
        no_of_nodes,
        edge_probability,seed=None,directed=False)
    
    adjacency_matrix=nx.to_numpy_matrix(graph).astype(int)
    

    return adjacency_matrix


#GRAPH = init_graph(NO_OF_NODES,0.7).tolist()
#print(GRAPH)


# In[6]:


GRAPH = [
        #0  1  2  3
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 0],
    ]
    # (3)---(2)
    # |   /  |
    # |  /   |
    # | /    |
    # (0)---(1)
#colors: 0 1 2 1 


# In[7]:


#Grover Iterations subcircuit
def get_grover_iteration_subcircuit():
    fit_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS+1,"fqreg")
    oracle_ws = QuantumRegister(1,"ows")
    grover_circ = QuantumCircuit(fit_qreg,oracle_ws,name ="U$_s$")

    grover_circ.h(fit_qreg)
    grover_circ.x(fit_qreg)

    grover_circ.h(oracle_ws[0])

    grover_circ.mct(list(range(NO_OF_QUBITS_FITNESS+1)), oracle_ws[0])  # multi-controlled-toffoli

    grover_circ.h(oracle_ws[0])


    grover_circ.x(fit_qreg)
    grover_circ.h(fit_qreg)
    grover_circ.h(oracle_ws)

    #grover_circ.draw(output='mpl', plot_barriers=False, filename="grover.png") 
    return grover_circ.to_instruction()


# In[8]:


#Adder subcircuit

#it's FITNESS - VALUE not VALUE - FITNESS AS IT's implemented

def get_adder_instruction():
    def majority(circ,a,b,c):
        circ.cx(c,b)
        circ.cx(c,a)
        circ.ccx(a, b, c)
    def unmaj(circ,a,b,c):
        circ.ccx(a, b, c)
        circ.cx(c, a)
        circ.cx(a, b)
    def adder_4_qubits(p, a0, a1, a2, a3, b0, b1, b2, b3, cin, cout):
        majority(p, cin, b0, a0)
        majority(p, a0, b1, a1)
        majority(p, a1, b2, a2)
        majority(p, a2, b3, a3)
        p.cx(a3, cout)
        unmaj(p, a2, b3, a3)
        unmaj(p, a1, b2, a2)
        unmaj(p, a0, b1, a1)
        unmaj(p, cin, b0, a0)
    a = QuantumRegister(8, "aop")
    b = QuantumRegister(8, "bop")
    c = QuantumRegister(2, "carry")

    add_circ = QuantumCircuit(a, b, c,name="Add")
    adder_4_qubits(add_circ, a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0], c[1])
    adder_4_qubits(add_circ, a[4], a[5], a[6], a[7], b[4], b[5], b[6], b[7], c[1], c[0])
    add8 = add_circ.to_instruction()
    #add_circ.draw(output='mpl', plot_barriers=False, filename="adder.png") 
    return add8


# In[9]:


def check_edges_validity(graph, colors):
    no_of_valid_edges = 0
    for color in colors:
        if color in INVALID_COLORS_LIST:
            return -1
    for i in range(NO_OF_NODES):
        for j in range(i + 1, NO_OF_NODES):
            if graph[i][j]:#daca am legatura
                if colors[j]==colors[i]:
                    continue
                else:
                    no_of_valid_edges +=1

    return no_of_valid_edges


# In[10]:


def pairs_colors(colors_list):
    pairs=list()
    for i in range(0,len(colors_list),2):
        pairs.append((colors_list[i],colors_list[i+1]))
    return pairs


# In[11]:


def get_number_of_edges(graph):
    no_of_edges = 0
    for i in range(NO_OF_NODES):
        for j in range(i + 1, NO_OF_NODES):
            if graph[i][j]:
                no_of_edges +=1    
    return no_of_edges


# In[12]:


def get_ufit_instruction():
    ind_qreg = QuantumRegister(NO_OF_QUBITS_INDIVIDUAL,"ind_qreg")
    fit_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS+1,"fit_qreg")
    qc = QuantumCircuit(ind_qreg,fit_qreg,name="U$_fit$")
    for i in range(0,POPULATION_SIZE):
        individual_binary = to_binary(i, NO_OF_QUBITS_INDIVIDUAL, True)
        #set individual
        for k in range(0,NO_OF_QUBITS_INDIVIDUAL):
            if individual_binary[k] == 0:
                qc.x(ind_qreg[k])
        #create list of colors
        colors = pairs_colors(individual_binary)
        #calculate valid score
        valid_score = check_edges_validity(GRAPH,colors)
        valid_score_binary = to_binary(valid_score,NO_OF_QUBITS_FITNESS,True)
        print("{0}\t{1}\t{2}".format(individual_binary,valid_score,valid_score_binary))
        for k in range(0,NO_OF_QUBITS_FITNESS):
            if valid_score_binary[k]==1:
                qc.mct([ind_qreg[j] for j in range(0,NO_OF_QUBITS_INDIVIDUAL)],fit_qreg[k])
        #if valid_score si greater than 0 then set the valid qubit to 1
        if valid_score > 0:
            qc.mct([ind_qreg[j] for j in range(0,NO_OF_QUBITS_INDIVIDUAL)],fit_qreg[NO_OF_QUBITS_FITNESS])
        #reset individual
        for k in range(0,NO_OF_QUBITS_INDIVIDUAL):
            if individual_binary[k] == 0:
                qc.x(ind_qreg[k])
        qc.barrier()
    #qc.draw(output='mpl', plot_barriers=True, filename="ufit.png") 
    return qc.to_instruction()


# In[13]:


#Oracle subcircuit
def get_oracle_instruction():
    neg_value_reg = QuantumRegister(NO_OF_QUBITS_FITNESS,"nqreg")
    pos_value_reg = QuantumRegister(NO_OF_QUBITS_FITNESS,"pqreg")
    fit_reg = QuantumRegister(NO_OF_QUBITS_FITNESS,"fqreg")
    oracle = QuantumRegister(1,"oqreg")
    carry = QuantumRegister(2,"creg")
    oracle_circ = QuantumCircuit(neg_value_reg,pos_value_reg,fit_reg,oracle,carry,name="O")
    adder_8_qubits_instr = get_adder_instruction()

        
    #a,b,carry
    oracle_circ.append(adder_8_qubits_instr,[  
        neg_value_reg[0],neg_value_reg[1],neg_value_reg[2],neg_value_reg[3],
        neg_value_reg[4],neg_value_reg[5],neg_value_reg[6],neg_value_reg[7],
        fit_reg[0],fit_reg[1],fit_reg[2],fit_reg[3],
        fit_reg[4],fit_reg[5],fit_reg[6],fit_reg[7],
        carry[0],carry[1]
   ])
    
    oracle_circ.h(oracle[0])
    oracle_circ.mct([fit_reg[i] for i in range(0,NO_OF_QUBITS_FITNESS)],oracle[0])
    oracle_circ.h(oracle[0])
    
    
    oracle_circ.reset(carry)

            
    oracle_circ.append(adder_8_qubits_instr,[
        pos_value_reg[0],pos_value_reg[1],pos_value_reg[2],pos_value_reg[3],
        pos_value_reg[4],pos_value_reg[5],pos_value_reg[6],pos_value_reg[7],
        fit_reg[0],fit_reg[1],fit_reg[2],fit_reg[3],
        fit_reg[4],fit_reg[5],fit_reg[6],fit_reg[7],
        carry[0],carry[1]
    ])
    #oracle_circ.draw(output='mpl', plot_barriers=True, filename="oracle.png") 
    return oracle_circ.to_instruction()


# In[14]:


print(GRAPH)
pos_no_of_edges = get_number_of_edges(GRAPH)
neg_no_of_edges = 0 - pos_no_of_edges
print("No of edges:{0}\t{1}".format(pos_no_of_edges,neg_no_of_edges))


# In[15]:


ind_qreg = QuantumRegister(NO_OF_QUBITS_INDIVIDUAL,"ireg")
fit_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS+1,"freg") #8 qubits fitness + 1 valid
carry_qreg = QuantumRegister(2,"qcarry")
oracle = QuantumRegister(1,"oracle")
creg = ClassicalRegister(NO_OF_QUBITS_INDIVIDUAL,"reg")
pos_no_of_edges_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS,"pos_max_qreg")
neg_no_of_edges_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS,"neg_max_qreg")
#create a circuit out of individuals, fitness,carry, oracle
qc = QuantumCircuit(ind_qreg,fit_qreg,carry_qreg,oracle,neg_no_of_edges_qreg,pos_no_of_edges_qreg,creg)



# In[16]:


#create superposition of individuals
qc.h(ind_qreg)
qc.h(oracle)


# In[17]:


#Prepare registers with max number of edges (positive and negative)
neg_value_bin = to_binary(neg_no_of_edges,NO_OF_QUBITS_FITNESS,True)
pos_value_bin = to_binary(pos_no_of_edges,NO_OF_QUBITS_FITNESS,True)
#prepare negative value register
for i in range(0,NO_OF_QUBITS_FITNESS):
    if neg_value_bin[i]==1:
        qc.x(neg_no_of_edges_qreg[i])


#prepare positive value register
for i in range(0,NO_OF_QUBITS_FITNESS):
    if pos_value_bin[i]==1:
        qc.x(pos_no_of_edges_qreg[i])


# In[18]:


#get instructions
ufit_instr = get_ufit_instruction()
oracle_instr = get_oracle_instruction()
grover_iter_inst = get_grover_iteration_subcircuit()


# In[19]:


#qc.barrier()

qc.append(ufit_instr,[ind_qreg[0],ind_qreg[1],ind_qreg[2],ind_qreg[3],
                      ind_qreg[4],ind_qreg[5],ind_qreg[6],ind_qreg[7],
                      fit_qreg[0],fit_qreg[1],fit_qreg[2],fit_qreg[3],
                      fit_qreg[4],fit_qreg[5],fit_qreg[6],fit_qreg[7],
                      fit_qreg[8]
                     ])
qc.barrier()


# In[20]:


qc.append(oracle_instr,[
     neg_no_of_edges_qreg[0],neg_no_of_edges_qreg[1],neg_no_of_edges_qreg[2],neg_no_of_edges_qreg[3],
     neg_no_of_edges_qreg[4],neg_no_of_edges_qreg[5],neg_no_of_edges_qreg[6],neg_no_of_edges_qreg[7],
     pos_no_of_edges_qreg[0],pos_no_of_edges_qreg[1],pos_no_of_edges_qreg[2],pos_no_of_edges_qreg[3],
     pos_no_of_edges_qreg[4],pos_no_of_edges_qreg[5],pos_no_of_edges_qreg[6],pos_no_of_edges_qreg[7],
     fit_qreg[0],fit_qreg[1],fit_qreg[2],fit_qreg[3],
     fit_qreg[4],fit_qreg[5],fit_qreg[6],fit_qreg[7],
     oracle[0],
     carry_qreg[0],carry_qreg[1]
])


qc.barrier()


# In[21]:


# for _ in range(0,1):
qc.append(grover_iter_inst,[
        fit_qreg[0], fit_qreg[1], fit_qreg[2], fit_qreg[3],
        fit_qreg[4], fit_qreg[5], fit_qreg[6], fit_qreg[7], fit_qreg[8],
        oracle[0]
    ])
#qc.draw()


# In[22]:


qc.measure(ind_qreg,creg)
qc.draw()


# In[25]:


qasm_simulator = Aer.get_backend('qasm_simulator')
qasm_simulator.set_options(method="matrix_product_state",
                           #method = "extended_stabilizer",
                           max_parallel_threads = 0,
                           max_parallel_experiments=1,#maximum threads
                           max_parallel_shots=0, # maximum threads
                           max_memory_mb=32000,
                           mps_sample_measure_algorithm="mps_apply_measure"
                          #  extended_stabilizer_measure_sampling = True
                          )
shots = 8
results = execute(qc, backend=qasm_simulator, shots=shots).result()

#plot_histogram(answer)
#print(answer)


# In[24]:


answer = results.get_counts()


# In[ ]:


import operator
print(max(answer.items(), key=operator.itemgetter(1))[0])


# In[ ]:




