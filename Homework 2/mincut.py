# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:20:41 2019

@author: SachaWattel
"""
## Reader for CodeForces :


#import fileinput
#import random
#import math
#
##random.seed(3)
#max_iter = 200000
#
#def read_input():
#    
#    allline = fileinput.input()
#    line = allline.readline().split()
#    n, m = int(line[0]),int(line[1])
#    flag = True
#    conn = []
#    while flag:
#        try:
#            line = allline.readline().split()
#            conn.append([int(line[0])-1,int(line[1])-1])
#        except:
#            flag = False
#    return n,m, conn
#
#n,m, conn = read_input()



john = '''70 103
47 44
5 7
47 5
44 13
7 68
44 7
13 34
68 31
13 68
34 70
31 11
34 31
70 20
11 19
70 11
20 62
19 49
20 19
62 57
49 25
62 49
57 66
25 53
57 25
66 38
53 58
66 53
38 45
58 63
38 58
45 2
63 48
45 63
2 23
48 14
2 48
23 32
14 65
23 14
32 24
65 56
32 65
24 69
56 9
24 56
69 54
9 1
69 9
54 6
1 52
54 1
6 17
52 64
6 52
17 12
64 35
17 64
12 42
35 36
12 35
42 21
36 15
42 36
21 39
15 22
21 15
39 41
22 29
39 22
41 4
29 16
41 29
4 59
16 55
4 16
59 8
55 61
59 55
8 26
61 27
8 61
26 51
27 28
26 27
51 60
28 37
51 28
60 30
37 50
60 37
30 46
50 40
30 50
46 18
40 10
46 40
18 67
10 33
18 10
67 43
33 3
67 33
43 3'''

## Reader for test purposes

import random
import math

split_line = [list(map(lambda x: int(x)-1,line.split(' '))) for line in john.split('\n')]


n, m = list(map(lambda x:x+1,split_line[0]))
conn = split_line[1:]




# Source for Union Find code: https://github.com/coells/100days, with path compression but not rank union
# Straightforward from wikipedia pseudocode

def find(data, i):
    if i != data[i]:
        data[i] = find(data, data[i])
    return data[i]

def union(data, i, j):
    pi, pj = find(data, i), find(data, j)
    if pi != pj:
        data[pi] = pj

def connected(data, i, j):
    return find(data, i) == find(data, j)





def min_cut(n,m, conn, data):
    n_vertices = n
    edges_list = [i for i in range(m)]
    # Create a list of edge in random order from which we pop an edge each iteration
    # Avoids picking an edge twice and acceleration edge counting in second part
    
    random.shuffle(edges_list)
    
    # Number of steps done useful for probability estimation later
    n_step = 0
    
    while n_vertices>2:
        n_step +=1
        u,v = conn[edges_list.pop()]
        
        s1 = find(data,u)
        s2 = find(data,v)
        if s1 != s2:
            n_vertices -=1
            union(data, s1, s2)
    
    # Find the number of cuts to split the last two node groups
    num_cut = 0
    for edge in edges_list:
        u,v = conn[edge]
        
        s1 = find(data,u)
        s2 = find(data,v)

        if s1 != s2:
          num_cut += 1 
    
    return num_cut, n_step
   




# Several runs of Karger algorithm
    
min_cuts = []
#vertices_group = []
n_steps = []
max_iter = 10000
for i in range(max_iter):
    data = [i for i in range(n)]
    mini, n_step = min_cut(n,m, conn, data)
    
    n_steps.append(n_step)
    min_cuts.append(mini)

    

# We hope we at least found min cut once
k = min(min_cuts)

# Find the successful runs
good_karger = [1 if min_cut == k else 0 for min_cut in min_cuts]

# For a graph with m edges, a min cut of k and if Karger algo ran n_step,
# we get a probability of p that we return a specific min-cut
# Adapted from proof of theorem 3 in lecture 11 were we know min cut (k instead of 2)
# and |E| = m
    
def probability_of_specific_min_cuts(m, k, n_step):
    p = 1
    for step in range(n_step):
        p *= float(m-k-step)/(m-step)
    return p

# Compute the probability of them being succesfull
ps = []
for n_step in n_steps:
    ps.append(probability_of_specific_min_cuts(m, k, n_step))


# If we have a probability of p of finding a specific min cut and a total of n_k min cuts
# The function compute the probability of finding any min cut
# Compute : p + p*(1-p) + p*(1-p)^2 + ... + p*(1-p)^(n_k-1) = p* (1 - (1-p)^n_k)/(1 - (1-p))
#    = 1 - (1-p)^n_k
def compute_p(p, n_k):
    return (1-math.pow(1-p,n_k))

# Compute the log likelihood to be exact, for an hypothese of n_k existing min cut
def compute_likelihood(results, ps, n_k):
    likelihood = 0
    for result, p in zip(good_karger, ps):
        if result == 0:
            likelihood += math.log(1-compute_p(p, n_k+1))
        else:
            likelihood += math.log(compute_p(p, n_k+1))
    return likelihood


# COmpute the likelihood for different n_k
   
max_nk = max(5,int(max(list(map(lambda x:1/x, ps)))+0.5)) # For small test case
lhs = []
max_p = max(ps)
for i in range(max_nk):
    if compute_p(max_p, i+1)>0.999: # Heuristic to avoid too much computation and computer precision problem
        break
    lhs.append(compute_likelihood(good_karger,ps,i+1))
    
if n == 2: # For small case
    n_k_good = 1
else: # Find argmax of likelihood
    n_k_good = lhs.index(max(lhs))+1


print(k,n_k_good)


    
    
