#! python3

import numpy as np

class K_Reciprocal_Encoding():

    def __init__(self, distance_function, k1=20):
        self.k1 = k1
        self.R_star_threshold = 2./3.

        ''' Distance Function should be the form of the following descriptions:
        # Input: (A, B)
            - A: [n_A, n_feature]
            - B: [n_B, n_feature]
        # Output: D
            - D: [n_A, n_B]
                - D[i, j] = the distance between A[i] and B[j]
        '''
        self.distance_function = distance_function

    def compute_N(self, rank, k):
        return [ rank[i, :k] for i in range(rank.shape[0]) ]

    def compute_R(self, N_forward, N_backward=None):
        if N_backward is None:
            N_backward = N_forward
        R = []
        for p in range(len(N_forward)):
            R.append( [ x for x in N_forward[p] if p in N_backward[x] ] )
        return R

    def compute_R_star(self, R, dist=None):
        R_star = []
        for p in range(len(R)):
            R_star.append(R[p])
            for q in range(len(R[p])):
                n_1 = len(self.GG_R_2[q])
                n_2 = len(np.intersect1d(R[p], self.GG_R_2[q]))
                if (dist is not None) and (dist[p, q] < self.GG_dist[ q, self.GG_rank[q, int(self.k1/2)-1] ]):
                    # check if probe itself should be in GG_R_2[q]
                    n_1 += 1
                    n_2 += 1
                if (n_2/n_1) > self.R_star_threshold:
                    R_star[p] += self.GG_R_2[q]
            R_star[p] = np.unique(R_star[p])
        return R_star

    def compute_V(self, R, sim):
        V = np.zeros(sim.shape)
        for i in range(V.shape[0]):
            if len(R[i]) == 0:
                continue
            V[ i, R[i] ] = sim[ i, R[i] ]
            s = np.sum(V[i])
            if s > 0:
                V[i] /= s
        return V

    def fit(self, gallery_X, verbose=True):
        self.gallery_X = gallery_X
        self.n_gallery, self.n_feature = gallery_X.shape

        print('[*] Compute the distance matrix within Gallery')
        self.GG_dist = self.distance_function(gallery_X, gallery_X)

        print('[*] Compute the similarity matrix within Gallery')
        GG_sim  = np.exp( -self.GG_dist )       

        print('[*] Compute nearest neighbors within Gallery')
        self.GG_rank = np.argsort(self.GG_dist, axis=1)
        
        GG_N_1 = self.compute_N(self.GG_rank, self.k1)
        GG_N_2 = self.compute_N(self.GG_rank, int(self.k1/2))

        print('[*] Compute reciprocal nearest neighbors within Gallery')
        self.GG_R_1 = self.compute_R(GG_N_1)
        self.GG_R_2 = self.compute_R(GG_N_2)
        del GG_N_1, GG_N_2

        print('[*] Compute R*')
        self.GG_R_star = self.compute_R_star(self.GG_R_1)
        
        print('[*] Compute V within Gallery')
        self.GG_V = self.compute_V(self.GG_R_star, GG_sim)
        
    def metric(self, query_X, k2=6):
        print('[*] Compute the distance matrix between Query(Probe) and Gallery')
        QG_dist = self.distance_function(query_X, self.gallery_X)

        n_query = query_X.shape[0]
        
        print('[*] Compute the similarity matrix between Query(Probe) and Gallery')
        QG_sim  = np.exp( -QG_dist )
        
        print('[*] Compute nearest neighbors between Query(Probe) and Gallery')
        '''
        probe's k-nearest neighbors among gallery, excluding probe itself
        '''
        QG_rank = np.argsort(QG_dist, axis=1)
        
        QG_N = self.compute_N(QG_rank, self.k1)
        GQ_N = []
        GQ_dist = QG_dist.T
        for i in range(self.n_gallery):
            GQ_N.append( np.where(GQ_dist[i] < self.GG_dist[i, self.GG_rank[i, self.k1] ])[0] )

        print('[*] Compute reciprocal nearest neighbors between Query(Probe) and Gallery')
        QG_R = self.compute_R(QG_N, GQ_N)
        del QG_N, GQ_N

        print('[*] Compute R*')
        QG_R_star = self.compute_R_star(QG_R, QG_dist)

        print('[*] Compute V between Query(Probe) and Gallery')
        QG_V = self.compute_V(QG_R_star, QG_sim)
        
        print('[*] Local Query Expansion')
        QG_V_local = np.zeros([n_query, self.n_gallery])
        for i in range(n_query):
            QG_V_local[i] = (QG_V[i] + np.sum(self.GG_V[ QG_rank[i, :k2-1] ], axis=0)) / (k2)
        del QG_V
        
        print('[*] Compute the Jaccard distance matrix between Query(Probe) and Gallery')
        '''
        Since we have normalized V so that each row's sum = 1
        we get Intersect + Union = 2
        so that we can replace "Intersect / Union" with "Intersect / (2 - Intersect)"
        '''
        intersect = np.zeros([n_query, self.n_gallery])
        for q in range(n_query):
            print(q, end='\r')
            QG_V_nonzero = np.where(QG_V_local[q, :] > 0)[0]
            for g in range(self.n_gallery):     
                intersect[q][g] = np.sum( np.minimum(QG_V_local[q, QG_V_nonzero], self.GG_V[g, QG_V_nonzero]) )
            
        Jaccard_dist = 1. - np.divide(intersect, 2. - intersect)

        return Jaccard_dist
