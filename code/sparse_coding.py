'''
% Fast sparse coding algorithms
%
%    minimize_B,S   0.5*||X - B*S||^2 + beta*sum(abs(S(:)))
%    subject to   ||B(:,j)||_2 <= l2norm, forall j=1...size(S,1)
% 
% The detail of the algorithm is described in the following paper:
% 'Efficient Sparse Codig Algorithms', Honglak Lee, Alexis Battle, Rajat Raina, Andrew Y. Ng, 
% Advances in Neural Information Processing Systems (NIPS) 19, 2007
%
% Written by Honglak Lee <hllee@cs.stanford.edu>
% Copyright 2007 by Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Y. Ng
%
% options:
% X_total: training set 
% num_bases: number of bases
% beta: sparsity penalty parameter
% sparsity_func: sparsity penalty function ('L1', or 'epsL1')
% epsilon: epsilon for epsilon-L1 sparsity
% num_iters: number of iteration
% batch_size: small-batch size
% fname_save: filename to save
% pars: additional parameters to specify (see the code)
% Binit: initial B matrix
% resample_size: (optional) resample size 
'''
'''
example parameters:
num_bases = 128;
beta = 0.4;
batch_size = 1000;
num_iters = 100;
sparsity_func = 'epsL1' or 'L1'
epsilon = 0.01 or []
'''

import numpy as np
from time import time
import scipy.sparse as ssp
#add path sc2

def sparsecoding(X_total, num_bases, beta, sparsity_func,epsilon, num_iters, fname_save, Binit=None, resample_size=None):

		#check if need to resample X_total

		patchsize = X_total.shape[0]
		num_patches = X_total.shape[1]
		noise_var = 1
		sigma = 1
		VAR_basis = 1
		tol = 0.005

		if Binit is None:
				B = np.random.rand(patchsize,num_bases) - 0.5
				B = B - np.tile(B.mean(axis=0),(B.shape[0],1))
				B = np.dot(B,np.sqrt(np.sum(B*B,axis=0)))
		else:
				B = Binit

		l,m = B.shape

		S_all = np.zeros((m, num_patches))

		if t is None:
				t = 0

				stat = {}
				stat['fobj_avg'] = []
				stat['fresidue_avg'] = []
				stat['fsparsity_avg'] = []
				stat['var_tot'] = []
				stat['svar_tot'] = []
				stat['elapsed_time'] = 0
		else:
				t = len(stat['fobj_avg']-1)

		while t < num_iters
				t += 1
				start_time = time()

				stat['fobj_total'] = 0
				stat['fresidue_total'] = 0
				stat['fsparsity_total'] = 0
				stat['var_tot'] = 0
				stat['svar_tot'] = 0

				#check if need to resmaple X_total

				indperm = np.random.permutation(X.shape[1])

				for batch in range(X.shape[1]/batch_size):
						print "batch: " + str(batch+1)

						batch_idx = np.random.permutation(np.arange(batchsize) + batchsize*(batch-1))
						Xb = X[batch_idx]

						#learn coefficients( conjugate gradient)
						if t==1 or not reuse_coeff:
								if sparsity_func == 'L1' or sparsity_func == 'LARS' or sparsity_func == 'FS':
										S = l1ls_featuresign(B,Xb, beta/sigma*noise_var)
								else:
										S = cgf_fitS_sc2(B, Xb, sparsity_func, noise_var, beta, epsilon, sigma, tol, False, False, False)
								idx = np.where(S == np.nan)
								S[idx] = 0;
								S_all[:,batch_idx] = S
						else:
								if sparsity_func == 'L1' or sparsity_func == 'LARS' or sparsity_func == 'FS':
										S = l1ls_featuresign(B,Xb, beta/sigma*noise_var, S_all[:,batch_idx])
										FS_time = time() - start_time
								else:
										S = cgf_fitS_sc2(B, Xb, sparsity_func, noise_var, beta, epsilon, sigma, tol, False, False, False, S_all[:, batch_idx])
								idx = np.where(S.flatten() == np.nan)
								S[idx] = 0;
								S_all[:,batch_idx] = S

						if sparsity_func == 'L1' or sparsity_func == 'LARS' or sparsity_func == 'FS':
								idx = np.where(S.flatten() != 0)
								sparsity_S = np.sum(S[idx])/len(S.flatten())

						#get objective
						fobj, fresidue, fsparsity = getObjective2(B, S, Xb, sparsity_func, noise_var, beta, sigma, epsilon)

						stat['fobj_total'] = stat['fobj_total'] + fobj
						stat['fresidue_total'] = stat['fresidue_total'] + fresidue
						stat['fsparsity_total'] = stat['fsparsity_total'] + fsparsity
						stat['var_tot'] = stat['var_tot'] + np.sum(sum(S*S)/S.shape[0])

						print "epoch= %d, fobj= %f, fresidue= %f, took %0.2f seconds" % (t, stat['fobj_avg'][t-1], stat['fresidue_avg'][t-1], stat['fsparsity_avg'][t-1], time() - start_time)

						#[TO BE ADDED] save

		return B, S, stat

def l1ls_featuresign(A, Y, gamma, Xinit=None):

		if Xinit is not None:
				use_Xinit = True
		else:
				use_Xinit = False

		Xout = np.zeros(A.shape[1], Y.shape[1])
		AtA = np.dot(A.T,A)
		AtY = np.dot(A.T,Y)

		rankA = min(A.shape[0]-10, A.shape[1]-10)

		for i in range(Y.shape[1]):
				if use_Xinit:
						idx1 = np.where(Xinit[:,i] != 0)
						maxn = min(len(idx1),rankA)
						xinit = np.zeros((Xinit[:,i].shape))
						xinit[idx[0:maxn]] = Xinit[idx1[1:maxn],i]
						Xout[:,i], fobj = ls_featuresign_sub(A, Y[:,,i], Ata, AtY[:,i], gamma, xinit)
				else
						Xout[:,i], fobj = ls_featuresign_sub(A, Y[:,,i], Ata, AtY[:,i], gamma)
		return Xout

def ls_featuresign_sub(A,y, AtA, Aty, gamma, xinit=None):

		L, M = A.shape

		rankA = min(A.shape[0]-10, A.shape[0]-10)

		# Step 1: initialize
		usexinit=False
		if xinit is None:
					xinit = []
					x = ssp.csr(np.zeros((M,1)))
					theta = ssp.csr(np.zeros((M,1)))
					act = ssp.csr(np.zeros((M,1)))
					allowZero = False
		else
					x = ssp.csr(xinit)
					theta = ssp.csr(x)
					act = ssp.csr(np.abs(theta))
					usexinit = True
					allowZero = True

		#[TO BE INSERTED] debug file

		fobj = 0

		ITERMAX=1000
		optimality1=False
		for iter in range(ITERMAX):
				act_indx0 = np.where(act==0)
				grad = np.dot(AtA,ssp.csr(x)) - Aty
				theta = np.sign(x)

				optimality0 = False
				#step 2
				mx, indx = max(np.abs(grad[act_indx0]))

				if mx is None and mx >= gamma and (iter > 1 or not usexinit):
						act[act_indx0[idx]] = 1
						theta[act_indx0[idx]] = -np.sign(grad[act_indx0[idx]])
						usexinit = False
				else
						optimality0 =True
						if optimality1: break

				act_indx1 = np.where(act == 1)

				if len(act_indx1) > rankA:
						print "warning: sparsity penality is too small: too many coefficients are activated!"
						return

				if act_indx1.size == 0:
						if allowZero:
								allowZero = False
								continue
						return


				k=0
				while 1:
						k +=1

						if k > ITERMAX
								print "Maximum number of iterations reached. The solution may not be optimal"
								return

						if act_indx1.size == 0:
									if allowZero:
											allowZero = False
											break
									return

						#step 3
						x, theta, act, act_indx1, optimality1, lsearch, fobj = compute_FS_step(x,A,y,AtA,Aty,theta, act, act_indx1, gamma)

						#step 4
						if optimality1: break;
						if lsearch>0: continue;


				if iter >= ITERMAX:
						print "maximum number of iterations reached. The solution may not be optimal"

				#[add later] if 0 #check optimality

		fobj = fobj_featersign(x,A,y,AtA, Aty, gamma)

		return x, fobj

def compute_FS_step(x,A,y, AtA, Aty, theta, act, act_indx1, gamma):

		x2 = x[act_indx1]
		AtA2 = AtA[act_indx1,act_indx1]
		theta2 = theta[act_indx1]

		#call optimization solver
		x_new,resid,rank,s = np.linalg.lstsq(AtA2, Aty[act-Indx1] - gamma*theta2)

		optimality1= False
		if (np.sign(x_new) == np.sign(x2))
				optimality1=True
				x[act_indx1] = x_new
				fobj = 0
				lsearch = 1
				return

		progress = (0 - x2)*(1/(x_new - x2))
		lsearch = 0
		a = 0.5*sum(np.pow(np.dot(A[:, act_indx1],(x_new-x2)),2))
		b = (np.dot(x2.T,np.dot(AtA2,(x_new-x2))) - np.dot((x_new - x2).T,Aty[act_indx1]))
		fobj_lsearch = gamma*sum(abs(x2))
		sort_lsearch, ix_lsearch = sort([progress.T, 1])
		remove_idx = np.array([])

		for i in range(len(sort_lsearch)):
				t = sort_lsearch[i]
				if t < 0 or t >= 1: continue;
				s_temp = np.pow(x2 + (x_new - x2),t)
				fobj_temp = a*t**2 + b*t + gamma*sum(abs(s_temp))
				if fobj_temp < fobj_lsearch:
					fobj_lsearch = fobj_temp
					lsearch = t
					if t<=1:
							remove_idx = np.concatenate(remove_idx, ix_lsearch[i])
					elif fobj_temp > fobj_lsearch:
							break
					else
							if (sum(x2==0)) == 0:
									lsearch = t;
									fobj_lsearch = fobj_temp
									if t <=1
											remove_idx = np.concatenate(remove_idx, ix_lsearch[i])


		if lsearch > 0
				#update x
				x_new = x2 + (x_new -x2)*lsearch
				x[act_indx1] = x_new
				theta[act_indx1] = np.sign(x_new)

		#if x encounters zeros along the line search, then remove it from the active set
		if lsearch <1 and lsearch > 0:
				remove_idx = np.where(np.abs(x[act_idx1]) < eps)
				x[act_indx1[remove-idx]] = 0

				theta[act_indx1[remove_idx]] = 0
				act[act_indx1[remove_idx]] = 0
				act_indx1[remove_idx] = np.array([])
		fobj_new = 0

		fobj = fobj_new

		return x, theta, act, act_indx1, optimality1, lsearch, fobj

def fobj_features(x,A,y, AtA,Aty, gamma):
		f = 0.5*norm(y-np.dot(A,x))**2
		f = f + gamma*norm(x,1)
				g = np.dot(AtA,x) - Aty
				g = g+ gamma*np.sign(x)

		return f,g
