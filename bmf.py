import numpy as np
from copy import copy

T1 = 100
T2 = 5

class sampler():
	def __init__(self, R, mask, k=10, beta=1, mu_u0=0, mu_v0=0, lambda_u0=1, lambda_v0=1):
		self.R = R
		self.mask = mask
		self.m = R.shape[0]
		self.n = R.shape[1]
		self.k = 5
		self.beta = 1
		self.mu_u0 = mu_u0
		self.mu_v0 = mu_v0
		self.lambda_u0 = lambda_u0
		self.lambda_v0 = lambda_v0
		
		self.U = np.random.randn(self.m, self.k)
		self.V = np.random.randn(self.n, self.k)
		# self.S_ut = None
		# self.mu_ut = None
		# self.S_vt = None
		# self.mu_vt = None

	def sample_U(self):
		for i in range(self.m):
			m = np.zeros(self.k)
			s = np.zeros([self.k,self.k])
			for j in range(self.n):
				if self.mask[i,j]:
					s = s + np.outer(self.V[j,:],self.V[j,:])
					m = m + self.R[i,j]*self.V[j,:]
			S_ut = np.linalg.inv(np.add(s, self.lambda_u0*np.identity(self.k)))
			mu_ut = np.matmul(S_ut, self.beta*np.transpose(m))
			self.U[i,:] = np.random.multivariate_normal(mu_ut, S_ut)

	def sample_V(self):
		for i in range(self.n):
			m = np.zeros(self.k)
			s = np.zeros([self.k,self.k])
			for j in range(self.m):
				if self.mask[j,i]:
					s = s + np.outer(self.U[j,:],self.U[j,:])
					m = m + self.R[j,i]*self.U[j,:]
			S_vt = np.linalg.inv(np.add(s, self.lambda_v0*np.identity(self.k)))
			mu_vt = np.matmul(S_vt, self.beta*np.transpose(m))
			self.V[i,:] = np.random.multivariate_normal(mu_vt, S_vt)


burnout = 500
gap = 10
numsamples = 1000



a = np.random.randint(2,6, [3,4])
mask = np.random.randint(0,2,a.shape)
print(a,mask)
mean = np.zeros(a.shape)
sampler = sampler(a, mask=mask)

for i in range(burnout):
	sampler.sample_U()
	sampler.sample_V()

for i in range(numsamples):
	sampler.sample_U()
	sampler.sample_V()
	a = a + np.matmul(sampler.U, np.transpose(sampler.V))

	for j in range(gap-1):
		sampler.sample_U()
		sampler.sample_V()


print(a/numsamples)