import numpy as np
from copy import copy

T1 = 100
T2 = 5


class sampler():
	def __init__(self, R=None, mask=None, k=10, beta=1, mu_u0=0, mu_v0=0, lambda_u0=1, lambda_v0=1, rand_flag=True):
		if rand_flag == True:
			self.random_data()
		else:
			self.load_data()

		self.m = self.R.shape[0]
		self.n = self.R.shape[1]
		self.k = 10
		self.beta = 1
		self.mu_u0 = mu_u0
		self.mu_v0 = mu_v0
		self.lambda_u0 = lambda_u0
		self.lambda_v0 = lambda_v0
		
		self.U = np.random.randn(self.m, self.k)
		self.V = np.random.randn(self.n, self.k)

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

	def load_data(self):
		data = np.loadtxt('ml-100k/trainA', delimiter='\t')
		data[:,2] = data[:,2]*2
		data = data.astype(int)
		users = np.asarray(np.unique(data[:,0]), dtype=int)
		movies = np.asarray(np.unique(data[:,1]), dtype=int)
		self.m = users.shape[0]
		self.n = movies.shape[0]

		users_dict = {users[i]:i for i in range(self.m)}
		movies_dict = {movies[i]:i for i in range(self.n)}
		self.mask = np.zeros((self.m,self.n), dtype=bool)
		self.test_mask = np.zeros((self.m,self.n), dtype=bool)
		self.R = np.zeros((self.m,self.n))

		for i in range(data.shape[0]):
			self.R[users_dict[data[i,0]], movies_dict[data[i,1]]] = data[i,2]/2
			self.mask[users_dict[data[i,0]], movies_dict[data[i,1]]] = 1

		test = np.loadtxt('ml-100k/testA', delimiter='\t')
		test[:,2] = test[:,2]*2
		test = test.astype(int)

		for i in range(test.shape[0]):
			if test[i,1] not in movies_dict.keys():
				continue
			if self.R[users_dict[test[i,0]], movies_dict[test[i,1]]] != 0:
				print(i)
			self.R[users_dict[test[i,0]], movies_dict[test[i,1]]] = test[i,2]/2
			self.test_mask[users_dict[test[i,0]], movies_dict[test[i,1]]] = 1

	def random_data(self):
		self.R = np.random.randint(1,6, [4,3])
		self.mask = np.random.randint(0,2,self.R.shape)
		print(self.R, self.mask)
		self.test_mask = np.bitwise_xor(np.ones(self.mask.shape, dtype=bool),self.mask)

	def find_error(self, pred_mean):
		test = np.multiply(self.test_mask, self.R)
		pred = np.multiply(self.test_mask, pred_mean)
		return np.linalg.norm(np.subtract(test,pred))/np.sqrt(np.sum(self.test_mask))


burnout = 500
gap = 5
numsamples = 1000
random = False

sampler = sampler(rand_flag=random)

# mean = np.load("pred20.npy")
# mean[np.where(mean>=5)] = 5
# mean[np.where(mean<=0)] = 0
# print(sampler.find_error(mean))

mean = np.zeros([sampler.m, sampler.n])
for i in range(burnout):
	sampler.sample_U()
	sampler.sample_V()

for i in range(1,numsamples+1):
	sampler.sample_U()
	sampler.sample_V()
	mean = mean*(i-1)/i + np.matmul(sampler.U, np.transpose(sampler.V))/i

	for j in range(gap-1):
		sampler.sample_U()
		sampler.sample_V()

	if i%50 == 0:
		print(i, sampler.find_error(mean))
		np.save('pred.npy', mean)

if random == True:
	print(mean)
else:
	np.save('pred.npy', mean)