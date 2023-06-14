import numpy as np 
import math

class Canonical: 
	def __init__(self, mean=np.zeros(10), sigma=0.05, lamb=5, cm=1 ) -> None:
		self._mean=mean
		self._sigma=sigma
		self.lamb=lamb
		# weights *
		# cma-agent
		n_dim = len(mean)
		mu = lamb // 4
		# mu=4
		weights_prime = np.array([(np.log((lamb + 1) / 2) - np.log(i + 1))for i in range(lamb)])
		mu_eff = (np.sum(weights_prime[:mu]) ** 2) / np.sum(weights_prime[:mu] ** 2)
		mu_eff_minus = (np.sum(weights_prime[mu:]) ** 2) / np.sum(weights_prime[mu:] ** 2)
		# learning rate for the rank-one update
		alpha_cov = 2
		c1 = alpha_cov / ((n_dim + 1.3) ** 2 + mu_eff)
		# learning rate for the rank-μ update
		cmu = min(1 - c1 - 1e-8,  alpha_cov* (mu_eff - 2 + 1 / mu_eff)/ ((n_dim + 2) ** 2 + alpha_cov * mu_eff / 2),)
		min_alpha = min(1 + c1 / cmu, 1 + (2 * mu_eff_minus) / (mu_eff + 2),  (1 - c1 - cmu) / (n_dim * cmu), )
		positive_sum = np.sum(weights_prime[weights_prime > 0])
		negative_sum = np.sum(np.abs(weights_prime[weights_prime < 0]))
		self._weights = np.where(weights_prime >= 0,1 / positive_sum * weights_prime,min_alpha / negative_sum * weights_prime,)
		# cm & mu
		self._cm=cm
		self._mu=mu
		# Customization 
		self.fmean=0
		self.alpha=1e-1
		self.proportion=1/2

	def ask(self):
		return self._mean+np.random.normal(0, self._sigma,self._mean.shape)


	def tell(self,solutions):
		# sort
		solutions.sort(key=lambda s: s[1])
		x_k = np.array([s[0] for s in solutions])  # ~ N(m, σ^2 C)
		y_k = (x_k - self._mean) / self._sigma  # ~ N(0, C)
		# Selection and recombination
		y_w = np.sum(y_k[: self._mu].T * self._weights[: self._mu], axis=1)  # eq.41
		self._mean += float(self._cm) * self._sigma * y_w

def quadratic(x1,x2):
	return (x1 + 3) ** 2+(x2 - 6) ** 2 

if __name__=="__main__":
	optimizer = Canonical(mean=np.array([0,0], dtype='float64'), sigma=0.5,lamb=20)
	for generation in range(20):
		solutions = []
		for _ in range(optimizer.lamb):
			x = optimizer.ask()
			value = quadratic(x[0],x[1])
			solutions.append((x, value))
			# print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
		optimizer.tell(solutions)
		print("mean : ",optimizer._mean)
