import numpy as np
from scipy.stats import norm

RNG = np.random.default_rng()



class PricerMC:

    def __init__(self,N,M,T,sigma,r ):
        self.N = N
        self.M = M
        self.T = T
        self.sigma = sigma
        self.r = r


    def monte_carlo(self,S0,N,M,T,sigma,r):
        St=S0
        for i in range(N):
            Wt = RNG.normal(0,1,M)*np.sqrt(T/N)
            St = St*np.exp( (r - sigma**2/2)*(T/N) + sigma*Wt )
        return St
    
    def monte_carlo_path_dep(self,S0,N,M,T,sigma,r):
        ST = np.log(S0) +  np.cumsum(((r  - sigma**2/2)*(T/N) +sigma*np.sqrt(T/N) * np.random.normal(size=(N,M))),axis=0)
        return np.exp(ST)

    def prix_MC(self, S, sigma):
        payoffcall = np.where(S-K > 0, S-K  ,0)
        payoffput = np.where(K-S > 0, K-S, 0)
        return np.mean(payoffcall)*np.exp(-r*T), np.mean(payoffput)*np.exp(-r*T)

    def prix_BS(self,S,sigma):
        d1 = (np.log(S/K) + r - (sigma**2)*T/2. )/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)#,  K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1) 

    def greeks_BS(self,S):
        d1 = (np.log(S/K) + r - (sigma**2)*T/2. )/(sigma*np.sqrt(T))
        delta_call = norm.cdf(d1)
        delta_put = delta_call - 1
        gamma = norm.pdf(d1)* S*sigma*np.sqrt(T)
        return delta_call, delta_put, gamma
    
    def greeks_FD_MC(self, S):
        payoffcallh = np.where(S-K > 0, S-K  ,0)
        payoffcallminush = np.where(S-(T/N) -K > 0, S-(T/N)-K  ,0)
        payoffcallplush = np.where(S+(T/N) -K > 0, S+(T/N)-K  ,0)

        prixcallh = np.mean(payoffcallh)*np.exp(-r*T)
        prixcallplush = np.mean(payoffcallplush)*np.exp(-r*T)
        prixcallminush = np.mean(payoffcallminush)*np.exp(-r*T)

        deltacallFD = (prixcallplush - prixcallminush)/(2*(T/N))
        gammacallFD = (prixcallplush + prixcallminush -2*prixcallh)/((T/N)**2)



        return deltacallFD, gammacallFD





def dichotomie(f,a,b,seuil):
    delta = 1
    while delta > seuil:
        mid = (a+b)/2
        delta = abs(b-a)
        if f(mid) == 0:
            return mid
        elif f(a)*f(mid)>0:
            a = mid
        else:
            b = mid
    return a,b, delta


def newton(f,a,seuil):
    delta = 1
    while delta > seuil:

        def df(a):
            dx = 0.00001
            return (f(a + dx) - f(a))/dx
        x = a - f(a)/df(a) 
        delta = abs(x-a)
        a = x
    return x,delta




if __name__ == '__main__':

    T = 1
    N = 252
    M = 1000
    sigma = 0.2
    r = 0.01
    S0 =100
    K = S0
    True_price = 10


    BS_Call = PricerMC(N,M,T,sigma,r)
    Prix_BS_Call = BS_Call.prix_BS(S0,sigma)

    BS_Call_vol = lambda sigma: BS_Call.prix_BS(S0,sigma) - True_price 
    Init = np.sqrt(abs(np.log(S0/(K*np.exp(-r*T))))*2/T) # Valeur initiale Optimale



    print('Dichotomie :', dichotomie( BS_Call_vol,0.0000001,1,0.00001) )
    print('Newton-Raphson : ', newton(BS_Call_vol, Init,0.00001))
