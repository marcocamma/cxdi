from scipy import stats
from scipy.special import gamma
from math import factorial

def _probability(x,mu,beta):
    up1 = (beta*mu)**x
    up2 = (1+beta*mu)**(-(1+beta*x)/beta)
    up3 = gamma(x+1/beta)
    down1 = factorial(x)
    down2 = gamma(1/beta)
    up = up1*up2*up3
    down = down1*down2
    return up/down

def _probability2(x,mu,beta):
    # from doi:10.1107/S1600577514015847
    M = 1/beta
    f1 = gamma(x+M)/gamma(x+1)/gamma(M)
    f2 = (M/(mu+M))**M
    f3 = (mu/(M+mu))**x
    return f1*f2*f3



def probability(x,mu,beta):
    # mu = average counts/pix
    # beta is contrast (0-1)
    if np.isclose(beta,1):
        beta = 0.99999 # avoid numerical issues
    elif beta > 1:
        raise ValueError("beta (contrast) should be <= 1")
    if isinstance(x,int):
        return _probability(x,mu,beta)
    else:
        return np.asarray( [_probability(xi,mu,beta) for xi in x] )

def fit_histogram(bins,counts,plot=True):
    bins = np.asarray(bins)
    counts = np.asarray(counts)
    import lmfit
    model = lmfit.Model(probability)
    # make sure counts are normalised
    counts = counts/counts.sum()
    avg = sum(bins*counts)
    pars = model.make_params(mu=avg,beta=0.7)
    r= model.fit(counts,x=bins,params=pars)
    if plot:
        r.plot()
    print(r.best_values())
    return r

def probability_random_vars(mu,beta,size=1):
    """see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html"""
    var = mu+beta*mu**2 # variance
    p = mu/var
    n = mu**2/(var-mu)
    return stats.nbinom.rvs(n, p, size=size)
