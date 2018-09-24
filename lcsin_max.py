
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from astropy import units as u
from scipy.optimize import curve_fit, minimize_scalar

#region gera curva de luz com as características das modulações de FO Aqr
def sinmod(tt, aa, pp, tt0, ey, rdnseed=None):
    import datetime

    if rdnseed is None: # gera seed com os microssegundos do tempo atual
        ms = datetime.datetime.now()
        rdnseed = ms.microsecond
    np.random.seed(rdnseed) # seed para o gerador de números aletórios

    sinterm = 2 * np.pi / pp
    ys = aa * np.sin(sinterm * tt - sinterm * tt0)
    es = np.random.normal(0, ey, tt.shape)
    return ys + es, es

tt = np.arange(0, 0.1, 0.000116) # run de 0.1 d (= 2.4 h) com texp = 10 s
Pref = 1254 * u.s # período de rotação da primária (modulação) em segundos
Aref = 0.2 * u.mag # amplitude da modulação em magnitude
pp = Pref.to(u.day).value # converte Pref para dias
aa = Aref.value
tt0 = 0.3 # fase da senoide
ydata, edata = sinmod(tt, aa, pp, tt0, 0.025)
#endregion

maxrange = (0.038, 0.047) # intervalo para ajuste do máximo
tm = np.logical_and(tt > maxrange[0], tt < maxrange[1]) # máscara

#region ajuste de função parabólica e determinação do máximo
def p2(xx, c2, c1, c0):
    p2 = np.poly1d([c2, c1, c0])
    return p2(xx)

p2init = 0, 0 , np.max(ydata[tm]) 
p2opt, p2cov = curve_fit(p2, tt[tm], ydata[tm], p2init, 
                             sigma=edata[tm], absolute_sigma=True)
p2fit = p2(tt[tm], *p2opt)
p2max = minimize_scalar(p2, method='bounded', bounds=maxrange, 
                        args=tuple(-1.0 * p2opt)) # máx. => parâmentros * -1 
print(f't_max2 = {p2max.x:.8f} d')
#endregion

#region ajuste de função cúbica e determinação do máximo
def p3(xx, c3, c2, c1, c0):
    p3 = np.poly1d([c3, c2, c1, c0])
    return p3(xx)

p3init = 0, 0, 0, np.max(ydata[tm])
p3opt, p3cov = curve_fit(p3, tt[tm], ydata[tm], p3init, 
                         sigma=edata[tm], absolute_sigma=True)
p3fit = p3(tt[tm], *p3opt)
p3min = minimize_scalar(p3, method='bounded', bounds=maxrange, 
                        args=tuple(-1.0 * p3opt))
print(f't_max3 = {p3min.x:.8f} d') # máx. => parâmentros * -1 
#endregion

#region plota a curva de luz e ajustes
plt.close('all')
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,5), 
                        sharey=True, squeeze=True)
fig.subplots_adjust(hspace=0, wspace=0)

# painel à esquerda (curva de luz com região escolhida)
ax = axs[0]
ax.set_xlim(-0.001, 0.101)
ax.set_ylim(-0.4, 0.4)
ax.set_xlabel('$t$ (d)')
ax.xaxis.set_major_locator(mtick.MaxNLocator(prune='upper', nbins=7))
ax.set_ylabel('$\Delta mag$ (mag)')
ax.plot([0.0, 0.11], [0.0, 0.0], color='black', linestyle=':', zorder=0)
ax.errorbar(tt, ydata, edata, marker='o', color='green', ms=5, ls='none', 
            fillstyle='none', zorder=1)
ax.plot([maxrange[0], maxrange[0]], [-0.4, 0.4], color='gray', 
        linestyle='--', zorder=2)
ax.plot([maxrange[1], maxrange[1]], [-0.4, 0.4], color='gray', 
        linestyle='--', zorder=2)

#painel à direita (detalhe da região escolhida)
ax = axs[1]
ax.set_xlim(maxrange[0], maxrange[1])
ax.set_xlabel('$t$ (d)')
ax.xaxis.set_major_locator(mtick.MaxNLocator(prune='both', nbins=9))
ax.plot([0.0, 0.11], [0.0, 0.0], color='black', linestyle=':', zorder=0)
ax.errorbar(tt, ydata, edata, marker='o', color='green', ms=5, ls='none', 
            fillstyle='none', zorder=1)
ax.plot(tt[tm], p2fit , color='red', linestyle='-', zorder=2,
        label='ajuste parabólico')
ax.plot(tt[tm], p3fit , color='black', linestyle='--', zorder=3,
        label='ajuste cúbico')
ax.legend(loc='lower right')

plt.show()
#endregion