import numpy as np
import matplotlib.pyplot as plt 
import scipy.integrate as inte 
import matplotlib.ticker as mticker

def S2(x,u):
    Bz,Ba,Bb,yR,yB,yL,va,vb=u
    F0 = 25.78
    F1 = 77.79
    F2 = 5.6 * 1e-6
    F3 = 356 * 1e+6
    F4 = 77809 * 1e+8
    F5 = 0.304
    F6 = 7.12
    aY = 0.0095
    Gam0 = 121
    k2=1
    Pi=np.pi

    dyR = F0*((Ba/1e+20)**2 + (Bb/1e+20)**2)* x**(3/2)\
        - F1*(yR - yL/2 + (3/8)*yB)*((Bz/1e+20)**2 + (Ba/1e+20)**2 +\
            (Bb/1e+20)**2) * x**(3/2) -\
                F2*(yR**2 - yL**2)*(va*Ba/1e+20 + vb*Bb/1e+20) * x**(1/2)\
                        - Gam0 * (1 - x) * (yR - yL)/x**(1/2)

    dyL = -F0/4*((Ba/1e+20)**2 + (Bb/1e+20)**2) * x**(3/2)\
        + F1/4*(yR - yL/2 + (3/8)*yB)*((Bz/1e+20)**2 + (Ba/1e+20)**2\
            + (Bb/1e+20)**2) * x**(3/2) +\
                F2/4*(yR**2 - yL**2)*(va*Ba/1e+20 + vb*Bb/1e+20) * x**(1/2)\
                        + Gam0*(1 - x)*(yR - yL) /(2 * x**(1/2))

    dyB = (3/2)*F0*((Ba/1e+20)**2 +\
        (Bb/1e+20)**2) * x**(3/2) - \
            (3/2)*F1*(yR - yL/2 + (3/8)*yB)*((Bz/1e+20)**2\
                + (Ba/1e+20)**2 + (Bb/1e+20)**2) * x**(3/2)\
                    - (3/2)*F2*(yR**2 - yL**2)*(va*Ba/1e+20 + vb*Bb/1e+20) * x**(1/2)

    dBz = -Bz/x

    dBa = 356*k2*(aY*(yR - yL/2 + (3/8)*yB)/Pi\
        - k2/1e+3) * Ba/x**(1/2) + F3*vb*Bz/x**(1/2) \
            + F4*(yR**2 - yL**2)*va/x**(3/2) - Ba/x

    dBb = 356*k2*(aY*(yR - yL/2 + (3/8)*yB)/Pi - 1/1e+3)*Bb/x**(1/2) \
        - F3*va*Bz/x**(1/2) + F4*(yR**2 - yL**2) * vb/x**(3/2) - Bb/x

    dva = F5*(Bz/1e+20)*(Bb/1e+20) * x**(3/2) - F6*va/(aY**2 * x**(1/2))

    dvb = -F5*(Bz/1e+20)*(Ba/1e+20) * x**(3/2) - F6*vb/(aY**2 * x**(1/2))
    return np.array([dBz,dBa,dBb,dyR,dyB,dyL,dva,dvb])


# S0_color=[Bz0,Ba0,Bb0,yR0,yB0,yL0,va0,vb0]

#Red
S0_red=[1e+22,1e+20,1e+20,0,0,0,0,0]

#Blue
S0_blue=[1e+20,1e+20,1e+20,0,0,0,0,0]

#Green
S0_green=[1e+18,1e+20,1e+20,0,0,0,0,0]

def sol(initials):
    res=list()
    for i in initials:
        s= inte.solve_ivp(fun=S2,t_span=(1e-4,1),y0=i,method='LSODA',rtol=5e-12,atol=1e-40)
        Bz,Ba,Bb,yR,yB,yL,va,vb=s.y
        x=s.t
        nR = 3.55929 * 1e-7 * yR
        nL = 3.55929 * 1e-7 * yL
        nB = 3.55929 * 1e-7 * yB
        By = (Ba**2 + Bb**2 + Bz**2)**0.5
        
        ner_=(x,nR)
        nel_=(x,nL)
        nb_=(x,nB)
        ba_=(x,Ba)
        bb_=(x,Bb)
        va_=(x,va)
        vb_=(x,vb)
        by_=(x,By)

        res.append([ner_,nel_,nb_,ba_,bb_,va_,vb_,by_])

    return res

# print('number of data points:',len(x))
print(np.shape(sol([S0_red,S0_blue,S0_green])))

def symp(l,label,x_scale,y_scale,linthresh,ty):

    fig, ax = plt.subplots()
    for i in range(len(l)):
        style=[':','dashed','dashdot','dotted']
        dashes=[(5, 1),(1,1),(3, 2),(3, 1, 1, 1)]
        col=['Red','Blue','Green','Purple']
        csfont = {'fontname':'Leelawadee UI'}
        line= plt.plot(l[i][0],l[i][1],linewidth=3.0)
        plt.xscale(x_scale)
        plt.yscale(y_scale,linthresh=linthresh)
        plt.xticks(fontsize=16, fontweight='bold',**csfont)
        plt.yticks(fontsize=16, fontweight='bold',**csfont)
        plt.xlabel('x',fontsize=26, fontweight='bold',**csfont)
        plt.ylabel(label,fontsize=26, fontweight='bold',**csfont)
        plt.setp(line, linestyle=style[i],color=col[i],dashes=dashes[i])
        plt.locator_params(axis='y')
    yticks = ax.yaxis.get_major_ticks()
    if ty is not None:
        for i in ty:
            yticks[i].set_visible(False)
    # yticks[15].set_visible(False)
    # yticks[9].set_visible(False)
    # yticks[1].set_visible(False)
    plt.tight_layout()
    # plt.savefig(str(label+'.eps'), format='eps')
    plt.show()

def zsymp(l,label,x_scale,y_scale,linthresh,ty,lower_bound,upper_bound,left, bottom, width, height):
    linthresh=linthresh

    for i in range(len(l[0][0])):
        if upper_bound<l[0][0][i]:
            print(upper_bound,l[0][0][i])
            zf_u=i
            break

    for i in range(len(l[0][0])):
        if lower_bound<l[0][0][i]:
            print(lower_bound,l[0][0][i])
            zf_l=i
            break

    # zf=np.where(l[0][1] > 0, l[0][1], np.inf).argmin()
    f=10
    fig, ax1 = plt.subplots()
    for i in range(len(l)):
        style=[':','dashed','dashdot','dotted']
        dashes=[(5, 1),(1,1),(3, 2),(3, 1, 1, 1)]
        col=['Red','Blue','Green','Purple']
        csfont = {'fontname':'Leelawadee UI'}
        line= plt.plot(l[i][0],l[i][1],linewidth=3.0)
        plt.xscale(x_scale)
        plt.yscale(y_scale,linthresh=linthresh)
        plt.xticks(fontsize=16, fontweight='bold',**csfont)
        plt.yticks(fontsize=16, fontweight='bold',**csfont)
        yticks = ax1.yaxis.get_major_ticks()
        if ty is not None:
            for i in ty:
                yticks[i].set_visible(False)
        # yticks[9].set_visible(False)
        # yticks[7].set_visible(False)
        plt.xlabel('x',fontsize=26, fontweight='bold',**csfont)
        plt.ylabel(label,fontsize=26, fontweight='bold',**csfont)
        plt.setp(line, linestyle=style[i],color=col[i],dashes=dashes[i])
        plt.locator_params(axis='y')
    # left, bottom, width, height = [0.66, 0.6, 0.28, 0.28]
    ax2 = fig.add_axes([left, bottom, width, height])
    line2=ax2.plot(l[0][0][zf_l:zf_u],l[0][1][zf_l:zf_u])
    # ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(g))
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(g))
    plt.setp(line2, linestyle=style[0],color=col[0],dashes=dashes[0])
    plt.locator_params(axis='x',nbins=2)
    plt.locator_params(axis='y',nbins=2)
    plt.tight_layout()
    plt.show()

def logp(l,label,x_scale,y_scale):
    for i in range(len(l)):
        style=[':','dashed','dashdot','dotted']
        dashes=[(5, 1),(1,1),(3, 2),(3, 1, 1, 1)]
        col=['Red','Blue','Green','Purple']
        csfont = {'fontname':'Leelawadee UI'}
        line= plt.plot(l[i][0],l[i][1],linewidth=3.0)
        plt.xscale(x_scale)
        plt.yscale(y_scale)
        plt.xticks(fontsize=12, fontweight='bold',**csfont)
        plt.yticks(fontsize=12, fontweight='bold',**csfont)
        plt.xlabel('x',fontsize=26, fontweight='bold',**csfont)
        plt.ylabel(label,fontsize=26, fontweight='bold',**csfont)
        plt.setp(line, linestyle=style[i],color=col[i],dashes=dashes[i])
        plt.locator_params(axis='y')
    plt.tight_layout()
    # plt.savefig(str(label+'.eps'), format='eps')
    plt.show()

RED,BLUE,GREEN=sol([S0_red,S0_blue,S0_green])
ner_red,nel_red,nb_red,ba_red,bb_red,va_red,vb_red,by_red=RED
ner_blue,nel_blue,nb_blue,ba_blue,bb_blue,va_blue,vb_blue,by_blue=BLUE
ner_green,nel_green,nb_green,ba_green,bb_green,va_green,vb_green,by_green=GREEN


ner_=[ner_red,ner_blue,ner_green]
nel_=[nel_red,nel_blue,nel_green]
nb_=[nb_red,nb_blue,nb_green]
ba_=[ba_red,ba_blue,ba_green]
bb_=[bb_red,bb_blue,bb_green]
va_=[va_red,va_blue,va_green]
vb_=[vb_red,vb_blue,vb_green]
by_=[by_red,by_blue,by_green]

symp(bb_,r'${B}_{b}(x)[G]$','log','symlog',linthresh=10e-4,ty=None)
logp(by_,r'${B}_{Y}(x)[G]$','log','log')
symp(ner_,r'$\eta_{e_{R}}(x)$','log','symlog',linthresh=10e-20,ty=None)
symp(nel_,r'$\eta_{e_{L}}(x)$','log','symlog',linthresh=10e-20,ty=None)
symp(nb_,r'$\eta_{B}(x)$','log','symlog',linthresh=10e-20,ty=None)
symp(ba_,r'${B}_{a}(x)[G]$','log','symlog',linthresh=10e-4,ty=None)
zsymp(va_,r'${v}_{a}(x)$','log','symlog',linthresh=10e-32,ty=None,lower_bound=1.0001e-4,upper_bound=1.008e-4,left=0.66, bottom=0.3, width=0.28, height=0.28)
zsymp(vb_,r'${v}_{b}(x)$','log','symlog',linthresh=10e-32,ty=None,lower_bound=1.0001e-4,upper_bound=1.008e-4,left=0.66, bottom=0.6, width=0.28, height=0.28)



# def f(t,u):
#     # Defining parameters
#     w1=1
#     vx=u[1]
#     ax=-w1**2 * u[0]
#     return np.array([vx,ax])

# s= inte.solve_ivp(fun=f,t_span=(0,np.pi),y0=[0,1],method='DOP853',rtol=5e-14,atol=1e-40)
# x,v=s.y
# t=s.t
# # plt.plot(t,x,c='blue')
# # plt.plot(t,np.sin(t),c='red')
# # plt.show()
# # print('number of data points:',len(x))

# Decimal_error_location=list()

# for j in range(len(x)):
#     Nsolve=str("{:.100f}".format(x[j]))
#     Exact=str("{:.100f}".format(np.sin(t[j])))
#     for i in range(len(Nsolve)):
#         if Nsolve[i]!=Exact[i]:
#             Decimal_error_location.append(i)
#             break
# Decimal_error_location=Decimal_error_location[:-1]

# print(min(Decimal_error_location))
