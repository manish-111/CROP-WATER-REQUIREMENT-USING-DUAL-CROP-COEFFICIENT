#!/usr/bin/env python
# coding: utf-8

# In[210]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import array as arr


# In[211]:


dataset = pd.read_excel('C:/Users/rukmangadan/Desktop/reference ETo calculation/Manish.xlsx')


# In[212]:


start=135
end=253


# In[213]:


dataset.head()


# In[ ]:





# In[214]:


dataset


# In[215]:


R=dataset['Rain(mm)'][start:end]
R_total=dataset['Rain(mm)'][2*start-end:end]
#plt.plot(np.cumsum(R))
max(np.cumsum(R))
#plt.plot(R_total)


# In[216]:


R_total=dataset['Rain(mm)'][2*start-end:end]
ori_rain=np.zeros((2*(end-start),1))
for i in range (2*(end-start)):
    ori_rain[i]=R_total[i+2*start-end]
plt.plot(ori_rain)


# In[ ]:





# In[ ]:





# In[217]:


Rs = dataset['GR(J/cm²)']*.01
RH = dataset['RH(%)']
u = dataset['Wind(m/s)']
m= dataset['Month']
Tmax=dataset['MaxTemp(degree Celsius)']
Tmin=dataset['MinTemp(degree Celsius)']
gamma=.6232
alfa=0.23
phi=0.2196
sigma=4.903*10**-9


# In[218]:


RH


# In[219]:


T=np.zeros((1097,1))
for i in range(1097):
    T[i]=(Tmax[i]+Tmin[i])/2


# In[220]:


eoTmax=np.zeros((1097,1))
for i in range(1097):
    eoTmax[i]=.6108*np.exp(17.27*Tmax[i]/(237.3+Tmax[i]))


# In[ ]:





# In[221]:


eoTmin=np.zeros((1097,1))
for i in range(1097):
    eoTmin[i]=.6108*np.exp(17.27*Tmin[i]/(237.3+Tmin[i]))


# In[222]:


es=np.zeros((1097,1))
for i in range(1097):
    es[i]=(eoTmax[i]+eoTmin[i])/2


# In[223]:


delta=np.zeros((1097,1))
for i in range(1097):
    delta[i]=(.6108*np.exp(17.27*T[i]/(237.3+T[i])))/(237.3+T[i])**2


# In[224]:


ea=np.zeros((1097,1))
for i in range (1097):
    ea[i]=(RH[i]/100)*(es[i])


# In[225]:


ea[255]


# In[226]:


Rns=np.zeros((1097,1))
for i in range (1097):
    Rns[i]=(1-alfa)*Rs[i]


# In[227]:


Del=np.zeros((1097,1))
J=0
for i in range(1097):
    J=i
    Del[i]=0.409*math.sin(2*math.pi*J/365-1.39)
    


# In[228]:


X=np.zeros((1097,1))
for i in range(1097):
    X[i]=1-((math.tan(phi))**2)*((math.tan(Del[i]))**2)
    

    


# In[229]:


Ws=np.zeros((1097,1))
for i in range (1097):
    Ws[i]=math.pi/2-math.acos(-1*math.tan(phi)*math.tan(Del[i]))


# In[230]:


dr=np.zeros((1097,1))
for i in range(1097):
    J=i
    dr[i]=1+0.33*math.cos(2*math.pi*J/365)


# In[231]:


Ra= np.zeros((1097,1))
for i in range(1097):
    Ra[i]=24*60*.082*dr[i]*(Ws[i]*math.sin(phi)*math.sin(Del[i])+math.cos(phi)*math.cos(Del[i])*math.sin(Ws[i]))


# In[232]:


1/math.pi


# In[ ]:





# In[233]:


Rso=np.zeros((1097,1))
for i in range (1097):
    Rso[i]=(0.75+(2*10**-5)*662)*Ra[i]


# In[234]:


Rnl=np.zeros((1097,1))
for i in range(1097):
    Rnl[i]=sigma*((((Tmax[i]+273.16)**4)+((Tmin[i]+273.16)**4))/2)*(0.34-0.14*np.sqrt(ea[i]))*(1.35*Rs[i]/Rso[i]-0.35)


# In[235]:


Rn=np.zeros((1097,1))
for i in range(1097):
    Rn[i]=Rns[i]-Rnl[i]


# In[236]:


PET=np.zeros((1097,1))
for i in range(1097):
    PET[i]=(.408*Del[i]*Rn[i]+(gamma*900*u[i]*(es[i]-ea[i])/(T[i]+273)))/(Del[i]+gamma*(1+0.34*u[i]))


# In[237]:


np.mean(dataset['PET(mm)'])


# In[238]:


np.min(PET)


# In[239]:


np.mean(PET)


# In[240]:


np.min(X)


# In[241]:


r = np.where(PET == np.max(PET))
r[0], PET[86]


# In[242]:


np.max(PET)


# In[243]:


np.max(dataset['PET(mm)']),np.min(dataset['PET(mm)'])


# In[244]:


import matplotlib.pyplot as plt
plt.plot(dataset['PET(mm)'])
plt.plot(PET)


# In[245]:


plt.plot(abs(PET[start:end]))
plt.xlabel('days')
plt.ylabel('(mm)')
plt.title('Reference Evapotranspiration 2016 maize')
plt.savefig('C:/Users/rukmangadan/Desktop/New folder (2)/PET 2016 maize')


# In[246]:


#Total water need 
twn=np.int(np.max(np.cumsum(abs(PET[start:end]))))
twn


# In[247]:


#MDIR   maize daily crop irrigation requirement
MDCR=np.zeros((118,1))
MDCR=PET[start:end]
MDCR[117]


# In[248]:


may=np.int(np.max(np.cumsum(MDCR[0:15])))
june=np.int(np.max(np.cumsum(MDCR[15:45])))
july=np.int(np.max(np.cumsum(MDCR[45:76])))
aug=np.int(np.max(np.cumsum(MDCR[76:107])))
sep=np.int(np.max(np.cumsum(MDCR[107:117])))
may,june,july,aug,sep


# In[249]:


d_net=60
ea=75
d_gross=100*d_net/ea
d_gross


# In[250]:


no_of_app=np.int(twn/d_net)
no_of_app+1


# In[251]:


#irrigation interval
INT=np.int((end-start)/no_of_app)
INT


# In[252]:


#Adjusting the simple calculation method for peak period
IN1=np.array([may,june,july,aug,sep])
IN=IN1.reshape(5,1)
IN


# In[253]:


#per month water required
PMW=np.int(31*d_net/INT)
PMW


# In[254]:


207/60


# In[255]:


err1=np.zeros((5,1))
for i in range(5):
    err1[i]=IN[i]-PMW
err1


# In[256]:


itr2=np.int((june+july+aug)/d_net+1)
itr2


# In[257]:


days=int(90/7)


# In[258]:


118/12*60


# In[259]:


(.81-.69)*.67+.69


# In[ ]:





# In[ ]:





# In[260]:


590*10/118


# In[261]:


np.max(PET)


# In[262]:


np.min(PET)


# In[263]:


r=np.where((PET<=0))
r[0]


# In[264]:


#plt.plot(T)


# In[265]:


#plt.plot(RH)


# In[266]:


#plt.plot(u)


# In[267]:


#plt.plot(Rs)


# In[ ]:





# In[ ]:





# In[268]:


np.mean(PET)


# In[ ]:





# In[269]:


plt.plot(abs(PET))
plt.xlabel('Days')
plt.ylabel('PET(mm)')
plt.title('Evapotranspiration')
plt.savefig('c:/Users/rukmangadan/Desktop/New folder (2)/PET(mm)')


# In[270]:


#PET1 = np.concatenate((PET[:334],PET[337:346],PET[348:]),axis = 0)
#PET2 =np.concatenate((dataset['PET(mm)'][:334],dataset['PET(mm)'][337:346],dataset['PET(mm)'][348:]),axis = 0)
#plt.plot(PET1)
#plt.plot(PET2)


# In[271]:


a=np.cumsum(PET)
b = np.cumsum(dataset['PET(mm)'])
plt.plot(a)
plt.plot(b)


# In[272]:



ini_period=18
dov_period=12
mid_period=70
end_period=18


# In[273]:


Kcin=np.zeros((ini_period,1))
for a in range(ini_period):
    Kcin[a] =0.3
cbest_in=0.3


# In[274]:


plt.plot(Kcin)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[275]:


#RHmin=np.zeros((1097,1))
#for i in range (1097):
 #   RHmin[i]=ea[i]*eoTmax[i]


# In[ ]:





# In[ ]:



    


# In[276]:


Kcmid=np.zeros((mid_period,1))
for i in range(mid_period):
    Kcmid[i] = 1.13+(0.04*(u[i+start+ini_period+dov_period]-2)-0.004*(RH[i+start+ini_period+dov_period]-45))*(2/3)**.3
    


# In[277]:


Kcmax=np.zeros((end-start,1))

for i in range (end-start):
    Kcmax[i]=max((1.2+(0.04*(u[i+start]-2)-0.004*(RH[i+start]-45))*(2/3)**.3), (Kcb[i]-0.05))

plt.plot(Kcmax)


# In[278]:


for i in range(mid_period):
    Kcmid[i]=min(Kcmid[i],Kcmax[i+ini_period+dov_period])
plt.plot(Kcmid)


# In[279]:


def minfind(c):
    err=0
    for i in range (mid_period):
        err = err + (Kcmid[i]-c)**2
    err = err/mid_period
    return err

minierr = minfind(min(Kcmid))
miniarr = np.zeros((20))
c =min(Kcmid)
cbest = 0
for i in range(20):
    if minfind(c) < minierr:
        minierr = minfind(c)
        cbest = c
    miniarr[i] = minfind(c)
    c = c+0.025

minierr, miniarr, cbest


# In[280]:


aa=np.zeros((mid_period,1))
for i in range(mid_period):
    aa[i]=cbest


# In[281]:


x1 = np.linspace(1,mid_period,mid_period)
y1 = aa

plt.show()


# In[282]:


plt.scatter(x1,Kcmid ,color = 'g',label='daily values')
plt.plot(x1, y1, color = 'r',label='min squared error value')

plt.xlabel('Days')
plt.ylabel('adjusted value')
plt.legend()
plt.savefig('c:/Users/rukmangadan/Desktop/New folder (2)/adjusted mid kc mize 2016')


# In[ ]:



        


# In[ ]:





# In[ ]:





# In[ ]:





# In[283]:


cbest, cbest_in


# In[284]:


g=np.arange(dov_period)
kc_dov=np.zeros((dov_period,1))
slope = (cbest-0.3)/dov_period
for i in range(dov_period):
    kc_dov[i]= slope*(i+1)

plt.scatter(g,kc_dov)
    


# In[285]:


yfit


# In[286]:


answer = np.zeros((end-start,1))
answer[:ini_period] = 0.3
answer[ini_period:ini_period+dov_period] = 0.3+kc_dov
answer[ini_period+dov_period:ini_period+dov_period+mid_period] = cbest
answer[ini_period+dov_period+mid_period:] = yfit
plt.plot(answer)


# In[ ]:





# In[287]:


minkcmid=np.min(Kcmid)
minkcmid


# In[288]:


cc=np.arange(end_period)

Kcend=np.zeros((end_period,1))
for i in range (end_period):
    Kcend[i]=(0.6-cbest)*i/end_period+cbest
    i=i+1
#plt.plot(cc,Kcend)
plt.scatter(cc,Kcend)


# In[289]:


# sample points 
X3 = np.arange(end_period)
Y3 = Kcend

# solve for a and b
def best_fit(X3, Y3):

    xbar = sum(X3)/len(X3)
    ybar = sum(Y3)/len(Y3)
    n = len(X3) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X3, Y3)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X3]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    return a, b

# solution
a, b = best_fit(X3, Y3)
#best fit line:
#y = 0.80 + 0.92x

# plot points and fit line
import matplotlib.pyplot as plt
plt.scatter(X3, Y3)
yfit = [a + b * xi for xi in X3]
plt.plot(X3, yfit)


# In[290]:


yfit=np.array(yfit)
yfit = yfit.reshape(end_period,1)


# In[ ]:





# In[ ]:





# In[291]:


r=np.where((Kcend<0))
r[0]


# In[ ]:





# In[292]:


cotto = np.concatenate((PET[:334],PET[337:346],PET[348:]),axis = 0)


# In[293]:


Kcb=np.zeros((end-start,1))
for i in range(end-start):
    Kcb[i]=answer[i]
plt.plot(Kcb)


# In[ ]:





# In[ ]:





# In[294]:


#calculation of Kr
thetaFc=.25
thetaWp=0.1
Ze=0.1
REW=9

TEW=1000*(thetaFc-0.5*thetaWp)*Ze
TEW


# In[ ]:





# In[295]:


Kcmax.shape


# In[ ]:





# In[296]:


plt.plot(R)


# In[297]:


Ke=np.zeros((end-start,1))
Kr=np.zeros((end-start,1))
De=np.zeros((end-start,1))
for i in range (end-start-1):
    if De[i]<=REW:
        Kr[i]=1
        Ke[i]=Kr[i]*(Kcmax[i]-Kcb[i])
    else:
        Kr[i]=(TEW-De[i])/(TEW-REW)
        Ke[i]=Kr[i]*(Kcmax[i]-Kcb[i])
    De[i+1] = Ke[i]*PET[i+start] + De[i]
    
plt.plot(Ke) 
De.shape


# In[ ]:





# In[298]:


rainfall=np.where(R>0.5*(thetaFc-thetaWp)*100)
r = rainfall[0]
r


# In[299]:


r = np.append(r,end-start)
r


# In[300]:


def efunc(interval):
    Ke = np.zeros([interval])
    for i in range (interval):
        if De[i]<=REW:
            Kr[i]=1
            Ke[i]=Kr[i]*(Kcmax[i]-Kcb[i])
        else:
            Kr[i]=(TEW-De[i])/(TEW-REW)
            Ke[i]=Kr[i]*(Kcmax[i]-Kcb[i])
        De[i+1] = Ke[i]*PET[i+start] + De[i]
    return Ke


# In[301]:


efunc(r[0]).shape


# In[302]:


Kcmin=0.3


# In[303]:


antar=np.zeros((end-start))
for i in range(len(r)):
    if i == 0:
        antar[:r[0]] = efunc(r[0])
    else:
        antar[r[i-1]:r[i]] = efunc(r[i] - r[i-1])


# In[304]:


plt.plot(antar)


# In[305]:


fc=np.zeros((end-start,1))
for i in range(end-start):
    fc[i]=(max((Kcb[i]-Kcmin),.01)/((Kcmax[i]-Kcmin)))**(1+0.5*2)
    

for i in range (end-start):
    if fc[i]>.99:
        fc[i]=.99
    else:
        fc[i]=fc[i]

max(fc)


# In[306]:


#for basin irrigation fw=1
fw=1
few=np.zeros((end-start,1))
for i in range (end-start):
    few[i]=min((1-fc[i]),(fw))


# In[307]:


for i in range(end-start):
    Ke[i]=min(antar[i],few[i]*Kcmax[i])


# In[308]:


plt.plot(Ke)


# In[309]:


Kc=np.zeros((end-start,1))
for i in range (end-start):
    Kc[i]=Kcb[i]+Ke[i]
Kc
plt.plot(Kc)

plt.xlabel('days')
plt.title('crop coefficient Kc=Ke+Kcb')
plt.savefig('C:/Users/rukmangadan/Desktop/New folder (2)/Kc for maize 2016')


# In[310]:


ETc=np.zeros((end-start,1))
for i in range(end-start):
    ETc[i]=Kc[i]*PET[i+start]
plt.plot(ETc)
plt.plot(PET[start:end])


# In[311]:


max(np.cumsum(ETc))


# In[312]:


R1=dataset['Rain(mm)'][(2*start-end):end]
plt.plot(R1)
o=np.arange((2*start-end),end,1)
plt.plot(o,abs(PET[(2*start-end):end]))


# In[313]:


mb=2*(end-start)
R1= np.array(R1)
R1 = R1.reshape(mb,1)


# In[314]:


eff_R1=np.zeros((mb,1))
for i in range(mb):
    if R1[i]-PET[i+(2*start-end)]>0:
        eff_R1[i]=R1[i]
    else:
        eff_R1[i]=R1[i]


# In[315]:


plt.plot(eff_R1)


# In[316]:


eff_R1.shape


# In[317]:


#rainy=np.where(eff_R1!=0)[0]
#rainy1=np.reshape(rainy)


# In[318]:


#rainy=np.array([ 3,   4,  83,  89, 105, 106, 111, 113, 120, 131, 139, 156, 158,
 #       161, 162, 163, 164, 165, 170, 171, 174, 185, 192,118])
#rainy


# In[319]:


theta_s=0.35
theta_r=0.05
Rz=1000
AW=(theta_s-theta_r)*Rz
AW


# In[ ]:





# In[320]:


##########code for available water for double the crop period################
sowing = 118
thetaFc=.25
AW1=np.zeros((2*(end-start),1))

for i in range(1,2*(end-start)):
    if i<sowing:
        if AW1[i-1]>(thetaFc*Rz):
            AW1[i]=AW1[i-1]+ori_rain[i]-0.1*(ori_rain[i]+AW1[i-1])-evaporation[i]
        else:
            AW1[i]=AW1[i-1]+ori_rain[i]
    else:
        
        if AW1[i-1]>(thetaFc*Rz):
            AW1[i]=AW1[i-1]+ori_rain[i]-0.1*(ori_rain[i]+AW1[i-1])-ETc[i-sowing]
            AW1[118]=AW1[117]+(.75*(thetaFc*Rz)-AW1[117])
        else:
            AW1[i]=AW1[i-1]+ori_rain[i]-ETc[i-sowing]
            AW1[118]=AW1[117]+(.75*(thetaFc*Rz)-AW1[117])
        #if AW1[i]<=.5*(thetaFc*Rz):
         #   AW1[i]=.75*(thetaFc*Rz)


# In[ ]:





# In[321]:


recharge=np.zeros(((end-start),1))
for i in range ((end-start)):
    if AW1[i+sowing]>(thetaFc*Rz):
        recharge[i]=recharge[i]+0.1*(ori_rain[i]+AW1[i+sowing]-AW1[sowing])
    else:
        recharge[i-sowing]=recharge[i-sowing]
        
plt.plot(recharge)
max(np.cumsum(recharge))


# In[322]:


plt.plot((AW1))
oo=np.arange((end-start),2*(end-start),1)
oo1=np.arange(2*(end-start))
plt.plot(oo,ETc,label='crop evapotranspiration')
thetafc_arr=np.zeros((2*(end-start),1))
for i in range (2*(end-start)):
    thetafc_arr[i]=0.75*thetaFc*Rz
plt.plot(oo1,thetafc_arr,label='y=281.25,0.75 OF FC')



FC=np.zeros((2*(end-start),1))
for i in range (2*(end-start)):
    FC[i]=thetaFc*Rz
plt.plot(oo1,FC,label='y=375,FC')


sat=np.zeros((2*(end-start),1))
for i in range (2*(end-start)):
    sat[i]=theta_s*Rz
    
plt.plot(oo1,sat,label='y=525,saturation')
thetawp_arr=np.zeros((2*(end-start),1))
for i in range (2*(end-start)):
    thetawp_arr[i]=0.5*thetaFc*Rz
plt.plot(oo1,thetawp_arr,label='y=187.5,half of FC')

plt.xlabel('days')
plt.ylabel('water depth in soil (mm)')

plt.plot(ori_rain,label="rainfall")
plt.title('irrigation scheduling 2016 maize')
plt.legend()
plt.savefig('C:/Users/rukmangadan/Desktop/New folder (2)/irrigation scheduling 2016 maize mid 70')
AW1[sowing-1],AW1[sowing]


# In[323]:


irri_demand=np.zeros(((end-start),1))
for i in range (end-start):
    irri_demand[i]=.75*FC[i]-AW1[i+sowing]

    plt.plot((irri_demand))
    


# In[ ]:





# In[324]:


end-start


# In[325]:


plt.plot(ori_rain)


# In[326]:


plt.plot(np.cumsum(PET[start:end]))
plt.xlabel('days')
plt.ylabel('Total PET (mm)')
max(np.cumsum(PET[start:end]))


# In[327]:


plt.plot(np.cumsum(Kcb*PET[start:end]))
plt.xlabel('days')
plt.ylabel('Total transpiration (mm)')
max(np.cumsum(Kcb*PET[start:end]))


# In[328]:


AW1_2017 = np.loadtxt('AW12017.csv',delimiter=',')


# In[329]:


plt.plot(AW1[end-start:2*(end-start)])
plt.plot(AW1_2017[end-start:2*(end-start)])

w16=max(np.cumsum(AW1[end-start:2*(end-start)]))/(end-start)
w17=max(np.cumsum(AW1_2017[end-start:2*(end-start)]))/(end-start)

w16,w17


# In[330]:


evaporation=np.zeros((end-start,1))
for i in range(end-start):
    evaporation[i]=Ke[i]*PET[i+start]
plt.plot(np.cumsum(evaporation))
plt.xlabel('days')
plt.ylabel('evaporation (mm)')
max(np.cumsum((evaporation)))


# In[331]:


np.cumsum(eff_R1[end-start:2*(end-start)])
plt.plot(np.cumsum(eff_R1[end-start:2*(end-start)]))
plt.xlabel('days')
plt.ylabel('cum rainfall (mm)')
max((np.cumsum(eff_R1[end-start:2*(end-start)])))


# In[ ]:





# In[332]:


plt.plot(np.cumsum(PET[start:end]),label="Total PET")
plt.plot(np.cumsum(Kcb*PET[start:end]),label="Total Transpiration")
plt.plot(np.cumsum(evaporation),label="Total evaporation")

oo2=np.arange(end-start)
plt.plot(oo2,np.cumsum(R),label="Total rainfall")

plt.plot(oo2,np.cumsum(recharge),label="recharge")
plt.plot(oo2,np.cumsum(ETc),label="Total AET")
plt.xlabel('days')
plt.ylabel('(mm)')
plt.title ('maize 2016')

plt.legend()
plt.savefig('C:/Users/rukmangadan/Desktop/New folder (2)/Total 2016 maize mid 70')


# In[ ]:





# In[ ]:




