
# coding: utf-8




from keras.layers import Dense,SimpleRNN,Input,LSTM
from keras.models import Model
import numpy as np





bot=0
human=0
base=[]




def ir(m):
    if m==0:
        return "rock"
    if m==1:
        return "paper"
    if m==2:
        return "scissors"
def rep(m):
    a=[1,2,0]
    return a[m]
def ell(m,num):
    q=(m+1)/(num+1)
    if q==3 or q==0.5 or q==2/3:
        return "you won"
    if q==2 or q==1/3 or q==1.5:
        return "you lost"
    if q==1:
        return "tie"
def nell(m,num):
    q=(m+1)/(num+1)
    if q==3 or q==0.5 or q==2/3:
        return 2
    if q==2 or q==1/3 or q==1.5:
        return 1
    if q==1:
        return 0





a=np.zeros((30,3))
for i in range(30):
    print("choose one 1:rock,2:paper or 3:scissors")
    num=int(input())-1
    a[i,num]=1
    m=np.random.randint(0,3)
    print("your opponent said :", ir(m), "and",ell(m,num))




v=10
h=len(a)-v
chars_in  = np.zeros((h,v,3))
chars_out = np.zeros((h,3))
for i in range(h):
    for k in range(v):
        chars_in[i,k,:]=a[i+k,:]
    chars_out[i,:]=a[i+v,:]





inp=Input(shape=(10,3))
net=LSTM(128,activation='relu',return_sequences=True,dropout=0.3,recurrent_dropout=0.15)(inp)
net=LSTM(128,dropout=0.3,activation='relu',recurrent_dropout=0.15)(net)
net=Dense(3,activation='softmax')(net)
model=Model(inp,net)
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')




model.fit(chars_in,chars_out,epochs=10)




base=a





while True:
    num=int(input())
    q=num-1
    z=np.zeros(3)
    z[q]=1
    output=model.predict(chars_in)
    d=np.zeros((h,3))
    for i in range(h):
        q=np.argmax(output[i,:])
        d[i,q]=1
    u=nell(rep(np.argmax(d[0:])),num-1)
    if u==1:
        bot+=1
        print("score : human : %d ,bot : %d,winrate: %d"%(human,bot,human/(human+bot)))
    if u==2:
        human+=1
        print("score : human : %d ,bot : %d,winrate %d"%(human,bot,human/(human+bot)))
    base=np.vstack([base,z])
    v=10
    h=len(base)-v
    chars_in  = np.zeros((h,v,3))
    chars_out = np.zeros((h,3))
    for i in range(h):
        for k in range(v):
            chars_in[i,k,:]=base[i+k,:]
        chars_out[i,:]=base[i+v,:]
    
    model.fit(chars_in,chars_out,epochs=1)
    

