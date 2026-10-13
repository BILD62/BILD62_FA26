def load_dictionary():
    dictionary = {'PASL': {'time': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
                                    1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
                                    2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
                           'label_times':[0.6, 0.8, 1.0]},
                  'PCASL': {'time': [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2,
                                     2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4,
                                     4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.4, 5.6, 5.8],
                           'label_times':[1.6, 1.8, 2.0]}}
    return dictionary

import numpy as np

def PCASL3T(LD, PLD, CBF=60, ATT=.6):

    LD = np.asarray(LD)
    PLD = np.asarray(PLD)
    
    SI_PCASL = np.zeros(LD.shape)
    pre_idx = np.where(LD + PLD < ATT)[0]
    label_idx = np.where((ATT <= LD + PLD) &((LD + PLD) <= (LD + ATT)))[0]
    decay_idx = np.where(ATT <= PLD)[0]
    #print(pre_idx, label_idx, decay_idx)

    T1b = 1.65 #3T
    M0a = 100
    alpha = 0.85
    alpha_BS = 1
    scaling_factor = 6000

    SI_PCASL[pre_idx] = 0
    SI_PCASL[label_idx] = 2*(alpha*alpha_BS*T1b*M0a*CBF*np.exp(-ATT/T1b)*(1-np.exp(-(LD[label_idx]+PLD[label_idx]-ATT)/T1b)))/scaling_factor
    SI_PCASL[decay_idx] = 2*(alpha*alpha_BS*T1b*M0a*CBF*np.exp(-PLD[decay_idx]/T1b)*(1-np.exp(-LD[decay_idx]/T1b)))/scaling_factor

    return SI_PCASL


def PASL3T(TI1, TI, CBF=60, ATT=0.6):

    TI1 = np.asarray(TI1)
    TI = np.asarray(TI)
    
    SI_PASL = np.zeros(TI1.shape)
    pre_idx = np.where(TI < ATT)[0]
    label_idx = np.where((ATT <= TI) & (TI < (ATT+TI1)))[0]
    decay_idx = np.where(ATT + TI1 <= TI)[0]
    #print(pre_idx, label_idx, decay_idx)

    T1b = 1.65 #1.5T
    M0a = 100
    alpha = 0.85
    alpha_BS = 1
    scaling_factor = 6000

    SI_PASL[pre_idx] = 0
    SI_PASL[label_idx] = 2 * (alpha*alpha_BS*M0a*CBF*np.exp(-TI[label_idx]/T1b)*(TI[label_idx]-ATT)) / scaling_factor
    SI_PASL[decay_idx] = 2 * (alpha*alpha_BS*M0a*CBF*np.exp(-TI[decay_idx]/T1b)*(TI1[decay_idx])) / scaling_factor

    return SI_PASL

def get_experiment_timings(time, label_time, mode='PASL'):

    if mode == 'PASL':
        TI = time.copy()
        TI1 = time.copy()
        TI1 = [t if t <= label_time else label_time for t in TI1]
        return TI, TI1
        
    elif mode == 'PCASL':
        LD = time.copy()
        PLD = time.copy()
        LD = [t if t <= label_time else label_time for t in LD]
        PLD = [0 if t <= label_time else t - label_time for t in PLD]
        return LD, PLD

import matplotlib.pyplot as plt

def plot_computed_models(time1, TI1, TI,
                         time2, LD, PLD):
    
    SI_PASL = PASL3T(TI1, TI)
    SI_PCASL = PCASL3T(LD, PLD)

    plt.plot(time1, SI_PASL)
    plt.plot(time2, SI_PCASL)
    plt.legend(['PASL', 'PCASL'])
    plt.ylabel('Signal Intensity')
    plt.xlabel('time (s)')
    plt.title('Models of Arterial Spin Labeling Intensities')
