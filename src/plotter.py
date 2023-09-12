import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap
import numpy as np
from src.wiener_filter import *
from matplotlib.pyplot import cm

def plot_raster(df):
    fig=plt.figure()
    ax= fig.add_subplot(111)
    ax.set_axis_off()
    ax.table(cellColours=plt.cm.Greens(np.array(df)),rowLabels = df.index,
            colLabels = df.columns, cellLoc='center', loc = 'upper left')
    

def plot_gait_state_space_3D(list_of_array, subsample=5):
    #should be in gaits x gait samples x pca_dimensions
    #ONLY 2D FOR NOW
    #this is a little hard to explaijection='3d')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    average_gait = np.average(list_of_array, axis=0)
    random_sampling = np.random.randint(0, list_of_array.shape[0], subsample)
    gait_sampling = np.vstack(list_of_array[random_sampling,:,:])

    ax.scatter3D(average_gait[:,0], average_gait[:,1], average_gait[:,2], color='blue')
    ax.plot(average_gait[:,0], average_gait[:,1], average_gait[:,2], color='blue')

    ax.plot(gait_sampling[:,0], gait_sampling[:,1], gait_sampling[:,2], alpha=0.2, color='blue')
 
    return fig, ax

def plot_gait_state_space_2D(list_of_array, subsample=5):
    #should be in gaits x gait samples x pca_dimensions
    #ONLY 2D FOR NOW
    #this is a little hard to explain
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #color = iter(cm.rainbow(np.linspace(0, 1, len(list_of_array))))
    #c_current = next(color)
    average_gait = np.average(list_of_array, axis=0)
    random_sampling = np.random.randint(0, list_of_array.shape[0], subsample)
    gait_sampling = np.vstack(list_of_array[random_sampling,:,:2])


    ax.scatter(average_gait[:,0], average_gait[:,1], color='blue')
    ax.plot(average_gait[:,0], average_gait[:,1], color='blue')

    ax.plot(gait_sampling[:,0], gait_sampling[:,1], alpha=0.2, color='blue')
     

def plot_wiener_filter_predic(test_x, test_y, h, time=None):
    predic_y = test_wiener_filter(test_x, h)
    vaffy = vaf(test_y, predic_y)
    
    samples = np.shape(test_y)[0]

    ts = np.linspace(0, (samples*50)/1000,
            samples)

    if time is not None:
        start=time[0]
        end=time[-1]
        ts = ts[start:end]
        test_y=test_y[start:end,:]
        predic_y=predic_y[start:end,:]

    fig, ax = plt.subplots()
    ax.set_title(f'vaf:{vaffy}')
    ax.plot(ts, test_y, c='black')
    ax.plot(ts, predic_y, c='red')

    return fig, ax

def plot_both(array1, array2, subsample=5): #stupid function to save time tonight
    fig = plt.figure()
    ax = fig.add_subplot()
    #for gait in array:
    #    ax.plot3D(gait[0,:], gait[1,:], gait[2,:], color='lightsteelblue')
    avg1 = np.average(array1, axis=0)
    avg1 = np.vstack((avg1.T, avg1[:,0].T)).T

    avg2 = np.average(array2, axis=0)
    avg2 = np.vstack((avg2.T, avg2[:,0].T)).T
    
    random_sampling = np.random.randint(0, array1.shape[0], subsample)
    gait_sampling1 = np.vstack(array1[random_sampling,:,:2])
    gait_sampling2 = np.vstack(array2[random_sampling, :, :2])


    ax.plot(avg1[:,0], avg1[:,1], color='blue')
    ax.scatter(avg1[:,0], avg1[:,1], color='blue')
    
    ax.plot(avg2[:,0], avg2[:,1], color='orange')
    ax.scatter(avg2[:,0], avg2[:,1], color='orange')
    
    ax.plot(gait_sampling1[:,0], gait_sampling1[:,1],alpha=0.2, color='blue')
    ax.plot(gait_sampling2[:,0], gait_sampling2[:,1], alpha=0.2, color='orange')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

def plot_all(list_of_arrays, labels=None, subsample=5):

    #set subsample to 1 to turn off sampling
    fig = plt.figure()
    ax = fig.add_subplot()
    random_sampling = np.random.randint(0, list_of_arrays[0].shape[0],
            subsample)
    color = iter(cm.rainbow(np.linspace(0, 1, len(list_of_arrays))))
    for idx, array in enumerate(list_of_arrays):

        c_current = next(color)
        avg = np.average(array, axis=0)
        gait_sample = np.vstack(array[random_sampling, :, :2])

        ax.plot(avg[:,0], avg[:,1], label=labels[idx], color=c_current)
        ax.scatter(avg[:,0], avg[:,1], color=c_current)
        if subsample>1:
            ax.plot(gait_sample[:,0], gait_sample[:,1], alpha=0.2,
            color=c_current)



    ax.legend()
    ax.grid(True)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')







