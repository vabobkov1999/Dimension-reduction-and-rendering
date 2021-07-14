import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random

h5_data = 'data_LR3.h5'
train_path = "dataset/train"

train_labels = os.listdir(train_path)
train_labels.sort()
print(train_labels) #['Clefable', 'Clefairy', 'Ditto', 'Drowzee', 'Pikachu', 'Raichu']

h5f_data = h5py.File(h5_data, 'r')
global_features = np.array(h5f_data["dataset_1"])
h5f_data.close()

#Start PCA
pca_model = PCA(n_components=2, random_state=0)
tmp = pca_model.fit_transform(global_features)
print(tmp.shape)
fig, ax = plt.subplots()    #SAME as: fig = plt.figure()
#                                     ax = fig.add_subplot(111)
ax.set(title='PCA for LR3')
ax.scatter(tmp[:57][:,0], tmp[:57][:,1], color='red', label='Chansey')
ax.scatter(tmp[58:105][:,0], tmp[58:105][:,1], color='green', label='Grimer')
ax.scatter(tmp[106:156][:,0], tmp[106:156][:,1], color='blue', label='Lickitung')
ax.scatter(tmp[157:209][:,0], tmp[157:209][:,1], color='yellow', label='Muk')
ax.scatter(tmp[210:261][:,0], tmp[210:261][:,1], color='pink', label='Slowbro')
ax.scatter(tmp[262:][:,0], tmp[262:][:,1], color='black', label='Slowpoke')
ax.legend()
plt.show()
fig.savefig('PCA_LR3.png')


'''
Изменённый метод PCA для доп задания

#Start PCA
pca_model = PCA(n_components=3, random_state=0)
tmp = pca_model.fit_transform(global_features)
print(tmp.shape)
#fig, ax = plt.subplots()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set(title='PCA for LR3')
ax.scatter(tmp[:57][:,0], tmp[:57][:,1], tmp[:57][:,1], color='red', label='Clefable')
ax.scatter(tmp[58:105][:,0], tmp[58:105][:,1], tmp[58:105][:,1], color='green', label='Clefairy')
ax.scatter(tmp[106:156][:,0], tmp[106:156][:,1], tmp[106:156][:,1], color='blue', label='Ditto')
ax.scatter(tmp[157:209][:,0], tmp[157:209][:,1], tmp[157:209][:,1], color='yellow', label='Drowzee')
ax.scatter(tmp[210:261][:,0], tmp[210:261][:,1], tmp[210:261][:,1], color='pink', label='Pikachu')
ax.scatter(tmp[262:][:,0], tmp[262:][:,1], tmp[262:][:,1], color='black', label='Raichu')
ax.legend()
plt.show()
fig.savefig('PCA_LR3.png')

'''


#Start t-SNE;
seed = random.randint(0,300)
print(seed)
tsne_model = TSNE(n_components=2, perplexity=3, random_state=seed)
tmp = tsne_model.fit_transform(global_features)
print(tmp.shape)
fig, ax = plt.subplots()    #SAME as: fig = plt.figure()
#                                     ax = fig.add_subplot(111)
ax.set(title='TSNE for LR3')
ax.scatter(tmp[:57][:,0], tmp[:57][:,1], color='red', label='Chansey')
ax.scatter(tmp[58:105][:,0], tmp[58:105][:,1], color='green', label='Grimer')
ax.scatter(tmp[106:156][:,0], tmp[106:156][:,1], color='blue', label='Lickitung')
ax.scatter(tmp[157:209][:,0], tmp[157:209][:,1], color='yellow', label='Muk')
ax.scatter(tmp[210:261][:,0], tmp[210:261][:,1], color='pink', label='Slowbro')
ax.scatter(tmp[262:][:,0], tmp[262:][:,1], color='black', label='Slowpoke')
ax.legend()
plt.show()
fig.savefig('TSNE_LR3.png')



'''

Изменённый метод t-SNE для доп задания

#Start t-SNE;
seed = random.randint(0,300)
print(seed)
tsne_model = TSNE(n_components=3, perplexity=3, random_state=seed)
tmp = tsne_model.fit_transform(global_features)
print(tmp.shape)
#fig, ax = plt.subplots()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set(title='TSNE for LR3')
ax.scatter(tmp[:57][:,0], tmp[:57][:,1], tmp[:57][:,1], color='red', label='Clefable')
ax.scatter(tmp[58:105][:,0], tmp[58:105][:,1], tmp[58:105][:,1], color='green', label='Clefairy')
ax.scatter(tmp[106:156][:,0], tmp[106:156][:,1], tmp[106:156][:,1], color='blue', label='Ditto')
ax.scatter(tmp[157:209][:,0], tmp[157:209][:,1], tmp[157:209][:,1], color='yellow', label='Drowzee')
ax.scatter(tmp[210:261][:,0], tmp[210:261][:,1], tmp[210:261][:,1], color='pink', label='Pikachu')
ax.scatter(tmp[262:][:,0], tmp[262:][:,1], tmp[262:][:,1], color='black', label='Raichu')
ax.legend()
plt.show()
fig.savefig('TSNE_LR3.png')

'''
