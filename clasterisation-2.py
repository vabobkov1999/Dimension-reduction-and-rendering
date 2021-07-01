import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

data = [
    #Mathematicians
    "Mathematician", "John Forbes Nash Jr. (mathematician)", "Mahāvīra (mathematician)", "Jim Simons (mathematician)",
    "Mathematics", "A Mathematician's Apology", "Mikhael Gromov (mathematician)", "Robert Morris (mathematician)",
    "John J. O'Connor (mathematician)", "I Am a Mathematician", "Robin Wilson (mathematician)",
    "Recreational mathematician", "Indian mathematician)", "Srinivasa Ramanujan", "John Dee (mathematician)",

    #History of Asia
    "History of Asia", "History of Asian art", "History of East Asia", "History of Southeast Asia",
    "Outline of South Asian history", "History of Asian Americans", "History of Central Asia",
    "Genetic history of East Asians", "Slavery in Asia", "Asian immigration to the United States",
    "Military history of Asia", "North Asia", "History of printing in East Asia", "History of the Middle East",
    "Christianity in Asia",

    #Media
    "Otitis media", "Social media", "Mass media", "Streaming media", "Media (communication)",
    "Vice Media", "Media studies", "Influence of mass media", "Alternative media", "Media conglomerate",
    "Virgin Media", "Multi-media", "Digital media", "Gawker Media", "Media ethics"
]
print(data)

text_from_wiki = []

for x in data:
    text_from_wiki.append(wikipedia.page(title=x).content)

model = TfidfVectorizer(stop_words={'english'})
X = model.fit_transform(text_from_wiki)
print(X.shape)

#Start PCA
pca_model = PCA(n_components=2)
tmp = pca_model.fit_transform(X.toarray())
print(tmp.shape)
fig, ax = plt.subplots()    #SAME as: fig = plt.figure()
#                                     ax = fig.add_subplot(111)
ax.set(title='PCA for LR4')
ax.scatter(tmp[:14][:,0], tmp[:14][:,1], color='red', label='Mathematicians')
ax.scatter(tmp[15:30][:,0], tmp[15:30][:,1], color='green', label='History of Asia')
ax.scatter(tmp[31:][:,0], tmp[31:][:,1], color='blue', label='Media')
ax.legend()
plt.show()
fig.savefig('PCA_LR4.png')

#Start TSNE
tsne_model = TSNE(n_components=2, perplexity=10, random_state=10)
tmp = tsne_model.fit_transform(X.toarray())
print(tmp.shape)
fig, ax = plt.subplots()    #SAME as: fig = plt.figure()
#                                     ax = fig.add_subplot(111)
ax.set(title='TSNE for LR4')
ax.scatter(tmp[:14][:,0], tmp[:14][:,1], color='red', label='Mathematicians')
ax.scatter(tmp[15:30][:,0], tmp[15:30][:,1], color='green', label='History of Asia')
ax.scatter(tmp[31:][:,0], tmp[31:][:,1], color='blue', label='Media')
ax.legend()
plt.show()
fig.savefig('TSNE_LR4.png')
