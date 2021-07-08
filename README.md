# Лабораторная работа 6

| 🔢  | Ход работы   | ℹ️ |
| ------------- | ------------- |------------- |
| 1️⃣ | На основе признаков из заданий [№3](https://github.com/vabobkov1999/Image-Training) и [№4]https://github.com/vabobkov1999/Clustering-Wikipedia-Articles) построить PCA и TSNE распределения в двумерном пространстве.| ✅ |
| 2️⃣ | Визуализировать результат с помощью scatter графика. |✅  |

Цель работы
------------
С помощью python3.8 и методов PCA и TNCE визуализировать фитчи из многомерной плоскости на двумерную плоскость, которые были получены в прошлых заданиях(см. таблицу выше).

Выполнение работы
-----------------


Для реализации поставленной задачи воспользуемся двумя методами указанными выше.
Первый из них это метод PCA находит наилучшую n-мерную плоскость, используется при компрессии изображений и видео, подавлении шума на изображениях и конечно визуализации данных. Работает следующим образом:
* Находит базис в пространстве меньшей размерности (преобразует данные в
линейно независимые вектора)
* Первый компонент имеет максимальную дисперсию, вторая – вторая по
величине и т.д.


Второй это метод t-SNE работает по принципу сохранения расстояний между двумя точка, используется для исследования данных и визуализации многомерных данных. Работает следующим образом:
* При уменьшении размерности старается сохранить окружение каждой точки. Это приводит к сохранению локальной структуры
* Сначала t-SNE создаёт распределение вероятностей по парам объектов высокой
размерности таким образом, что похожие объекты будут выбраны с большой
вероятностью, в то время как вероятность выбора непохожих точек будет мала.
* Затем t-SNE определяет похожее распределение вероятностей по точкам в пространстве малой размерности и минимизирует расстояние Кульбака — Лейблера между двумя распределениями с учётом положения точек.

Чтобы визуализировать признаки на плоскости нужно уменьшить размерность
векторой до двух компонентов.


Работа сделана на основании двух предыдущих


### Для работы Clustering-Wikipedia-Articles


В файле clasterisation-2.py содержится программа для кластеризации статей из Wikipedia, с внесёнными изменениями, которые позволяют визуализировать набор фитч на двумерной плоскости.
До момента #Start PCA работы полностью совпадают.
С помощью метода TfidfVectorizer достаём фитчи из нашего сета статей и запоминаем их (Фрагмент кода представлен ниже).

```python
model = TfidfVectorizer(stop_words={'english'})
X = model.fit_transform(text_from_wiki)
print(X.shape)
```

Метод PCA ищет двухмерную плоскость для нашего многомерного пространства. С помощью функции fit_transform запихиваем данные из X, то есть  вектора с фитчами в tmp, который в конечном итоге у нас уже запоминает двухмерные вектора (Фрагмент кода представлен ниже).

```python
pca_model = PCA(n_components=2)
tmp = pca_model.fit_transform(X.toarray())
print(tmp.shape)
```
В результате мы получаем следующий график:

<p align="center">
  <img src="https://raw.githubusercontent.com/vabobkov1999/Dimension-reduction-and-rendering/master/PCA_LR4.png" />
</p>

С методом t-SNE идея таже, создается одноимённая модель, только тут параметры n_components размерность, perplexity - минимальное расстояние между двумя точками, random_state это константа, которая рандомно выбирает точку с которой начинает обработку. Аналогично с помощью функции fit_transform запихиваем данные из X, то есть  вектора с фитчами в tmp, который в конечном итоге у нас уже запоминает двухмерные вектора (Фрагмент кода представлен ниже).

```python
tsne_model = TSNE(n_components=2, perplexity=10, random_state=10)
tmp = tsne_model.fit_transform(X.toarray())
print(tmp.shape)
```
В результате мы получаем следующий график:

<p align="center">
  <img src="https://raw.githubusercontent.com/vabobkov1999/Dimension-reduction-and-rendering/master/TSNE_LR4.png" />
</p>


### Для работы Image-Training

Для этой реализации этого задания необходимо два файла testing.py и training.py.

Файл training.py полностью совпадает с предыдущей работой и предназначен для получения фитч из картинок и сохраняет результат в файл h5.
В Файл testing.py находится изменённый под задания главный код.

Методы PCA и t-SNE точно так же работают как и в работе Clustering-Wikipedia-Articles.

Фрагмент кода и график для метода PCA представлены ниже:
```python
pca_model = PCA(n_components=3, random_state=0)
tmp = pca_model.fit_transform(global_features)
print(tmp.shape)
```
<p align="center">
  <img src="https://raw.githubusercontent.com/vabobkov1999/Dimension-reduction-and-rendering/master/PCA_LR3_2D.png" />
</p>


Фрагмент кода и график для метода t-SNE представлены ниже:

```python
seed = random.randint(0,300)
print(seed)
tsne_model = TSNE(n_components=3, perplexity=3, random_state=seed)
tmp = tsne_model.fit_transform(global_features)
print(tmp.shape)
```
<p align="center">
  <img src="https://raw.githubusercontent.com/vabobkov1999/Dimension-reduction-and-rendering/master/TSNE_LR3_2D.png" />
</p>

### Дополнительное задание:

В качестве доп.задания мне было предложено перестроить графики из двухмерной плоскости в трёхмерную.Такие преобразования были сделаны для работы [№3](https://github.com/vabobkov1999/Image-Training).

Для получения трехмерного графика были сделаны следующие преобразования:

* Параметр n_components стал равен трём
* Была добавлена строка ax = fig.add_subplot(111, projection='3d') вместе с новым параметром projection='3d'
* В ax.scatter был добавлен ещё один интервал tmp[k:n] (где k и n, некоторые числа которые совпадают с позициями фотографий в dataset)

#### По итогу были получены следующие графики:

<p align="center">
  <img src="https://raw.githubusercontent.com/vabobkov1999/Dimension-reduction-and-rendering/master/PCA_LR3.png" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/vabobkov1999/Dimension-reduction-and-rendering/master/TSNE_LR3.png" />
</p>

Замечание:
-----------
Для лабораторной работы [№3](https://github.com/vabobkov1999/Image-Training) методы PCA и t-NSE работают ужасно. Это происходит из-за того, что некачественно выбраны фитчи. Вследствие чего они не подходят для кластеризации и построения графиков.Я пытался отрегулировать параметры фитч, но лучше не получилось.

#### Как должен выглядеть график методов PCA и t-NSE:

<p align="center">
  <img src="https://raw.githubusercontent.com/vabobkov1999/Dimension-reduction-and-rendering/master/Goog_Job.20.58.png" height="480" width="640"/>
</p>

#### Как график выглядит на самом деле:


<p align="center">
  <img src="https://raw.githubusercontent.com/vabobkov1999/Dimension-reduction-and-rendering/master/PCA_LR3_2D.png" />
</p>

<p align="center">
  <img src="https://memepedia.ru/wp-content/uploads/2020/09/b2b7c451cbddc634ecc0dc37031fb4d6.jpg" height="480" width="640"/>
</p>

Так же напоминаю для тех кому интересно выполнить задание самому или протестировать данную программу, то прошу перейти [сюда](https://drive.google.com/drive/folders/1b_molbj8z6JhHV6r178AeI1XpQezehsm?usp=sharing "Практикум по машинному обучению")
