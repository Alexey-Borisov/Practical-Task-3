# Practical-Task-3
Практическое задание  3. Ансамбли алгоритмов. Веб-сервер. Композиции алгоритмов для решения задачи регрессии


## Запуск веб сервера

Есть два основных способа запустить веб-сервер.

Для первого способа необходимо скопировать репозиторий и из директории с Dockerfile выполнить следующие команды:
```
chmod +x ./scripts/build.sh
chmod +x ./scripts/run.sh
./scripts/build.sh
./scripts/run.sh
```
Второй способ подразумевает загрузку docker-контейнера с dockerhub. Это можно осуществить следующими командами:

```
docker pull borisovalexey/mmp_practicum_3:prac_3
docker run -p 5000:5000 -i borisovalexey/mmp_practicum_3:prac_3
```

При возникновении проблем можно попробовать запустить те же команды с правами администратора.
## Содержимое веб-сервера

На основной странице создания моделей пользователь имеет возможность выбрать тип модели (Случайный лес и Градиентный бустинг), а также
гиперпараметры для модели. Значение -1 для максимальной глубины дерева соответствует неограниченной глубине.
Имеется возможность указать имя столбца, являющегося целевой переменной (по умолчанию target).
Пользователю необходимо выбрать csv-файл, содержащий датасет, на котором он хочет обучить модель.
Также при желании можно указать датасет для валидации.

![alt text](https://github.com/Alexey-Borisov/Practical-Task-3/blob/main/assets/web_1.png?raw=true)

После обучения модели можно передать в виде csv-файла датасет содержащий данные аналогичные тем, что использовались для обучения, но без столбца
целевой переменной. После предсказания у пользователя есть возможность скачать предсказание модели в виде csv-файла.

![alt text](https://github.com/Alexey-Borisov/Practical-Task-3/blob/main/assets/web_2.png?raw=true)

Кроме этого пользователь может посотреть параметры модели, а также потери на валидационной выборке. Также имеется возможность загрузить более подробную
информацию о потерях на валидационной выборке. При необходимости получить информацию об изменении значений функции потерь на обучающей во время обучения можно в качестве валидационной выборки еще раз передать обучающую.

![alt text](https://github.com/Alexey-Borisov/Practical-Task-3/blob/main/assets/web_3.png?raw=true)
![alt text](https://github.com/Alexey-Borisov/Practical-Task-3/blob/main/assets/web_4.png?raw=true)

Если пользователь хочет обучить новую модель, он может приступить к этому нажав кнопку "Создание новой модели".


## Тестирование

Для проверки работоспособности веб-сервера предлагаются файлы train_example.csv, val_example.csv и test_example.csv. Целевой переменной для этого набора датасетов является столбец target.

