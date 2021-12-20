# Practical-Task-3
Практическое задание  3. Ансамбли алгоритмов. Веб-сервер. Композиции алгоритмов для решения задачи регрессии


## Запуск

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
