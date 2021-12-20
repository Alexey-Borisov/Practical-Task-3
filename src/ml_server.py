import pickle

import pandas as pd
import numpy as np

from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for, send_file
from flask import render_template, redirect
from wtforms.validators import DataRequired, InputRequired
from wtforms import StringField, SubmitField, FileField, SelectField

from ensembles import GradientBoostingMSE, RandomForestMSE

app = Flask(__name__, template_folder='templates')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'bugslover'
Bootstrap(app)


class MainPageForm(FlaskForm):
    new_model_button = SubmitField("Создать новую модель", validators=None)


class ModelCreationForm(FlaskForm):
    model = SelectField("Модель", choices=[(1, "Случайный лес"), (2, "Градиентный бустинг")],
                        validators=[InputRequired()])
    n_estimators = StringField("Количество деревьев в ансамбле", validators=[DataRequired()])
    max_depth = StringField("Максимальная глубина дерева", validators=[DataRequired()])
    feature_subsample_size = StringField("Размерность подвыборки признаков", validators=[DataRequired()])
    learning_rate = StringField("Темп обучения (только для градиентного бустинга)", validators=[DataRequired()])
    train_dataset_file = FileField("Загрузка датасета для обучения", validators=[DataRequired()])
    val_dataset_file = FileField("Загрузка датасета для валидации", validators=None)
    target_column = StringField('Введите название столбца, содержащего целевую переменную', validators=[DataRequired()])
    fit_button = SubmitField("Обучение", validators=None)


class PredictionForm(FlaskForm):
    test_dataset_file = FileField("Загрузка датасета для предсказания", validators=None)
    predict_button = SubmitField("Предсказание", validators=None)
    val_result = SubmitField("Результат валидации", validators=None)
    model_info = SubmitField("Информация о модели", validators=None)
    new_model = SubmitField("Создание новой модели", validators=None)


class ValInfoForm(FlaskForm):
    load_val_info = SubmitField("Загрузить информацию о потерях после каждой итерации", validators=None)
    back_button = SubmitField("Назад", validators=None)


class PredInfoForm(FlaskForm):
    load_pred = SubmitField("Загрузить предсказание", validators=None)
    back_button = SubmitField("Назад", validators=None)


class ModelInfoForm(FlaskForm):
    back_button = SubmitField("Назад", validators=None)


def fit(train_filename, val_filename, model_type, target_column, n_estimators,
        max_depth, feature_subsample_size, learning_rate):

    train_dataset = pd.read_csv(train_filename)
    data_columns = train_dataset.columns.drop([target_column])
    train_data = train_dataset[data_columns]
    train_target = train_dataset[target_column]
    X_train, y_train = np.array(train_data, dtype=float), np.array(train_target, dtype=float)

    # validation data
    if val_filename is not None:
        val_dataset = pd.read_csv(val_filename)
        val_data = val_dataset[data_columns]
        val_target = val_dataset[target_column]

        X_val, y_val = np.array(val_data, dtype=float), np.array(val_target, dtype=float)
    else:
        X_val, y_val = None, None

    if model_type == "1":
        model = RandomForestMSE(n_estimators=n_estimators, max_depth=max_depth,
                                feature_subsample_size=feature_subsample_size)
    else:
        model = GradientBoostingMSE(n_estimators=n_estimators, max_depth=max_depth,
                                    feature_subsample_size=feature_subsample_size, learning_rate=learning_rate)
    val_loss = model.fit(X_train, y_train, X_val, y_val)

    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    pd.DataFrame(model.history['pred_loss'], columns=['RMSE']).to_csv('validation_losses.csv')
    return val_loss


def parameters_parsing(model_form):
    # train file
    train_file = request.files[model_form.train_dataset_file.name]
    train_file.save(train_file.filename)
    train_filename = train_file.filename

    # val file
    val_file = request.files[model_form.val_dataset_file.name]
    if val_file.filename != '':
        val_file.save(val_file.filename)
        val_filename = val_file.filename
    else:
        val_filename = None

    # target column
    target_column = model_form.target_column.data

    # model type
    model_type = model_form.model.data

    # model parameters
    n_estimators = int(model_form.n_estimators.data)
    max_depth = int(model_form.max_depth.data)
    if max_depth == -1:
        max_depth = None
    feature_subsample_size = float(model_form.feature_subsample_size.data)
    learning_rate = float(model_form.learning_rate.data)

    params = {'train_filename': train_filename, 'val_filename': val_filename,
              'target_column': target_column, 'model_type': model_type, 'n_estimators': n_estimators,
              'max_depth': max_depth, 'feature_subsample_size': feature_subsample_size,
              'learning_rate': learning_rate}
    print(params)
    return params


@app.route('/', methods=['GET', 'POST'])
def main_page():
    form = MainPageForm()
    if form.validate_on_submit():
        return redirect(url_for('new_model'))
    return render_template('main_page.html', form=form)


@app.route('/new_model', methods=['GET', 'POST'])
def new_model():

    if request.method == "GET":
        model_form = ModelCreationForm()
        model_form.n_estimators.data = "200"
        model_form.max_depth.data = "-1"
        model_form.feature_subsample_size.data = "0.33"
        model_form.learning_rate.data = "0.1"
        model_form.target_column.data = "target"
    else:
        model_form = ModelCreationForm()

    if model_form.validate_on_submit():
        if model_form.train_dataset_file.data.filename == '':
            return redirect(url_for('new_model'))
        try:
            params = parameters_parsing(model_form)
            with open("model_params.pkl", "wb") as params_file:
                pickle.dump(params, params_file)
            val_loss = fit(**params)
        except Exception:
            return redirect(url_for('new_model', message="Что-то пошло не так"))
        return redirect(url_for('prediction', val_loss=val_loss))
    return render_template('new_model_page.html', form=model_form)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():

    pred_form = PredictionForm()
    val_loss = request.args.get("val_loss")

    if pred_form.validate_on_submit():
        if pred_form.val_result.data:
            return redirect(url_for('val_information', val_loss=val_loss))
        if pred_form.model_info.data:
            return redirect(url_for('model_info', val_loss=val_loss))
        if pred_form.new_model.data:
            return redirect(url_for('new_model'))
        if pred_form.test_dataset_file.data.filename == '':
            return redirect(url_for('prediction', message="Нужно выбрать файл", val_loss=val_loss))
        else:
            try:
                with open("model.pkl", "rb") as model_file:
                    model = pickle.load(model_file)
                # test file
                test_file = request.files[pred_form.test_dataset_file.name]
                test_file.save(test_file.filename)
                test_filename = test_file.filename
                X_test = np.array(pd.read_csv(test_filename))
                pred = model.predict(X_test)
                pred_df = pd.DataFrame(pred, columns=["Prediction"])
                pred_df.to_csv("prediction.csv")
            except Exception:
                return redirect(url_for('prediction', message="Что-то пошло не так", val_loss=val_loss))
            return redirect(url_for('pred_information', val_loss=val_loss))
    return render_template('predict_page.html', form=pred_form, val_loss=val_loss)


@app.route('/pred_information', methods=['GET', 'POST'])
def pred_information():

    pred_form = PredInfoForm()
    val_loss = request.args.get("val_loss")
    if pred_form.validate_on_submit():
        if pred_form.back_button.data:
            return redirect(url_for('prediction', val_loss=val_loss))
        if pred_form.load_pred.data:
            return send_file("prediction.csv", as_attachment=True)
    return render_template('pred_info_page.html', form=pred_form, val_loss=val_loss)


@app.route('/val_information', methods=['GET', 'POST'])
def val_information():

    val_form = ValInfoForm()
    val_loss = request.args.get("val_loss")

    if val_form.validate_on_submit():
        if val_form.back_button.data:
            return redirect(url_for('prediction', val_loss=val_loss))
        if val_form.load_val_info.data:
            return send_file("validation_losses.csv", as_attachment=True)
    return render_template('val_info_page.html', form=val_form, val_loss=val_loss)


@app.route('/model_info', methods=['GET', 'POST'])
def model_info():

    model_info_form = ModelInfoForm()
    val_loss = request.args.get("val_loss")

    if model_info_form.back_button.data:
        return redirect(url_for('prediction', val_loss=val_loss))

    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    model_type = "Градиентный бустинг" if hasattr(model, 'learning_rate') else "Случайный лес"
    n_estimators = model.n_estimators
    max_depth = model.max_depth if model.max_depth is not None else "Неограничена"
    feature_subsample_size = model.feature_subsample_size
    learning_rate = model.learning_rate if hasattr(model, 'learning_rate') else "Отсутствует"

    with open("model_params.pkl", "rb") as params_file:
        params = pickle.load(params_file)
    return render_template('model_info_page.html', form=model_info_form, val_loss=val_loss,
                           model_type=model_type, n_estimators=n_estimators, max_depth=max_depth,
                           feature_subsample_size=feature_subsample_size, learning_rate=learning_rate,
                           target_column=params['target_column'])
