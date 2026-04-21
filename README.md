# 📞 ChurnGuard — Прогноз оттока клиентов телеком-компании

**ML-приложение для прогнозирования вероятности ухода клиента (churn) с интерпретацией результатов через SHAP**

## О проекте

Этот проект решает задачу бинарной классификации для предсказания вероятности оттока клиента телеком-оператора на основе 19 признаков.

**Ключевые особенности:**
- **Полный ML-пайплайн**: от EDA до деплоя
- **Сравнение 3 моделей**: Logistic Regression, Random Forest, XGBoost
- **Тюнинг гиперпараметров**: GridSearchCV для XGBoost и LogisticRegression
- **Интерпретация**: SHAP-анализ для объяснения предсказаний
- **Web-интерфейс**: интерактивное приложение на Streamlit
- **Визуализация**: Plotly, Matplotlib, Seaborn

## Результаты моделей
| Модель | Метрики |
|--------|---------|
| **XGBoost (tuned)** | $Accuracy:$ 0.794, $Precision:$ 0.638, $Recall:$ 0.519, $F1:$ 0.572, $ROC-AUC:$ **0.838** |
| **Logistic Regression (tuned)** | $Accuracy:$ 0.794, $Precision:$ 0.624, $Recall:$ 0.564, $F1:$ **0.593**, $ROC-AUC:$ 0.835 |
| **Random Forest** | $Accuracy:$ 0.787, $Precision:$ 0.610, $Recall:$ 0.476, $F1:$ 0.536, $ROC-AUC:$ 0.811 |

## Технологии

| Категория | Инструменты |
|-----------|-------------|
| **Язык** | Python 3.9+ |
| **ML/DL** | scikit-learn, XGBoost |
| **Интерпретация** | SHAP |
| **Визуализация** | Plotly, Matplotlib, Seaborn |
| **Web-фреймворк** | Streamlit |
| **Обработка данных** | pandas, numpy |
| **Сериализация** | joblib |

## Установка

1. Клонируйте репозиторий

    ```bash
    git clone https://github.com/yourusername/churnguard.git
    cd churnguard
    ```

2. Создайте виртуальное окружение
    ```bash
    python -m venv venv
    # Windows
   
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3. Установите зависимости
    ```bash
    pip install -r requirements.txt
    ```

4. Скачайте данные

   Датасет telco_churn.csv можно скачать с Kaggle: Telco Customer Churn.
   Поместите файл в корень проекта или папку data/.


## Быстрый старт

Запуск Streamlit-приложения

```bash
   streamlit run app.py