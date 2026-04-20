import streamlit as st
import pandas as pd
import joblib

# Настройка страницы
st.set_page_config(
    page_title="ChurnGuard 📞️",
    page_icon="📞️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Загрузка модели
try:
    model = joblib.load('models/best_xgb_model.pkl')
except FileNotFoundError:
    st.error("❌ Модель не найдена. Убедитесь, что файл `models/best_xgb_model.pkl` существует.")
    st.stop()

# Список признаков
feature_names = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# Кодировки (как в ноутбуке)
cat_map = {
    'gender': {'Male': 1, 'Female': 0},
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'Yes': 1, 'No': 0},
    'PhoneService': {'Yes': 1, 'No': 0},
    'MultipleLines': {'Yes': 1, 'No': 0, 'No phone service': 0},
    'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
    'OnlineSecurity': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'OnlineBackup': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'DeviceProtection': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'TechSupport': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'StreamingTV': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'StreamingMovies': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'Contract': {'Month-to-month': 1, 'One year': 0, 'Two year': 2},
    'PaperlessBilling': {'Yes': 1, 'No': 0},
    'PaymentMethod': {
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    }
}

# Заголовок
st.markdown("<h1>📞 ChurnGuard</h1>", unsafe_allow_html=True)

# Форма ввода
with st.form("churn_form"):
    st.subheader("👤 Профиль клиента")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        tenure = st.slider("Месяцы в компании", min_value=0, max_value=72, value=1)
        monthly_charges = st.number_input("Ежемесячная плата ($)", min_value=0.0, value=50.0)
        total_charges = st.number_input("Общая сумма ($)", min_value=0.0, value=50.0)
    with col2:
        gender = st.selectbox("Пол", ["Male", "Female"])
        senior = st.selectbox("Пенсионер", ["Yes", "No"])
        partner = st.selectbox("Партнёр", ["Yes", "No"])
        dependents = st.selectbox("Иждивенцы", ["Yes", "No"])
        phone = st.selectbox("PhoneService", ["Yes", "No"])
        multiple_lines = st.selectbox("MultipleLines", ["Yes", "No", "No phone service"])
    with col3:
        internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Контракт", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("PaperlessBilling", ["Yes", "No"])
        payment = st.selectbox("Способ оплаты", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    submitted = st.form_submit_button("🔍 Прогнозировать риск")

# Обработка результата
if submitted:
    # Формируем input_dict
    input_dict = {
        'gender': cat_map['gender'][gender],
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': cat_map['Partner'][partner],
        'Dependents': cat_map['Dependents'][dependents],
        'tenure': tenure,
        'PhoneService': cat_map['PhoneService'][phone],
        'MultipleLines': cat_map['MultipleLines'][multiple_lines],
        'InternetService': cat_map['InternetService'][internet],
        'OnlineSecurity': cat_map['OnlineSecurity'][online_security],
        'OnlineBackup': cat_map['OnlineBackup'].get(online_security, 0),
        'DeviceProtection': cat_map['DeviceProtection'].get(online_security, 0),
        'TechSupport': cat_map['TechSupport'][tech_support],
        'StreamingTV': cat_map['StreamingTV'].get(online_security, 0),
        'StreamingMovies': cat_map['StreamingMovies'].get(online_security, 0),
        'Contract': cat_map['Contract'][contract],
        'PaperlessBilling': cat_map['PaperlessBilling'][paperless],
        'PaymentMethod': cat_map['PaymentMethod'][payment],
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    # Создаём DataFrame в правильном порядке
    input_df = pd.DataFrame([input_dict])[feature_names]

    # Предсказание
    prob = model.predict_proba(input_df)[0][1]  # P(Churn = Yes)
    pred = 1 if prob > 0.5 else 0

    # Вывод результата
    if pred == 1:
        st.error(f"⚠️ Высокий риск оттока: {prob:.1%}")
    else:
        st.success(f"✅ Низкий риск оттока: {prob:.1%}")

# Футер
st.markdown("---")
st.caption("📞 ChurnGuard | ML-система прогнозирования оттока | © 2026")