import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
import pickle

import json
from streamlit_lottie import st_lottie

path = "assets/lottie.json"
with open(path, "r") as file:
    url = json.load(file)


def load_clean_data() -> pd.DataFrame:
    """Load fetal health data and remove duplicates"""
    data = pd.read_csv("data/fetal_health.csv").drop_duplicates()
    return data


def add_sidebar():
    data = load_clean_data()

    st.sidebar.header("CTG Features")
    st.sidebar.subheader(
        "Predict the Fetal Health by selecting the following features:"
    )
    slider_labels = [
        ("Baseline Fetal Heart Rate", "baseline value"),
        ("Accelerations", "accelerations"),
        ("Fetal Movements per second", "fetal_movement"),
        ("Uterine Contractions per second", "uterine_contractions"),
        ("Light Depressions per second", "light_decelerations"),
        ("Severe Depressions per second", "severe_decelerations"),
        ("Prolonged Decelerations per second", "prolongued_decelerations"),
        ("Abnormal short term variability", "abnormal_short_term_variability"),
        ("Short term variability (mean)", "mean_value_of_short_term_variability"),
        (
            "Long term variability (Percentage)",
            "percentage_of_time_with_abnormal_long_term_variability",
        ),
        ("Long term variability (Mean)", "mean_value_of_long_term_variability"),
        ("histogram width", "histogram_width"),
        ("histogram min", "histogram_min"),
        ("histogram max", "histogram_max"),
        ("histogram number of peaks", "histogram_number_of_peaks"),
        ("histogram number of zeros", "histogram_number_of_zeroes"),
        ("histogram mode", "histogram_mode"),
        ("histogram mean", "histogram_mean"),
        ("histogram median", "histogram_median"),
        ("histogram variance", "histogram_variance"),
        ("histogram tendency", "histogram_tendency"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean()),
        )

    return input_dict


def predict_with_randomn_forest(input_data):
    """Predict the Fetal Health using Random Forest Classifier"""
    random_forest_model = pickle.load(open("model/RandomForestClassifier.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = random_forest_model.predict(input_array_scaled)

    st.subheader("Random Forest Classifier Prediction")
    st.write("The CTG suggests the fetus is : ")
    if prediction[0] == 1:
        st.write(
            "<span class = 'prediction normal' >Normal</span>", unsafe_allow_html=True
        )

    if prediction[0] == 2:
        st.write(
            "<span class = 'prediction suspect' >Suspect</span>", unsafe_allow_html=True
        )

    if prediction[0] == 3:
        st.write(
            "<span class = 'prediction pathological' >Pathological</span>",
            unsafe_allow_html=True,
        )

    st.write(
        "Probability of being Normal: ",
        round(random_forest_model.predict_proba(input_array_scaled)[0][0], 3),
    )
    st.write(
        "Probability of being Suspicious: ",
        round(random_forest_model.predict_proba(input_array_scaled)[0][1], 3),
    )
    st.write(
        "Probability of being Pathological: ",
        round(random_forest_model.predict_proba(input_array_scaled)[0][2], 3),
    )

    return random_forest_model


def predict_with_gbc(input_data):
    """Predict the Fetal Health using Gradient Boosting Classifier"""
    gbc_model = pickle.load(open("model/GradientBoostingClassifier.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = gbc_model.predict(input_array_scaled)

    st.subheader("Gradient Boosting Classifier Prediction")
    st.write("The CTG suggests the fetus is : ")
    if prediction[0] == 1:
        st.write(
            "<span class = 'prediction normal' >Normal</span>", unsafe_allow_html=True
        )

    if prediction[0] == 2:
        st.write(
            "<span class = 'prediction suspect' >Suspect</span>", unsafe_allow_html=True
        )

    if prediction[0] == 3:
        st.write(
            "<span class = 'prediction pathological' >Pathological</span>",
            unsafe_allow_html=True,
        )

    st.write(
        "Probability of being Normal: ",
        round(gbc_model.predict_proba(input_array_scaled)[0][0], 3),
    )
    st.write(
        "Probability of being Suspicious: ",
        round(gbc_model.predict_proba(input_array_scaled)[0][1], 3),
    )
    st.write(
        "Probability of being Pathological: ",
        round(gbc_model.predict_proba(input_array_scaled)[0][2], 3),
    )

    return gbc_model


def calculate_errors(y_test, y_pred):
    mse_model = round(mean_squared_error(y_test, y_pred), 3)
    rmse_model = round(np.sqrt(mse_model), 3)
    return mse_model, rmse_model


def calculate_R2(model, X_train, y_train, X_test, y_test):
    scores_model_train = model.score(X_train, y_train)
    R2_train = round(scores_model_train, 3)
    scores_model_test = model.score(X_test, y_test)
    R2_test = round(scores_model_test, 3)
    return R2_train, R2_test


def plot_confusion_matrix(y_test, y_pred):
    # print classification report
    classificationReport = classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Suspicious", "Pathological"],
        output_dict=True,
    )

    # print confusion matrix
    confusionMatrix = confusion_matrix(y_test, y_pred)

    # plot confusion matrix as a heatmap
    ax = plt.subplot()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, ax=ax, cmap="BuGn")

    # labels, title and ticks
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(["Normal", "Suspect", "Pathological"])
    ax.yaxis.set_ticklabels(["Normal", "Suspect", "Pathological"])

    fig = ax.get_figure()
    # set the figure size to 15,10
    fig.set_size_inches(15, 10)
    return fig, classificationReport, confusionMatrix


def evaluate_model(model, X_train, y_train, X_test, y_test, y_pred):
    mse, rmse = calculate_errors(y_test, y_pred)
    R2_train, R2_test = calculate_R2(model, X_train, y_train, X_test, y_test)
    figure, classificationReport, confusionMatrix = plot_confusion_matrix(
        y_test, y_pred
    )
    return mse, rmse, R2_train, R2_test, figure, classificationReport, confusionMatrix


def main():
    st.set_page_config(
        page_title="Fetal Health",
        page_icon=":baby:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with open("assets/styles.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    data = load_clean_data()
    input_data = add_sidebar()
    # Heading and introduction
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st_lottie(
                url,
                height=300,
                width=300,
                key="mother",
                loop=True,
                quality="high",
            )
    with st.container():

        st.header("Fetal Health Prediction with Machine Learning")
        st.write(
            "Understanding fetal health is crucial during pregnancy, and one of the key tools used for this purpose is Cardiotocography (CTG). CTG is a non-invasive monitoring technique that tracks the fetal heart rate (FHR) and uterine contractions to assess the well-being of the unborn baby. By analyzing patterns and fluctuations in the fetal heart rate recorded over time, healthcare professionals can gain valuable insights into the fetal condition, oxygenation levels, and potential signs of distress. Additionally, CTG data provides important indicators of fetal health, helping to identify abnormalities or deviations from normal patterns that may require further evaluation or intervention. In summary, CTG plays a vital role in prenatal care by providing valuable information about fetal well-being and helping to ensure a safe and healthy pregnancy for both mother and baby."
        )
    # Model Predictions
    with st.container():
        st.header("Fetal Health Prediction based on CTG Features Selected")
        col1, col2 = st.columns([1, 1])
        with col1:
            rf_model = predict_with_randomn_forest(input_data)

        with col2:
            gbc_model = predict_with_gbc(input_data)
    # Model efficiency
    with st.container():
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        X = data.drop("fetal_health", axis=1)
        y = data["fetal_health"]

        X = scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        st.subheader("Model Efficiency")
        tab1, tab2 = st.tabs(
            ["Random Forest Classifier", "Gradient Boosting Classifier"]
        )

        with tab1:
            y_pred_rf = rf_model.predict(X_test)
            (
                mse_rf,
                rmse_rf,
                R2_train_rf,
                R2_test_rf,
                plot_rf,
                classificationReport_rf,
                confusionMatrix_rf,
            ) = evaluate_model(rf_model, X_train, y_train, X_test, y_test, y_pred_rf)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.dataframe(pd.DataFrame(classificationReport_rf).transpose())

                st.write("Mean squared error for Random Forest Classifier: ", mse_rf)
                st.write(
                    "Root means squared error for Random Forest Classifier: ", rmse_rf
                )
                st.write(
                    "R^2, Coefficienct of determination for training data for Random Forest Classifier: ",
                    R2_train_rf,
                )
                st.write(
                    "R^2, Coefficienct of determination for testing data for Random Forest Classifier: ",
                    R2_test_rf,
                )

            with col2:
                st.pyplot(plot_rf)

        with tab2:
            y_pred_gbc = gbc_model.predict(X_test)
            (
                mse_gbc,
                rmse_gbc,
                R2_train_gbc,
                R2_test_gbc,
                plot_gbc,
                classificationReport_gbc,
                confusionMatrix_gbc,
            ) = evaluate_model(gbc_model, X_train, y_train, X_test, y_test, y_pred_gbc)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.dataframe(pd.DataFrame(classificationReport_gbc).transpose())

                st.write(
                    "Mean squared error for Gradient Boosting Classifier: ", mse_gbc
                )
                st.write(
                    "Root means squared error for Gradient Boosting Classifier: ",
                    rmse_gbc,
                )
                st.write(
                    "R^2, Coefficienct of determination for training data for Gradient Boosting Classifier: ",
                    R2_train_gbc,
                )
                st.write(
                    "R^2, Coefficienct of determination for testing data for Gradient Boosting Classifier: ",
                    R2_test_gbc,
                )

            with col2:
                st.pyplot(plot_gbc)


if __name__ == "__main__":
    main()
