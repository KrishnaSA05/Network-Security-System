"""
Streamlit Web App for Phishing Detection
Interactive UI for testing the trained model
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Page configuration
st.set_page_config(
    page_title="Phishing Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .safe {
        color: #28a745;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .phishing {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_metadata():
    """Load the best model and its metadata"""
    try:
        # Use absolute path from project root
        model_path = PROJECT_ROOT / "src" / "models" / "saved_models" / "best_model.pkl"
        metadata_path = PROJECT_ROOT / "src" / "models" / "saved_models" / "best_model_metadata.json"

        st.info(f"Looking for model at: {model_path}")

        if not model_path.exists():
            st.error(f"❌ Model not found at: {model_path}")
            st.info("💡 Please run the training pipeline first: `python main.py`")
            st.warning(f"Project root: {PROJECT_ROOT}")
            return None, None

        model = joblib.load(model_path)

        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            st.warning("⚠️ Metadata file not found. Using default values.")
            metadata = {
                'model_name': 'unknown',
                'model_type': type(model).__name__,
                'all_metrics': {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'roc_auc': 0.0
                },
                'timestamp': 'Unknown'
            }

        return model, metadata

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None


def engineer_features(df):
    """Apply feature engineering to input data"""
    # SSLDomainTrust
    df['SSLDomainTrust'] = df['SSLfinal_State'] * df['Domain_registeration_length']

    # URLSuspicionScore
    df['URLSuspicionScore'] = (
        df['having_IP_Address'] + df['URL_Length'] + 
        df['having_At_Symbol'] + df['Prefix_Suffix']
    ) / 4.0

    # ContentCredibility
    df['ContentCredibility'] = (
        df['URL_of_Anchor'] + df['Links_in_tags'] + df['SFH']
    ) / 3.0

    # DomainReputation
    df['DomainReputation'] = (
        df['age_of_domain'] + df['DNSRecord'] + df['web_traffic']
    ) / 3.0

    # SecurityFeaturesCount
    security_features = ['SSLfinal_State', 'Domain_registeration_length',
                        'HTTPS_token', 'age_of_domain', 'DNSRecord']
    df['SecurityFeaturesCount'] = df[security_features].apply(
        lambda x: (x == 1).sum(), axis=1
    )

    # SuspiciousFeaturesCount
    suspicious_features = ['having_IP_Address', 'having_At_Symbol', 'Prefix_Suffix',
                          'having_Sub_Domain', 'Request_URL', 'Abnormal_URL', 'Redirect']
    df['SuspiciousFeaturesCount'] = df[suspicious_features].apply(
        lambda x: (x == -1).sum(), axis=1
    )

    # SSLAnchorInteraction
    df['SSLAnchorInteraction'] = df['SSLfinal_State'] * df['URL_of_Anchor']

    # Drop features that were removed during training
    drop_features = ['popUpWidnow', 'Favicon', 'port']
    for feat in drop_features:
        if feat in df.columns:
            df = df.drop(columns=[feat])

    return df


def create_prediction_gauge(probability):
    """Create a gauge chart for prediction probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Legitimacy Score", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#ffcccc'},
                {'range': [30, 70], 'color': '#ffffcc'},
                {'range': [70, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🔒 Phishing Detection System</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Load model
    model, metadata = load_model_and_metadata()

    if model is None:
        st.stop()

    # Sidebar - Model Info
    with st.sidebar:
        st.header("📊 Model Information")

        if metadata:
            st.metric("Model Type", metadata['model_name'].upper())
            st.metric("Accuracy", f"{metadata['all_metrics']['accuracy']:.2%}")
            st.metric("Precision", f"{metadata['all_metrics']['precision']:.2%}")
            st.metric("Recall", f"{metadata['all_metrics']['recall']:.2%}")
            st.metric("F1-Score", f"{metadata['all_metrics']['f1_score']:.2%}")
            st.metric("ROC-AUC", f"{metadata['all_metrics']['roc_auc']:.2%}")

            st.markdown("---")
            st.caption(f"Trained: {metadata['timestamp'][:10]}")

        st.markdown("---")

        # Navigation
        st.header("🧭 Navigation")
        page = st.radio("", ["🔍 Single Prediction", "📁 Batch Prediction", "📈 Model Analytics"])

    # Main content based on page selection
    if page == "🔍 Single Prediction":
        show_single_prediction_page(model, metadata)
    elif page == "📁 Batch Prediction":
        show_batch_prediction_page(model, metadata)
    else:
        show_analytics_page(metadata)


def show_single_prediction_page(model, metadata):
    """Single URL prediction interface"""
    st.header("🔍 Single URL Analysis")

    # Input method selection
    input_method = st.radio("Select Input Method:", ["Manual Entry", "Quick Presets"])

    if input_method == "Quick Presets":
        preset = st.selectbox("Choose a preset:", [
            "Legitimate Website (All Positive)",
            "Suspicious Website (Mixed)",
            "Phishing Website (All Negative)"
        ])

        if preset == "Legitimate Website (All Positive)":
            features = {feat: 1 for feat in get_feature_names()}
            features['Redirect'] = 0
        elif preset == "Phishing Website (All Negative)":
            features = {feat: -1 for feat in get_feature_names()}
            features['Redirect'] = 1
        else:  # Suspicious
            features = {feat: 0 for feat in get_feature_names()}

    else:  # Manual Entry
        st.subheader("Enter URL Features")

        # Organize features into categories
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🌐 URL-Based Features**")
            having_IP_Address = st.select_slider("Having IP Address", [-1, 0, 1], value=1)
            URL_Length = st.select_slider("URL Length", [-1, 0, 1], value=1)
            Shortining_Service = st.select_slider("Shortening Service", [-1, 1], value=1)
            having_At_Symbol = st.select_slider("Having @ Symbol", [-1, 1], value=1)
            double_slash_redirecting = st.select_slider("Double Slash Redirect", [-1, 1], value=1)
            Prefix_Suffix = st.select_slider("Prefix/Suffix (-)", [-1, 1], value=1)
            having_Sub_Domain = st.select_slider("Subdomain Count", [-1, 0, 1], value=1)
            Abnormal_URL = st.select_slider("Abnormal URL", [-1, 1], value=1)
            Redirect = st.select_slider("Redirect Count", [0, 1], value=0)
            HTTPS_token = st.select_slider("HTTPS in Domain", [-1, 1], value=1)

        with col2:
            st.markdown("**🔐 Security Features**")
            SSLfinal_State = st.select_slider("SSL Certificate", [-1, 0, 1], value=1)
            Domain_registeration_length = st.select_slider("Domain Age", [-1, 1], value=1)
            age_of_domain = st.select_slider("Age of Domain", [-1, 1], value=1)
            DNSRecord = st.select_slider("DNS Record", [-1, 1], value=1)

            st.markdown("**🎨 HTML/JS Features**")
            Favicon = st.select_slider("Favicon", [-1, 1], value=1)
            port = st.select_slider("Non-standard Port", [-1, 1], value=1)
            Request_URL = st.select_slider("External Request URL", [-1, 1], value=1)
            URL_of_Anchor = st.select_slider("Anchor URL", [-1, 0, 1], value=1)
            Links_in_tags = st.select_slider("Links in Tags", [-1, 0, 1], value=1)
            SFH = st.select_slider("Server Form Handler", [-1, 0, 1], value=1)

        with col3:
            st.markdown("**📊 Web Metrics**")
            web_traffic = st.select_slider("Web Traffic Rank", [-1, 0, 1], value=1)
            Page_Rank = st.select_slider("Page Rank", [-1, 1], value=1)
            Google_Index = st.select_slider("Google Index", [-1, 1], value=1)
            Links_pointing_to_page = st.select_slider("Backlinks", [-1, 0, 1], value=1)
            Statistical_report = st.select_slider("Statistical Report", [-1, 1], value=1)

            st.markdown("**⚠️ Behavioral Features**")
            Submitting_to_email = st.select_slider("Submit to Email", [-1, 1], value=1)
            on_mouseover = st.select_slider("On Mouse Over", [-1, 1], value=1)
            RightClick = st.select_slider("Right Click Disabled", [-1, 1], value=1)
            popUpWidnow = st.select_slider("Pop-up Window", [-1, 1], value=1)
            Iframe = st.select_slider("Iframe", [-1, 1], value=1)

        features = {
            'having_IP_Address': having_IP_Address,
            'URL_Length': URL_Length,
            'Shortining_Service': Shortining_Service,
            'having_At_Symbol': having_At_Symbol,
            'double_slash_redirecting': double_slash_redirecting,
            'Prefix_Suffix': Prefix_Suffix,
            'having_Sub_Domain': having_Sub_Domain,
            'SSLfinal_State': SSLfinal_State,
            'Domain_registeration_length': Domain_registeration_length,
            'Favicon': Favicon,
            'port': port,
            'HTTPS_token': HTTPS_token,
            'Request_URL': Request_URL,
            'URL_of_Anchor': URL_of_Anchor,
            'Links_in_tags': Links_in_tags,
            'SFH': SFH,
            'Submitting_to_email': Submitting_to_email,
            'Abnormal_URL': Abnormal_URL,
            'Redirect': Redirect,
            'on_mouseover': on_mouseover,
            'RightClick': RightClick,
            'popUpWidnow': popUpWidnow,
            'Iframe': Iframe,
            'age_of_domain': age_of_domain,
            'DNSRecord': DNSRecord,
            'web_traffic': web_traffic,
            'Page_Rank': Page_Rank,
            'Google_Index': Google_Index,
            'Links_pointing_to_page': Links_pointing_to_page,
            'Statistical_report': Statistical_report
        }

    # Predict button
    if st.button("🔍 Analyze URL", type="primary", use_container_width=True):
        # Create DataFrame
        df = pd.DataFrame([features])

        # Apply feature engineering
        df = engineer_features(df)

        # Make prediction
        prediction = model.predict(df)[0]

        # Get probability if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0]
            # Handle different model outputs
            if prediction in [0, 1]:  # XGBoost case
                probability_legitimate = proba[1] if prediction == 1 else proba[0]
            else:  # -1, 1 case
                probability_legitimate = proba[1] if len(proba) == 2 else 0.5
        else:
            probability_legitimate = 1.0 if prediction == 1 else 0.0

        # Display results
        st.markdown("---")
        st.header("📊 Analysis Results")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # Gauge chart
            fig = create_prediction_gauge(probability_legitimate)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Prediction")
            if prediction == 1 or prediction == 0:  # Handle both cases
                if prediction == 1:
                    st.markdown('<p class="safe">✅ LEGITIMATE</p>', unsafe_allow_html=True)
                    st.success("This website appears safe")
                else:
                    st.markdown('<p class="phishing">⚠️ PHISHING</p>', unsafe_allow_html=True)
                    st.error("This website appears malicious")
            else:  # prediction == -1
                st.markdown('<p class="phishing">⚠️ PHISHING</p>', unsafe_allow_html=True)
                st.error("This website appears malicious")

        with col3:
            st.markdown("### Confidence")
            confidence = max(probability_legitimate, 1 - probability_legitimate)
            st.metric("Confidence", f"{confidence:.1%}")
            st.metric("Model", metadata['model_name'].upper() if metadata else "Unknown")

        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            st.markdown("---")
            st.subheader("🔑 Top 10 Most Important Features")

            importances = model.feature_importances_
            feature_names = df.columns.tolist()

            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(10)

            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Blues')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


def show_batch_prediction_page(model, metadata):
    """Batch prediction from CSV file"""
    st.header("📁 Batch URL Analysis")

    st.info("Upload a CSV file containing URL features for batch prediction")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)

            st.success(f"✅ Loaded {len(df)} records")

            # Show preview
            with st.expander("📋 View Data Preview"):
                st.dataframe(df.head(10))

            if st.button("🚀 Run Batch Prediction", type="primary"):
                # Apply feature engineering
                df_processed = engineer_features(df.copy())

                # Make predictions
                predictions = model.predict(df_processed)

                # Get probabilities
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(df_processed)
                    df['Probability_Legitimate'] = probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities[:, 0]
                    df['Confidence'] = probabilities.max(axis=1)

                df['Prediction'] = predictions
                df['Prediction_Label'] = df['Prediction'].apply(
                    lambda x: 'Legitimate' if x == 1 or x == 0 else 'Phishing'
                )

                # Summary statistics
                st.markdown("---")
                st.header("📊 Batch Results Summary")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total URLs", len(df))
                with col2:
                    legitimate_count = ((predictions == 1) | (predictions == 0)).sum()
                    st.metric("✅ Legitimate", legitimate_count, 
                             delta=f"{legitimate_count/len(df)*100:.1f}%")
                with col3:
                    phishing_count = (predictions == -1).sum()
                    st.metric("⚠️ Phishing", phishing_count,
                             delta=f"{phishing_count/len(df)*100:.1f}%", delta_color="inverse")
                with col4:
                    avg_confidence = df['Confidence'].mean() if 'Confidence' in df.columns else 0
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")

                # Results table
                st.markdown("---")
                st.subheader("🔍 Detailed Results")
                st.dataframe(df[['Prediction_Label', 'Probability_Legitimate', 'Confidence']].head(20))

                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Results",
                    data=csv,
                    file_name=f"phishing_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")


def show_analytics_page(metadata):
    """Show model analytics and performance"""
    st.header("📈 Model Performance Analytics")

    if not metadata:
        st.warning("No metadata available")
        return

    # Metrics overview
    st.subheader("🎯 Model Metrics")

    col1, col2, col3 = st.columns(3)

    metrics = metadata['all_metrics']

    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        st.metric("Precision", f"{metrics['precision']:.4f}")

    with col2:
        st.metric("Recall", f"{metrics['recall']:.4f}")
        st.metric("F1-Score", f"{metrics['f1_score']:.4f}")

    with col3:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        st.metric("Model Type", metadata['model_name'].upper())

    # Metrics comparison chart
    st.markdown("---")
    st.subheader("📊 Metrics Visualization")

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Score': [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                 metrics['f1_score'], metrics['roc_auc']]
    })

    fig = px.bar(metrics_df, x='Metric', y='Score', 
                color='Score', color_continuous_scale='Viridis',
                text='Score')
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(height=400, showlegend=False, yaxis_range=[0, 1.1])
    st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix
    if 'confusion_matrix' in metadata:
        st.markdown("---")
        st.subheader("🔢 Confusion Matrix")

        cm = np.array(metadata['confusion_matrix'])

        fig = px.imshow(cm, 
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Phishing', 'Legitimate'],
                       y=['Phishing', 'Legitimate'],
                       text_auto=True,
                       color_continuous_scale='Blues')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Calculate additional metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("True Negatives", int(tn))
        with col2:
            st.metric("False Positives", int(fp))
        with col3:
            st.metric("False Negatives", int(fn))
        with col4:
            st.metric("True Positives", int(tp))


def get_feature_names():
    """Get all feature names"""
    return [
        'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
        'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
        'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
        'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
        'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
        'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
        'Statistical_report'
    ]


if __name__ == "__main__":
    main()
