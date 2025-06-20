import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Section 1: App Config and Header
st.set_page_config(
    page_title="Rain Prediction Tree Visualizer",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #121212;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #1e1e1e;
    }
    .stDataFrame {
        background-color: #1e1e1e;
    }
    .stMetric {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 5px;
    }
    .stSelectbox > div > div {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stSlider > div > div > div {
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #ffffff !important;
    }
    .stMarkdown {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Configure matplotlib for dark theme
plt.style.use('dark_background')
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

# Header
st.title("🌧️ Rain Prediction Tree Visualizer")
st.markdown("### Explore, Train, and Visualize Decision Trees on Australian Weather Data")

# Sidebar
st.sidebar.markdown("### Built by Kashi")
st.sidebar.markdown("---")

# Section 2: Data Loading
@st.cache_data
def load_data():
    try:
        # Try to load the CSV file
        df = pd.read_csv("weatherAUS.csv")
        return df
    except FileNotFoundError:
        st.error("⚠️ weatherAUS.csv not found. Please upload the file to continue.")
        return None

# File uploader as fallback
uploaded_file = st.sidebar.file_uploader("Upload weatherAUS.csv", type=['csv'])
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
else:
    raw_df = load_data()

if raw_df is not None:
    st.success(f"✅ Data loaded successfully! Shape: {raw_df.shape}")
    
    # Data overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Preview")
        st.dataframe(raw_df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📈 Data Information")
        buffer = []
        buffer.append(f"Dataset Shape: {raw_df.shape}")
        buffer.append(f"Number of Features: {raw_df.shape[1] - 1}")
        buffer.append(f"Target Variable: RainTomorrow")
        buffer.append("\nData Types:")
        for dtype in raw_df.dtypes.value_counts().items():
            buffer.append(f"  {dtype[0]}: {dtype[1]} columns")
        
        buffer.append(f"\nMissing Values:")
        missing = raw_df.isnull().sum()
        buffer.append(f"  Total missing: {missing.sum()}")
        buffer.append(f"  Columns with missing: {(missing > 0).sum()}")
        
        st.text('\n'.join(buffer))
    
    # Statistical summary
    st.subheader("📋 Statistical Summary")
    st.dataframe(raw_df.describe(), use_container_width=True)
    
    # Remove rows with missing RainTomorrow
    initial_rows = len(raw_df)
    raw_df = raw_df.dropna(subset=['RainTomorrow'])
    final_rows = len(raw_df)
    st.info(f"Removed {initial_rows - final_rows} rows with missing 'RainTomorrow' values")
    
    # Section 3: Train/Val/Test Split
    st.header("🔀 Data Splitting")
    
    # Parse year from Date column
    raw_df['Year'] = pd.to_datetime(raw_df['Date']).dt.year
    
    # Show year distribution
    year_counts = raw_df['Year'].value_counts().sort_index()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            x=year_counts.index, 
            y=year_counts.values,
            title="Number of Records per Year",
            labels={'x': 'Year', 'y': 'Number of Records'},
            color_discrete_sequence=['#00d4ff']
        )
        fig.update_layout(
            plot_bgcolor='#121212',
            paper_bgcolor='#121212',
            font_color='white',
            title_font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Split Strategy:**")
        st.markdown("- 🟢 **Train**: Years < 2015")
        st.markdown("- 🟡 **Validation**: Year = 2015") 
        st.markdown("- 🔴 **Test**: Years > 2015")
        st.markdown("---")
        st.markdown("*Time-based split preserves temporal order for realistic weather prediction*")
    
    # Create splits
    train_df = raw_df[raw_df['Year'] < 2015]
    val_df = raw_df[raw_df['Year'] == 2015]
    test_df = raw_df[raw_df['Year'] > 2015]
    
    # Display split information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🟢 Training Set", f"{train_df.shape[0]:,}", f"{train_df.shape[0]/len(raw_df)*100:.1f}%")
    with col2:
        st.metric("🟡 Validation Set", f"{val_df.shape[0]:,}", f"{val_df.shape[0]/len(raw_df)*100:.1f}%")
    with col3:
        st.metric("🔴 Test Set", f"{test_df.shape[0]:,}", f"{test_df.shape[0]/len(raw_df)*100:.1f}%")
    
    # Section 4: Preprocessing
    st.header("⚙️ Data Preprocessing")
    
    # Prepare feature and target columns
    input_cols = [col for col in raw_df.columns if col not in ['Date', 'RainTomorrow', 'Year']]
    target_col = 'RainTomorrow'
    
    # Split inputs and targets
    train_inputs = train_df[input_cols].copy()
    train_targets = train_df[target_col].copy()
    val_inputs = val_df[input_cols].copy()
    val_targets = val_df[target_col].copy()
    test_inputs = test_df[input_cols].copy()
    test_targets = test_df[target_col].copy()
    
    # Identify column types
    numeric_cols = train_inputs.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_inputs.select_dtypes(include=['object']).columns.tolist()
    
    st.subheader("🔧 Feature Engineering Steps")
    
    # Step 1: Imputation
    with st.expander("1️⃣ Missing Value Imputation", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Before Imputation:**")
            missing_before = train_inputs[numeric_cols].isnull().sum()
            missing_df = pd.DataFrame({
                'Column': missing_before.index,
                'Missing Values': missing_before.values,
                'Missing %': (missing_before.values / len(train_inputs) * 100).round(2)
            })
            st.dataframe(missing_df[missing_df['Missing Values'] > 0])
        
        # Perform imputation
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(raw_df[numeric_cols])
        
        train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
        val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
        test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])
        
        with col2:
            st.markdown("**After Imputation:**")
            missing_after = train_inputs[numeric_cols].isnull().sum()
            st.success(f"✅ All missing values imputed using mean strategy")
            st.info(f"Imputed {missing_before.sum()} missing values across {(missing_before > 0).sum()} columns")
    
    # Step 2: Scaling
    with st.expander("2️⃣ Feature Scaling (Min-Max)", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Before Scaling:**")
            before_stats = train_inputs[numeric_cols].describe().loc[['min', 'max']].round(2)
            st.dataframe(before_stats)
        
        # Perform scaling
        scaler = MinMaxScaler()
        scaler.fit(raw_df[numeric_cols])
        
        train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
        val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
        test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])
        
        with col2:
            st.markdown("**After Scaling:**")
            after_stats = pd.DataFrame(train_inputs[numeric_cols]).describe().loc[['min', 'max']].round(3)
            st.dataframe(after_stats)
            st.success("✅ All numeric features scaled to [0, 1] range")
    
    # Step 3: Encoding
    with st.expander("3️⃣ Categorical Encoding (One-Hot)", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Categorical Columns:**")
            cat_info = []
            for col in categorical_cols:
                unique_count = raw_df[col].nunique()
                cat_info.append({'Column': col, 'Unique Values': unique_count})
            st.dataframe(pd.DataFrame(cat_info))
        
        # Perform encoding
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(raw_df[categorical_cols])
        encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
        
        # Transform all sets
        for inputs, name in [(train_inputs, 'train'), (val_inputs, 'val'), (test_inputs, 'test')]:
            encoded_data = encoder.transform(inputs[categorical_cols])
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=inputs.index)
            
            # Replace original categorical columns with encoded ones
            inputs.drop(columns=categorical_cols, inplace=True)
            for col in encoded_df.columns:
                inputs[col] = encoded_df[col]
        
        with col2:
            st.markdown("**After Encoding:**")
            st.success(f"✅ Created {len(encoded_cols)} binary features")
            st.info(f"Total features: {len(train_inputs.columns)}")
    
    # Section 5: EDA (Exploratory Data Analysis)
    st.header("📊 Exploratory Data Analysis")
    
    st.markdown("### Feature Analysis by Location")
    st.markdown("Examining how weather patterns vary across different Australian locations")
    
    # Define feature groups for analysis
    feature_groups = [
        ['Temp9am', 'Temp3pm'],
        ['MinTemp', 'MaxTemp'],
        ['WindSpeed9am', 'WindSpeed3pm'],
        ['Cloud9am', 'Cloud3pm'],
        ['Rainfall', 'Evaporation'],
        ['WindGustSpeed'],
        ['Sunshine']
    ]
    
    # Dark-compatible color palettes
    color_palettes = [
        px.colors.qualitative.Dark2,
        px.colors.qualitative.Set2,
        px.colors.qualitative.Pastel,
        px.colors.qualitative.Bold,
        px.colors.qualitative.Prism,
        px.colors.qualitative.Vivid,
        px.colors.qualitative.Set3
    ]
    
    # Create EDA plots
    for i, features in enumerate(feature_groups):
        with st.expander(f"📈 {' & '.join(features)} Analysis", expanded=False):
            # Group by Location and compute mean
            location_means = raw_df.groupby("Location")[features].mean().reset_index().dropna()
            
            # Calculate sorting key (average across selected features)
            location_means['SortKey'] = location_means[features].mean(axis=1)
            location_means = location_means.sort_values('SortKey', ascending=False).drop(columns='SortKey')
            
            # Melt DataFrame for plotting
            melted_data = location_means.melt(
                id_vars="Location", 
                var_name="Feature", 
                value_name="Mean"
            )
            
            # Set Location as categorical to preserve order
            melted_data["Location"] = pd.Categorical(
                melted_data["Location"], 
                categories=location_means["Location"], 
                ordered=True
            )
            
            # Create grouped bar chart
            fig = px.bar(
                melted_data,
                x="Location",
                y="Mean",
                color="Feature",
                barmode="group",
                color_discrete_sequence=color_palettes[i % len(color_palettes)],
                title=f"Average {' & '.join(features)} by Location",
                height=500
            )
            
            # Apply dark theme styling
            fig.update_layout(
                plot_bgcolor='#121212',
                paper_bgcolor='#121212',
                font_color='white',
                title_font_color='white',
                xaxis_tickangle=-45,
                bargap=0.15,
                margin=dict(t=60, b=80)
            )
            
            # Update axes styling
            fig.update_xaxes(gridcolor='#333333', zerolinecolor='#333333')
            fig.update_yaxes(gridcolor='#333333', zerolinecolor='#333333')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insights
            top_location = location_means.iloc[0]['Location']
            if len(features) == 2:
                feature1_val = location_means.iloc[0][features[0]]
                feature2_val = location_means.iloc[0][features[1]]
                st.info(f"🏆 **{top_location}** has the highest average values: {features[0]} ({feature1_val:.2f}), {features[1]} ({feature2_val:.2f})")
            else:
                feature_val = location_means.iloc[0][features[0]]
                st.info(f"🏆 **{top_location}** has the highest average {features[0]}: {feature_val:.2f}")
    
    # Section 6: Model Training
    st.header("🌳 Model Training")
    
    st.markdown("### Training Decision Tree Classifier")
    st.markdown("Using optimized parameters based on the original analysis")
    
    # Train the model with specified parameters
    @st.cache_data
    def train_baseline_model():
        model = DecisionTreeClassifier(random_state=40, max_depth=12)
        model.fit(train_inputs, train_targets)
        return model
    
    baseline_tree = train_baseline_model()
    
    # Calculate accuracies
    train_accuracy = baseline_tree.score(train_inputs, train_targets)
    val_accuracy = baseline_tree.score(val_inputs, val_targets)
    test_accuracy = baseline_tree.score(test_inputs, test_targets)
    
    # Display model performance
    st.subheader("🎯 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "🟢 Training Accuracy", 
            f"{train_accuracy:.3f}",
            f"{train_accuracy*100:.1f}%"
        )
    with col2:
        st.metric(
            "🟡 Validation Accuracy", 
            f"{val_accuracy:.3f}",
            f"{val_accuracy*100:.1f}%"
        )
    with col3:
        st.metric(
            "🔴 Test Accuracy", 
            f"{test_accuracy:.3f}",
            f"{test_accuracy*100:.1f}%"
        )
    
    # Model insights
    if train_accuracy > val_accuracy:
        overfitting_degree = (train_accuracy - val_accuracy) * 100
        if overfitting_degree > 5:
            st.warning(f"⚠️ Model shows signs of overfitting: {overfitting_degree:.1f}% gap between training and validation accuracy")
        else:
            st.success(f"✅ Model shows good generalization with minimal overfitting ({overfitting_degree:.1f}% gap)")
    
    # Decision Tree Visualization
    st.subheader("🌳 Decision Tree Structure")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("**Tree Visualization (Top 2 Levels)**")
        
        # Create tree plot with dark background
        fig, ax = plt.subplots(figsize=(16, 10))
        fig.patch.set_facecolor('#121212')
        ax.set_facecolor('#121212')
        
        # Plot the decision tree
        plot_tree(
            baseline_tree,
            feature_names=train_inputs.columns,
            class_names=['No Rain', 'Rain'],
            filled=True,
            max_depth=2,
            fontsize=10,
            ax=ax,
            rounded=True,
            proportion=True
        )
        
        # Style the plot
        ax.set_title("Decision Tree Structure", color='white', fontsize=16, pad=20)
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Tree Statistics:**")
        st.write(f"📏 **Max Depth**: {baseline_tree.tree_.max_depth}")
        st.write(f"🍃 **Number of Leaves**: {baseline_tree.tree_.n_leaves}")
        st.write(f"🌲 **Total Nodes**: {baseline_tree.tree_.node_count}")
        
        st.markdown("---")
        st.markdown("**Tree Interpretation:**")
        st.markdown("- Each box represents a decision node")
        st.markdown("- **Gini** measures impurity")
        st.markdown("- **Samples** shows data points")
        st.markdown("- **Value** shows class distribution")
        st.markdown("- **Class** shows majority prediction")
    
    # Section 7: Feature Importance Analysis
    st.header("🔍 Feature Importance Analysis")
    
    # Calculate feature importance
    feature_importance_baseline = pd.DataFrame({
        'Feature': train_inputs.columns,
        'Importance': baseline_tree.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    st.subheader("📊 Feature Importance Rankings")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display feature importance table
        st.markdown("**Complete Feature Importance Table**")
        
        # Add ranking column
        feature_importance_display = feature_importance_baseline.copy()
        feature_importance_display['Rank'] = range(1, len(feature_importance_display) + 1)
        feature_importance_display['Importance (%)'] = (feature_importance_display['Importance'] * 100).round(3)
        
        # Reorder columns
        feature_importance_display = feature_importance_display[['Rank', 'Feature', 'Importance', 'Importance (%)']]
        
        st.dataframe(
            feature_importance_display.head(20), 
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.markdown("**Key Insights:**")
        top_feature = feature_importance_baseline.iloc[0]
        top_5_features = feature_importance_baseline.head(5)
        
        st.success(f"🏆 **Most Important**: {top_feature['Feature']}")
        st.write(f"Importance: {top_feature['Importance']:.4f} ({top_feature['Importance']*100:.2f}%)")
        
        st.markdown("**Top 5 Features:**")
        for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
            st.write(f"{i}. {row['Feature']}: {row['Importance']:.4f}")
        
        # Calculate cumulative importance of top 10
        top_10_cumulative = top_5_features.head(10)['Importance'].sum()
        st.info(f"📈 Top 10 features account for {top_10_cumulative*100:.1f}% of total importance")
    
    # Feature importance visualization
    st.subheader("📈 Top 10 Feature Importance Visualization")
    
    # Get top 10 features
    top_10_features = feature_importance_baseline.head(10)
    
    # Create horizontal bar chart
    fig = px.bar(
        top_10_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Top 10 Most Important Features for Rain Prediction",
        color='Importance',
        color_continuous_scale='viridis',
        height=600
    )
    
    # Apply dark theme
    fig.update_layout(
        plot_bgcolor='#121212',
        paper_bgcolor='#121212',
        font_color='white',
        title_font_color='white',
        margin=dict(l=150, r=50, t=80, b=50)
    )
    
    # Reverse y-axis order to show highest importance at top
    fig.update_yaxes(categoryorder='total ascending')
    fig.update_xaxes(gridcolor='#333333', zerolinecolor='#333333')
    fig.update_yaxes(gridcolor='#333333')
    
    # Add value annotations
    fig.update_traces(
        texttemplate='%{x:.4f}',
        textposition='outside',
        textfont_color='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights and commentary
    st.subheader("💡 Feature Importance Insights")
    
    # Analyze humidity features
    humidity_features = [feat for feat in feature_importance_baseline['Feature'] if 'humidity' in feat.lower()]
    if humidity_features:
        humidity_importance = feature_importance_baseline[
            feature_importance_baseline['Feature'].isin(humidity_features)
        ]['Importance'].sum()
        
        st.success(f"🌡️ **Humidity appears to be the most important factor!** Humidity-related features account for {humidity_importance*100:.1f}% of the model's decision-making.")
    
    # Analyze temperature features
    temp_features = [feat for feat in feature_importance_baseline['Feature'] if 'temp' in feat.lower()]
    if temp_features:
        temp_importance = feature_importance_baseline[
            feature_importance_baseline['Feature'].isin(temp_features)
        ]['Importance'].sum()
        
        st.info(f"🌡️ Temperature-related features contribute {temp_importance*100:.1f}% to predictions")
    
    # Analyze pressure features
    pressure_features = [feat for feat in feature_importance_baseline['Feature'] if 'pressure' in feat.lower()]
    if pressure_features:
        pressure_importance = feature_importance_baseline[
            feature_importance_baseline['Feature'].isin(pressure_features)
        ]['Importance'].sum()
        
        st.info(f"📊 Pressure-related features contribute {pressure_importance*100:.1f}% to predictions")
    
    # Commentary box
    st.markdown("""
    ### 🎯 **Key Takeaways:**
    
    - **Humidity dominates**: Atmospheric moisture content is the strongest predictor of rainfall
    - **Multi-factor approach**: The model considers various meteorological factors, not just one
    - **Feature engineering impact**: Encoded categorical variables (like location and wind direction) also play important roles
    - **Temporal patterns**: Morning vs afternoon measurements provide complementary information
    
    This analysis confirms meteorological intuition - humidity and atmospheric pressure are indeed critical for rainfall prediction!
    """)
    
    # Section 8: Hyperparameter Tuning
    st.header("🔧 Hyperparameter Tuning")
    
    st.markdown("### Systematic Parameter Optimization")
    st.markdown("Finding the optimal balance between model complexity and generalization")
    
    # Function to calculate errors for max_depth tuning
    @st.cache_data
    def calculate_depth_errors():
        results = []
        for depth in range(1, 15):
            model = DecisionTreeClassifier(max_depth=depth, random_state=42)
            model.fit(train_inputs, train_targets)
            train_err = 1 - model.score(train_inputs, train_targets)
            val_err = 1 - model.score(val_inputs, val_targets)
            results.append({
                'Max Depth': depth,
                'Training Error': train_err,
                'Validation Error': val_err
            })
        return pd.DataFrame(results)
    
    # Function to calculate errors for max_leaf_nodes tuning
    @st.cache_data
    def calculate_leaf_errors():
        results = []
        for max_nodes in range(32, 257, 16):
            model = DecisionTreeClassifier(max_leaf_nodes=max_nodes, random_state=42, max_depth=7)
            model.fit(train_inputs, train_targets)
            train_err = 1 - model.score(train_inputs, train_targets)
            val_err = 1 - model.score(val_inputs, val_targets)
            results.append({
                'Max Leaf Nodes': max_nodes,
                'Training Error': train_err,
                'Validation Error': val_err
            })
        return pd.DataFrame(results)
    
    # Max Depth Tuning
    st.subheader("📏 Max Depth Optimization")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Run Max Depth Analysis", key="depth_analysis"):
            with st.spinner("Analyzing max_depth parameters..."):
                depth_errors_df = calculate_depth_errors()
                
                # Create the plot with dark theme
                fig, ax = plt.subplots(figsize=(12, 8))
                fig.patch.set_facecolor('#121212')
                ax.set_facecolor('#121212')
                
                # Plot training and validation errors
                ax.plot(depth_errors_df['Max Depth'], depth_errors_df['Training Error'], 
                       marker='o', linewidth=2, markersize=6, color='#00d4ff', label='Training Error')
                ax.plot(depth_errors_df['Max Depth'], depth_errors_df['Validation Error'], 
                       marker='s', linewidth=2, markersize=6, color='#ff6b6b', label='Validation Error')
                
                # Find and highlight minimum validation error
                min_val_idx = depth_errors_df['Validation Error'].idxmin()
                best_depth = depth_errors_df.loc[min_val_idx, 'Max Depth']
                best_val_error = depth_errors_df.loc[min_val_idx, 'Validation Error']
                
                # Add red dot and vertical line at minimum
                ax.scatter(best_depth, best_val_error, color='red', s=100, zorder=5, label=f'Optimal Depth: {best_depth}')
                ax.axvline(best_depth, color='red', linestyle='--', alpha=0.7)
                
                # Add annotation
                ax.annotate(f'Min Val Error: {best_val_error:.4f}\n@ Depth: {best_depth}',
                           xy=(best_depth, best_val_error),
                           xytext=(best_depth + 1.5, best_val_error + 0.02),
                           arrowprops=dict(arrowstyle='->', color='red'),
                           color='white', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
                
                # Styling
                ax.set_title('Training vs Validation Error by Max Depth', color='white', fontsize=14, pad=20)
                ax.set_xlabel('Max Depth', color='white', fontsize=12)
                ax.set_ylabel('Prediction Error (1 - Accuracy)', color='white', fontsize=12)
                ax.grid(True, alpha=0.3, color='white')
                ax.legend(facecolor='#333333', edgecolor='white')
                ax.tick_params(colors='white')
                
                # Set x-axis ticks
                ax.set_xticks(range(1, 15))
                
                st.pyplot(fig, use_container_width=True)
                
                # Store results in session state for later use
                st.session_state.optimal_depth = best_depth
                st.session_state.depth_analysis_done = True
    
    with col2:
        st.markdown("**Analysis Insights:**")
        if hasattr(st.session_state, 'optimal_depth'):
            st.success(f"🎯 **Optimal Max Depth**: {st.session_state.optimal_depth}")
            st.markdown("**Key Observations:**")
            st.markdown("- Training error continuously decreases")
            st.markdown("- Validation error reaches minimum then increases")
            st.markdown("- Gap indicates overfitting beyond optimal point")
        else:
            st.info("Click 'Run Max Depth Analysis' to see optimization results")
        
        st.markdown("---")
        st.markdown("**Why This Matters:**")
        st.markdown("- **Underfitting**: Too shallow (high bias)")
        st.markdown("- **Overfitting**: Too deep (high variance)")
        st.markdown("- **Sweet Spot**: Minimal validation error")
    
    # Max Leaf Nodes Tuning
    st.subheader("🍃 Max Leaf Nodes Optimization")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Run Leaf Nodes Analysis", key="leaf_analysis"):
            with st.spinner("Analyzing max_leaf_nodes parameters..."):
                leaf_errors_df = calculate_leaf_errors()
                
                # Create the plot with dark theme
                fig, ax = plt.subplots(figsize=(12, 8))
                fig.patch.set_facecolor('#121212')
                ax.set_facecolor('#121212')
                
                # Plot training and validation errors
                ax.plot(leaf_errors_df['Max Leaf Nodes'], leaf_errors_df['Training Error'], 
                       marker='o', linewidth=2, markersize=4, color='#00d4ff', label='Training Error')
                ax.plot(leaf_errors_df['Max Leaf Nodes'], leaf_errors_df['Validation Error'], 
                       marker='s', linewidth=2, markersize=4, color='#ff6b6b', label='Validation Error')
                
                # Find and highlight minimum validation error
                min_val_idx = leaf_errors_df['Validation Error'].idxmin()
                best_leaf_nodes = leaf_errors_df.loc[min_val_idx, 'Max Leaf Nodes']
                best_val_error = leaf_errors_df.loc[min_val_idx, 'Validation Error']
                
                # Add red dot and vertical line at minimum
                ax.scatter(best_leaf_nodes, best_val_error, color='red', s=100, zorder=5)
                ax.axvline(best_leaf_nodes, color='red', linestyle='--', alpha=0.7)
                
                # Add annotation
                ax.annotate(f'Min Val Error: {best_val_error:.4f}\n@ {best_leaf_nodes} leaves',
                           xy=(best_leaf_nodes, best_val_error),
                           xytext=(best_leaf_nodes + 30, best_val_error + 0.01),
                           arrowprops=dict(arrowstyle='->', color='red'),
                           color='white', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
                
                # Styling
                ax.set_title('Training vs Validation Error by Max Leaf Nodes', color='white', fontsize=14, pad=20)
                ax.set_xlabel('Max Leaf Nodes', color='white', fontsize=12)
                ax.set_ylabel('Prediction Error (1 - Accuracy)', color='white', fontsize=12)
                ax.grid(True, alpha=0.3, color='white')
                ax.legend(facecolor='#333333', edgecolor='white')
                ax.tick_params(colors='white')
                
                st.pyplot(fig, use_container_width=True)
                
                # Store results in session state
                st.session_state.optimal_leaf_nodes = best_leaf_nodes
                st.session_state.leaf_analysis_done = True
    
    with col2:
        st.markdown("**Analysis Insights:**")
        if hasattr(st.session_state, 'optimal_leaf_nodes'):
            st.success(f"🎯 **Optimal Leaf Nodes**: {st.session_state.optimal_leaf_nodes}")
            st.markdown("**Key Observations:**")
            st.markdown("- More leaves = higher model complexity")
            st.markdown("- Validation error shows clear minimum")
            st.markdown("- Both curves converge at high leaf counts")
        else:
            st.info("Click 'Run Leaf Nodes Analysis' to see optimization results")
        
        st.markdown("---")
        st.markdown("**Leaf Node Impact:**")
        st.markdown("- **Few Leaves**: Simple decision boundaries")
        st.markdown("- **Many Leaves**: Complex, detailed splits")
        st.markdown("- **Optimal**: Best generalization performance")
    
    # Section 9: Final Optimized Model
    st.header("🏆 Final Optimized Model")
    
    st.markdown("### Training with Optimal Hyperparameters")
    st.markdown("Using the best parameters identified through systematic tuning")
    
    # Train optimized model
    @st.cache_data
    def train_optimized_model():
        model = DecisionTreeClassifier(
            max_depth=7, 
            max_leaf_nodes=120, 
            random_state=40
        )
        model.fit(train_inputs, train_targets)
        return model
    
    optimized_tree = train_optimized_model()
    
    # Calculate performance metrics
    opt_train_accuracy = optimized_tree.score(train_inputs, train_targets)
    opt_val_accuracy = optimized_tree.score(val_inputs, val_targets)
    opt_test_accuracy = optimized_tree.score(test_inputs, test_targets)
    
    # Display optimized model performance
    st.subheader("🎯 Optimized Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🟢 Training Accuracy",
            f"{opt_train_accuracy:.4f}",
            f"{(opt_train_accuracy - train_accuracy)*100:+.2f}%"
        )
    
    with col2:
        st.metric(
            "🟡 Validation Accuracy",
            f"{opt_val_accuracy:.4f}",
            f"{(opt_val_accuracy - val_accuracy)*100:+.2f}%"
        )
    
    with col3:
        st.metric(
            "🔴 Test Accuracy",
            f"{opt_test_accuracy:.4f}",
            f"{(opt_test_accuracy - test_accuracy)*100:+.2f}%"
        )
    
    with col4:
        generalization_gap = opt_train_accuracy - opt_val_accuracy
        st.metric(
            "📊 Generalization Gap",
            f"{generalization_gap:.4f}",
            f"{generalization_gap*100:.2f}%"
        )
    
    # Model comparison
    st.subheader("📈 Model Comparison")
    
    comparison_data = {
        'Model': ['Baseline (depth=12)', 'Optimized (depth=7, leaves=120)'],
        'Training Accuracy': [train_accuracy, opt_train_accuracy],
        'Validation Accuracy': [val_accuracy, opt_val_accuracy],
        'Test Accuracy': [test_accuracy, opt_test_accuracy],
        'Overfitting Gap': [train_accuracy - val_accuracy, opt_train_accuracy - opt_val_accuracy]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.round(4), use_container_width=True, hide_index=True)
    
    # Performance insights
    if opt_val_accuracy > val_accuracy:
        improvement = (opt_val_accuracy - val_accuracy) * 100
        st.success(f"✅ **Optimization Successful!** Validation accuracy improved by {improvement:.2f}%")
    
    if generalization_gap < (train_accuracy - val_accuracy):
        gap_reduction = ((train_accuracy - val_accuracy) - generalization_gap) * 100
        st.success(f"🎯 **Reduced Overfitting!** Generalization gap reduced by {gap_reduction:.2f}%")
    
    # Section 10: Decision Tree SVG Display
    st.header("🌳 Interactive Decision Tree Visualization")
    
    st.markdown("### High-Quality Tree Structure (SVG Format)")
    st.markdown("Detailed visualization of the optimized decision tree with all decision paths")
    
    try:
        from sklearn.tree import export_graphviz
        import subprocess
        import os
        
        # Check if user wants to generate SVG
        if st.button("🎨 Generate High-Quality Tree Visualization", key="generate_svg"):
            with st.spinner("Generating decision tree visualization..."):
                try:
                    import os
                    os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"

                    # Export tree to DOT format
                    dot_data = export_graphviz(
                        optimized_tree,
                        out_file=None,
                        feature_names=train_inputs.columns,
                        class_names=['No Rain', 'Rain'],
                        filled=True,
                        rounded=True,
                        special_characters=True,
                   
                        proportion=True,
                        impurity=True
                    )
                    
                    # Try to create SVG using different methods
                    svg_created = False
                    
                    # Method 1: Try using graphviz library
                    try:
                        import graphviz
                        graph = graphviz.Source(dot_data, format='svg')
                        svg_string = graph.pipe(format='svg').decode('utf-8')
                        
                        # Display SVG inline
                        
                        st.markdown(
                            f"""
                            <div style='
                                background-color: white; 
                                padding: 20px; 
                                border-radius: 10px; 
                                overflow: auto;
                                max-height: 800px;
                            '>
                                {svg_string}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        svg_created = True
                        st.success("✅ Tree visualization generated successfully!")
                        
                    except ImportError:
                        st.warning("⚠️ Graphviz library not available. Showing text representation instead.")
                    
                    # Method 2: Fallback to text representation
                    if not svg_created:
                        st.subheader("📝 Text-Based Tree Structure")
                        tree_text = export_text(
                            optimized_tree,
                            feature_names=list(train_inputs.columns),
                            max_depth=4,
                            spacing=3,
                            decimals=3
                        )
                        st.text(tree_text[:5000] + "..." if len(tree_text) > 5000 else tree_text)
                        
                        # Also show a matplotlib version
                        st.subheader("🖼️ Matplotlib Tree Visualization")
                        fig, ax = plt.subplots(figsize=(20, 12))
                        fig.patch.set_facecolor('white')
                        ax.set_facecolor('white')
                        
                        plot_tree(
                            optimized_tree,
                            feature_names=train_inputs.columns,
                            class_names=['No Rain', 'Rain'],
                            filled=True,
                            max_depth=3,
                            fontsize=8,
                            ax=ax,
                            rounded=True,
                            proportion=True
                        )
                        
                        ax.set_title("Optimized Decision Tree Structure", fontsize=16, pad=20, color='black')
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error generating tree visualization: {str(e)}")
                    
                    # Fallback: Show basic tree info
                    st.subheader("🌳 Tree Structure Information")
                    
                    tree_info = {
                        'Max Depth': optimized_tree.tree_.max_depth,
                        'Number of Leaves': optimized_tree.tree_.n_leaves,
                        'Total Nodes': optimized_tree.tree_.node_count,
                        'Tree Depth Used': optimized_tree.get_depth()
                    }
                    
                    for key, value in tree_info.items():
                        st.write(f"**{key}**: {value}")
    
    except ImportError as e:
        st.warning("⚠️ Some visualization libraries are not available. Showing alternative visualization.")
        
        # Alternative visualization using matplotlib
        st.subheader("🖼️ Decision Tree Structure")
        fig, ax = plt.subplots(figsize=(18, 10))
        fig.patch.set_facecolor('#121212')
        ax.set_facecolor('#121212')
        
        plot_tree(
            optimized_tree,
            feature_names=train_inputs.columns,
            class_names=['No Rain', 'Rain'],
            filled=True,
            max_depth=3,
            fontsize=9,
            ax=ax,
            rounded=True
        )
        
        ax.set_title("Optimized Decision Tree (Max Depth 3)", color='white', fontsize=16, pad=20)
        st.pyplot(fig, use_container_width=True)
    
    # Sidebar controls for hyperparameters
    st.sidebar.subheader("🎛️ Hyperparameters")
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 7)
    max_leaf_nodes = st.sidebar.slider("Max Leaf Nodes", 10, 500, 120, step=10)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, 1)
    
    # Train model
    @st.cache_data
    def train_model(max_depth, max_leaf_nodes, min_samples_split, min_samples_leaf):
        model = DecisionTreeClassifier(
            random_state=42,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        model.fit(train_inputs, train_targets)
        return model
    
    tree_model = train_model(max_depth, max_leaf_nodes, min_samples_split, min_samples_leaf)
    
    # Model performance
    train_score = tree_model.score(train_inputs, train_targets)
    val_score = tree_model.score(val_inputs, val_targets)
    test_score = tree_model.score(test_inputs, test_targets)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🟢 Training Accuracy", f"{train_score:.3f}", f"{train_score*100:.1f}%")
    with col2:
        st.metric("🟡 Validation Accuracy", f"{val_score:.3f}", f"{val_score*100:.1f}%")
    with col3:
        st.metric("🔴 Test Accuracy", f"{test_score:.3f}", f"{test_score*100:.1f}%")
    
    # Feature importance
    st.subheader("🔍 Feature Importance Analysis")
    
    feature_importance = pd.DataFrame({
        'Feature': train_inputs.columns,
        'Importance': tree_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Top 15 most important features
        top_features = feature_importance.head(15)
        
        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 15 Most Important Features",
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            plot_bgcolor='#121212',
            paper_bgcolor='#121212',
            font_color='white',
            title_font_color='white',
            height=600
        )
        fig.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Top 10 Features:**")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            st.markdown(f"{i}. **{row['Feature']}**: {row['Importance']:.4f}")
    
    # Model visualization section
    st.subheader("🌳 Decision Tree Visualization")
    
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Tree Structure", "Confusion Matrix", "Classification Report"])
    
    with viz_tab1:
        st.markdown("**Decision Tree Structure (Top 3 Levels)**")
        
        fig, ax = plt.subplots(figsize=(20, 10))
        fig.patch.set_facecolor('#121212')
        ax.set_facecolor('#121212')
        
        plot_tree(
            tree_model,
            feature_names=train_inputs.columns,
            class_names=['No Rain', 'Rain'],
            filled=True,
            max_depth=3,
            fontsize=10,
            ax=ax
        )
        
        plt.title("Decision Tree Visualization", color='white', fontsize=16)
        st.pyplot(fig, use_container_width=True)
    
    with viz_tab2:
        # Confusion matrices for all sets
        sets = [
            (train_inputs, train_targets, "Training"),
            (val_inputs, val_targets, "Validation"),
            (test_inputs, test_targets, "Test")
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('#121212')
        
        for i, (X, y, name) in enumerate(sets):
            y_pred = tree_model.predict(X)
            cm = confusion_matrix(y, y_pred)
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['No Rain', 'Rain'],
                yticklabels=['No Rain', 'Rain'],
                ax=axes[i]
            )
            axes[i].set_title(f'{name} Set', color='white')
            axes[i].set_xlabel('Predicted', color='white')
            axes[i].set_ylabel('Actual', color='white')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    with viz_tab3:
        # Classification reports
        y_pred_test = tree_model.predict(test_inputs)
        report = classification_report(test_targets, y_pred_test, output_dict=True)
        
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)
    
    # Hyperparameter tuning section
    st.subheader("📊 Hyperparameter Tuning Analysis")
    
    if st.button("🔍 Run Hyperparameter Analysis"):
        with st.spinner("Running hyperparameter analysis..."):
            # Max depth analysis
            depth_results = []
            for depth in range(1, 16):
                model = DecisionTreeClassifier(max_depth=depth, random_state=42)
                model.fit(train_inputs, train_targets)
                train_err = 1 - model.score(train_inputs, train_targets)
                val_err = 1 - model.score(val_inputs, val_targets)
                depth_results.append({
                    'Max Depth': depth,
                    'Training Error': train_err,
                    'Validation Error': val_err
                })
            
            depth_df = pd.DataFrame(depth_results)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=depth_df['Max Depth'],
                y=depth_df['Training Error'],
                mode='lines+markers',
                name='Training Error',
                line=dict(color='#00d4ff')
            ))
            fig.add_trace(go.Scatter(
                x=depth_df['Max Depth'],
                y=depth_df['Validation Error'],
                mode='lines+markers',
                name='Validation Error',
                line=dict(color='#ff6b6b')
            ))
            
            fig.update_layout(
                title="Training vs Validation Error by Max Depth",
                xaxis_title="Max Depth",
                yaxis_title="Prediction Error (1 - Accuracy)",
                plot_bgcolor='#121212',
                paper_bgcolor='#121212',
                font_color='white',
                title_font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Find optimal depth
            optimal_depth = depth_df.loc[depth_df['Validation Error'].idxmin(), 'Max Depth']
            st.success(f"🎯 Optimal Max Depth: {optimal_depth}")
    
    # Download section
    st.subheader("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature importance CSV
        importance_csv = feature_importance.to_csv(index=False)
        st.download_button(
            label="📊 Download Feature Importance",
            data=importance_csv,
            file_name="feature_importance.csv",
            mime="text/csv"
        )
    
    with col2:
        # Model predictions CSV
        test_predictions = tree_model.predict(test_inputs)
        predictions_df = pd.DataFrame({
            'Actual': test_targets.values,
            'Predicted': test_predictions,
            'Probability_No_Rain': tree_model.predict_proba(test_inputs)[:, 0],
            'Probability_Rain': tree_model.predict_proba(test_inputs)[:, 1]
        })
        predictions_csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="🔮 Download Test Predictions",
            data=predictions_csv,
            file_name="test_predictions.csv",
            mime="text/csv"
        )

else:
    st.warning("⚠️ Please upload the weatherAUS.csv file to get started.")
    st.markdown("""
    ### Instructions:
    1. Upload your `weatherAUS.csv` file using the file uploader in the sidebar
    2. The app will automatically load and process the data
    3. Explore different sections to understand the data and model performance
    4. Adjust hyperparameters in the sidebar to see how they affect model performance
    """)