#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import dice_ml
from dice_ml.utils import helpers 
from dice_ml.constants import ModelTypes, _SchemaVersions
from dice_ml.utils.serialize import DummyDataInterface
from IPython.display import display, HTML
import lime
from lime.lime_tabular import LimeTabularExplainer
import fraud_utils
from fraud_utils import *
import pandas as pd
import plotly.express as px
import json
import plotly




def cfe_live_v1(pipeline, X_train, y_train, live_instance, model_name, total_CFs=1):
    """
    Generate counterfactual explanations for a given live instance using DiCE.

    Parameters:
    - pipeline: Trained machine learning model pipeline.
    - X_train: Training data features.
    - y_train: Training data labels.
    - live_instance: The live instance for which counterfactual explanations are generated.
    - model_name: Name or identifier for the machine learning model.
    - total_CFs: Total number of counterfactuals to generate.
    - method: Method for generating counterfactuals ('genetic', 'kdtree', 'random').

    Returns:
    None
    """

    # Prepare outcome data with labels
    outcome_X_train = X_train.copy()
    outcome_X_train['isFraud'] = y_train

    # Define features for DiCE
    features = ['step', 'type', 'amount', 'oldbalanceOrg', 'bal_chg', 'orig_zero',
                'amt_bal_ratio', 'chg_amt_ratio']

    # Create DiCE data object
    d = dice_ml.Data(dataframe=outcome_X_train, continuous_features=features,
                     outcome_name='isFraud')

    # Create DiCE model object
    backend = 'sklearn'
    m = dice_ml.Model(model=pipeline, backend=backend)

    # Convert live_instance to a DataFrame if it's not already
    if not isinstance(live_instance, pd.DataFrame):
        live_instance = pd.DataFrame([live_instance], columns=X_train.columns)

    print(f"Counterfactual Explanation for {model_name} using Random Method")

    
    exp_random = dice_ml.Dice(d, m, method='random')
    dice_exp_random = exp_random.generate_counterfactuals(live_instance, total_CFs=total_CFs, desired_class="opposite", verbose=False)
    dice_exp_random.visualize_as_dataframe(show_only_changes=True)


def counterfactual_Mlexplainer(pipeline, X_train, y_train, live_instance, model_name, total_CFs=1, method='genetic'):
    """
    Generate counterfactual explanations for a given live instance using DiCE.

    Parameters:
    - pipeline: Trained machine learning model pipeline.
    - X_train: Training data features.
    - y_train: Training data labels.
    - live_instance: The live instance for which counterfactual explanations are generated.
    - model_name: Name or identifier for the machine learning model.
    - total_CFs: Total number of counterfactuals to generate.
    - method: Method for generating counterfactuals ('genetic', 'kdtree', 'random').

    Returns:
    None
    """

    # Prepare outcome data with labels
    outcome_X_train = X_train.copy()
    outcome_X_train['isFraud'] = y_train

    # Define features for DiCE
    features = ['step', 'type', 'amount', 'oldbalanceOrg', 'bal_chg', 'orig_zero',
                'amt_bal_ratio', 'chg_amt_ratio']

    # Create DiCE data object
    d = dice_ml.Data(dataframe=outcome_X_train, continuous_features=features,
                     outcome_name='isFraud')

    # Create DiCE model object
    backend = 'sklearn'
    m = dice_ml.Model(model=pipeline, backend=backend)

    # Convert live_instance to a DataFrame if it's not already
    if not isinstance(live_instance, pd.DataFrame):
        live_instance = pd.DataFrame([live_instance], columns=X_train.columns)

    print(f"Counterfactual Explanation for {model_name} using {method.capitalize()} Method")

    # Generate counterfactuals using the specified method
    exp = dice_ml.Dice(d, m, method=method)
    dice_exp = exp.generate_counterfactuals(live_instance, total_CFs=total_CFs, desired_class="opposite", verbose=True)
    dice_exp.visualize_as_dataframe(show_only_changes=True)
    
    
    
def limeExplainer_live(pipeline, X_train, live_instance, model_name):
    features = ['step', 'type', 'amount', 'oldbalanceOrg', 'bal_chg', 
                'orig_zero', 'amt_bal_ratio', 'chg_amt_ratio']

    def pipeline_predict(data):
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=features)
        return pipeline.predict_proba(data)

    # Convert X_train to NumPy array for LIME
    X_train_np = X_train.to_numpy()
    type_column_index = X_train.columns.get_loc('type')

    # Initialize LIME explainer with the training data
    explainer_lime = LimeTabularExplainer(
        training_data=X_train_np,
        feature_names=features,
        categorical_features=[type_column_index],
        class_names=['Legitimate', 'Fraudulent'],
        verbose=True,
        mode='classification',
        kernel_width=3
    )

    # Process the live instance and generate the LIME explanation
    live_instance_np = live_instance.iloc[0].to_numpy() if isinstance(live_instance, pd.DataFrame) else live_instance.to_numpy()
    lime_exp = explainer_lime.explain_instance(live_instance_np, pipeline_predict, num_features=len(features))

    # Extract only the feature explanations
    # Filtering out 'Intercept' and any other non-feature related information
    explanation_data = {feature: weight for feature, weight in lime_exp.as_list()}
    
    # Remove known non-feature keys
    non_feature_keys = ['Intercept', 'Prediction_local', 'Right']
    for non_feature_key in non_feature_keys:
        explanation_data.pop(non_feature_key, None)

    # Convert back to list of tuples and return
    feature_explanations = list(explanation_data.items())

    return feature_explanations


 
def interpret_lime_results_v1(pipeline, X_train, live_instance, model_name):
    # Generate the LIME explanation using adapted_lime_Explainer
    lime_explanation_list = limeExplainer_live(pipeline, X_train, live_instance, model_name)

    # Check the model's prediction for the live instance
    predicted_class_index = pipeline.predict(live_instance)[0]
    class_map = {0: 'Legitimate', 1: 'Fraudulent'}
    predicted_class = class_map.get(predicted_class_index, 'Unknown')

    interpretations = []

    # Interpret each feature's effect from the explanation list
    for feature, effect in lime_explanation_list:
        # Remove any leading or trailing mathematical symbols and values
        feature_name = ''.join([i for i in feature if i.isalpha() or i == '_']).strip()

        # Determine the qualitative impact of the feature
        impact_qualitative = 'considered' if effect != 0 else 'not considered a significant factor'

        # Format the feature explanation without using specific thresholds
        explanation = f"The feature '{feature_name}' is {impact_qualitative} in the model's classification as '{predicted_class}'."
        
        interpretations.append(explanation)

    # Provide a qualitative summary instead of percentages
    interpretations.append("The model considers multiple features to assess each transaction, focusing on patterns that indicate potentially fraudulent or legitimate activity.")

    return interpretations


def generate_transaction_features(df):
    """
    Adds transaction-related features to the DataFrame with shorter column names.

    Parameters:
    - df (pandas.DataFrame): Input DataFrame.

    Returns:
    - pandas.DataFrame: DataFrame with added transaction-related features.
    """
    # Calculate the change in balance for each transaction
    df['bal_chg'] = df['newbalanceOrig'] - df['oldbalanceOrg']

    # Create a binary flag indicating whether the original balance was zero
    df['orig_zero'] = np.where(df['oldbalanceOrg'].fillna(0.0) == 0.0, 1, 0)

    # Calculate the ratio of the transaction amount to the original balance
    df['amt_bal_ratio'] = df['amount'] / df['oldbalanceOrg']
    df['amt_bal_ratio'] = df['amt_bal_ratio'].replace(np.inf, 0)

    # Calculate the ratio of the balance change to the transaction amount
    df['chg_amt_ratio'] = df['bal_chg'] / df['amount']
    df['chg_amt_ratio'] = df['chg_amt_ratio'].replace([np.inf, -np.inf], 0)
    df.drop(columns='newbalanceOrig', inplace=True)

    return df


def get_data():
    df = load_fraud_data('fraud_data.csv')
   
    X_train_resampled_ad, X_test, y_train_resampled_ad, y_test = split_and_sample_data(df, target_column = 'isFraud',
                                                                 sampling_technique='adasyn', random_state=42)
    X_train_resampled_smote, X_test, y_train_resampled_smote, y_test = split_and_sample_data(df, target_column = 'isFraud',
                                                                 sampling_technique='smote', random_state=42)

    X_train_resampled_stomek, X_test, y_train_resampled_stomek, y_test = split_and_sample_data(df, target_column = 'isFraud',
                                                                 sampling_technique='smotetomek', random_state=42)
    return  X_train_resampled_ad, y_train_resampled_ad, X_train_resampled_smote, y_train_resampled_smote, X_train_resampled_stomek, y_train_resampled_stomek




def cf_explanations_v1(pipeline, X_train, y_train, live_instance, model_name, total_CFs):
    outcome_X_train = X_train.copy()
    outcome_X_train['isFraud'] = y_train

    continuous_features = ['step', 'amount', 'oldbalanceOrg', 'bal_chg', 'amt_bal_ratio', 'chg_amt_ratio']
    d = dice_ml.Data(dataframe=outcome_X_train, continuous_features=continuous_features, outcome_name='isFraud')
    m = dice_ml.Model(model=pipeline, backend='sklearn')

    if not isinstance(live_instance, pd.DataFrame):
        live_instance = pd.DataFrame([live_instance], columns=X_train.columns)

    exp = dice_ml.Dice(d, m, method='random')
    cf = exp.generate_counterfactuals(live_instance, total_CFs=total_CFs, desired_class="opposite")

    cf_as_dict = cf.cf_examples_list[0].final_cfs_df.drop(columns=['isFraud']).iloc[0].to_dict()
    original_instance = live_instance.iloc[0].to_dict()

    return cf_as_dict, original_instance


def visualize_counterfactuals_radar_plotly_v1(cf_as_dict):
    """
    Generate and visualize counterfactual explanations using a radar chart.

    Parameters:
    pipeline (Pipeline): The trained pipeline object containing the scaler and the model.
    X_train (DataFrame): The training dataset.
    y_train (Series): The training data labels.
    live_instance (DataFrame/Series): The live instance for which counterfactual explanations are generated.
    model_name (str): Name or identifier for the machine learning model.
    total_CFs (int): Total number of counterfactuals to generate.

    Returns:
    None: The function displays the radar chart.
    """

    # Generate counterfactual explanations
    #cf_dict, original_instance = cf_explanations2(pipeline, X_train, y_train, live_instance, model_name, total_CFs)

    # Prepare data for radar chart
    features = list(cf_as_dict.keys())
    values_original = [original_instance[feature] for feature in features]
    values_cf = [cf_as_dict[feature] for feature in features]

    df = pd.DataFrame(dict(
        r=values_original + values_cf,
        theta=features + features,
        type=['Original'] * len(features) + ['Counterfactual'] * len(features)
    ))

    # Create radar chart
    fig = px.line_polar(df, r='r', theta='theta', color='type', line_close=True,
                        color_discrete_sequence=px.colors.sequential.Plasma[-2::-1])
    fig.update_layout(title='Counterfactual Explanations Radar Chart')
    fig.show()

    
def visualize_counterfactuals_plotly_v1(original_instance, cf_as_dict):
    """
    Visualize counterfactual explanations.

    Parameters:
    original_instance (dict): A dictionary with the original values for each feature.
    cf_as_dict (dict): A dictionary with the counterfactual values for each feature.

    Returns:
    None: Displays a bar chart.
    """

    # Prepare data for visualization
    features = list(cf_as_dict.keys())
    filtered_data = {'Feature': [], 'Original': [], 'Counterfactual': []}

    for feature in features:
        original = original_instance.get(feature, None)
        counterfactual = cf_as_dict.get(feature, None)

        # Check if both original and counterfactual values are available
        if original is not None and counterfactual is not None:
            if original != counterfactual:  # Exclude features with no change
                filtered_data['Feature'].append(feature)
                filtered_data['Original'].append(original)
                filtered_data['Counterfactual'].append(counterfactual)

    # Creating a DataFrame for Plotly
    df = pd.DataFrame(filtered_data)

    # Creating the bar chart
    fig = px.bar(df, x='Feature', y=['Original', 'Counterfactual'], barmode='group',
                 labels={'value': 'Value', 'variable': 'Type'},
                 title='Counterfactual Explanations')
    fig.update_layout(xaxis_title='Feature', yaxis_title='Value')
    fig.show()

    
def explain_counterfactual_percentage_v1(original_instance, cf_as_dict):
    """
    Generate counterfactual explanations for a given live instance and transform them 
    into a user-friendly narrative.

    Parameters:
    pipeline (Pipeline): The trained pipeline object containing the scaler and the model.
    X_train (DataFrame): The training dataset.
    y_train (Series): The training data labels.
    live_instance (DataFrame/Series): The live instance for which counterfactual explanations are generated.
    model_name (str): Name or identifier for the machine learning model.
    total_CFs (int): Total number of counterfactuals to generate.

    Returns:
    str: A narrative explaining the counterfactuals.
    """

    # Generate counterfactual explanations
    #cf_dict, original_instance = cf_explanations2(pipeline, X_train, y_train, live_instance, model_name, total_CFs)

    # Construct narrative, skipping no-change scenarios
    narrative = "To change the model's prediction, consider the following adjustments: \n"
    for feature, new_value in cf_as_dict.items():
        original_value = original_instance[feature]
        if original_value != 0:
            percentage_change = ((new_value - original_value) / original_value) * 100
            if abs(percentage_change) > 0.01:  # Filter out negligible changes
                narrative += f"- Change '{feature}' by {percentage_change:.2f}% (from {original_value} to {new_value}).\n"
        elif new_value != 0:  # Handle cases where the original value is zero, but the new value is not
            narrative += f"- Set '{feature}' to {new_value} (currently zero or undefined).\n"

    return narrative

    
def cfe_summary(pipeline,X_train, y_train, live_instance, model_name, total_CFs):
    cfe_live(pipeline, X_train, y_train, live_instance, model_name, total_CFs)
    cf_as_dict, original_instance =  cf_explanations(pipeline, X_train, y_train, live_instance, model_name, total_CFs)
    visualize_counterfactuals_radar_plotly(cf_as_dict)
    visualize_counterfactuals_plotly(original_instance, cf_as_dict)
    narrative = explain_counterfactual_percentage(original_instance, cf_as_dict)
    print(narrative)   

    


def cfe_live(pipeline, X_train, y_train, live_instance, model_name, total_CFs=2):
    """
    Generate counterfactual explanations for a given live instance using DiCE.

    Parameters:
    - pipeline: Trained machine learning model pipeline.
    - X_train: Training data features.
    - y_train: Training data labels.
    - live_instance: The live instance for which counterfactual explanations are generated.
    - model_name: Name or identifier for the machine learning model.
    - total_CFs: Total number of counterfactuals to generate.
    - method: Method for generating counterfactuals ('genetic', 'kdtree', 'random').

    Returns:
    None
    """

    # Prepare outcome data with labels
    outcome_X_train = X_train.copy()
    outcome_X_train['isFraud'] = y_train

    # Define features for DiCE
    features = ['step', 'type', 'amount', 'oldbalanceOrg', 'bal_chg', 'orig_zero',
                'amt_bal_ratio', 'chg_amt_ratio']

    # Create DiCE data object
    d = dice_ml.Data(dataframe=outcome_X_train, continuous_features=features,
                     outcome_name='isFraud')

    # Create DiCE model object
    backend = 'sklearn'
    m = dice_ml.Model(model=pipeline, backend=backend)

    # Convert live_instance to a DataFrame if it's not already
    if not isinstance(live_instance, pd.DataFrame):
        live_instance = pd.DataFrame([live_instance], columns=X_train.columns)

    print(f"Counterfactual Explanation for {model_name} using Random Method")

    
    exp_random = dice_ml.Dice(d, m, method='random')
    dice_exp_random = exp_random.generate_counterfactuals(live_instance, total_CFs=total_CFs, desired_class="opposite", verbose=False)
    
    
    # Convert counterfactuals to a DataFrame and then to a dictionary for JSON serialization
    cf_dataframe = dice_exp_random.visualize_as_dataframe(show_only_changes=True)
    cf_dict = cf_dataframe.to_dict(orient='records')  # Converts the DataFrame to a list of dictionaries

    return cf_dict


def interpret_lime_results(pipeline, X_train, live_instance, model_name):
    # Generate the LIME explanation using adapted_lime_Explainer
    lime_explanation_list = limeExplainer_live(pipeline, X_train, live_instance, model_name)

    # Check the model's prediction for the live instance
    predicted_class_index = pipeline.predict(live_instance)[0]
    class_map = {0: 'Legitimate', 1: 'Fraudulent'}
    predicted_class = class_map.get(predicted_class_index, 'Unknown')

    interpretations = []

    # Interpret each feature's effect from the explanation list
    for feature, effect in lime_explanation_list:
        feature_name = ''.join([i for i in feature if i.isalpha() or i == '_']).strip()
        impact_qualitative = 'considered' if effect != 0 else 'not considered a significant factor'

        # Create a dictionary for each feature interpretation
        feature_explanation = {
            'feature_name': feature_name,
            'impact': impact_qualitative,
            'predicted_class': predicted_class
        }

        interpretations.append(feature_explanation)

    # Add a summary to the interpretations list
    summary = "The model considers multiple features to assess each transaction, focusing on patterns that indicate potentially fraudulent or legitimate activity."
    interpretations.append({'summary': summary})

    return interpretations


def cf_explanations(pipeline, X_train, y_train, live_instance, model_name, total_CFs=2):
    outcome_X_train = X_train.copy()
    outcome_X_train['isFraud'] = y_train

    continuous_features = ['step', 'amount', 'oldbalanceOrg', 'bal_chg', 'amt_bal_ratio', 'chg_amt_ratio']
    d = dice_ml.Data(dataframe=outcome_X_train, continuous_features=continuous_features, outcome_name='isFraud')
    m = dice_ml.Model(model=pipeline, backend='sklearn')

    if not isinstance(live_instance, pd.DataFrame):
        live_instance = pd.DataFrame([live_instance], columns=X_train.columns)

    exp = dice_ml.Dice(d, m, method='random')
    cf = exp.generate_counterfactuals(live_instance, total_CFs=total_CFs, desired_class="opposite")

    # Extracting multiple counterfactuals into a list of dictionaries
    cf_as_dicts = [cf_instance.final_cfs_df.drop(columns=['isFraud']).to_dict(orient='records')[0] for cf_instance in cf.cf_examples_list]
    original_instance = live_instance.iloc[0].to_dict()
    
    return cf_as_dicts, original_instance


def visualize_counterfactuals_radar_plotly(cf_as_dict, original_instance):
    features = list(cf_as_dict[0].keys())
    values_original = [original_instance.get(feature, 0) for feature in features]
    values_cfs = [[cf.get(feature, 0) for feature in features] for cf in cf_as_dict]

    df_original = pd.DataFrame(dict(r=values_original, theta=features))
    df_original['type'] = 'Original'

    df_cfs = pd.concat(
        [pd.DataFrame(dict(r=values_cf, theta=features)) for values_cf in values_cfs],
        ignore_index=True
    )
    df_cfs['type'] = 'Counterfactual'

    df = pd.concat([df_original, df_cfs], ignore_index=True)

    fig = px.line_polar(df, r='r', theta='theta', color='type', line_close=True,
                        color_discrete_sequence=px.colors.sequential.Plasma[-2::-1])

    fig.update_layout(
        title='Counterfactual Explanations Radar Chart',
        autosize=True,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white",
        polar=dict(radialaxis=dict(visible=True))
    )

    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json


def visualize_counterfactuals_plotly(original_instance, cf_as_dicts):
    # Prepare data for visualization
    filtered_data = {'Feature': [], 'Type': [], 'Value': []}

    # Assuming cf_as_dicts is a list of counterfactual instances
    for cf_as_dict in cf_as_dicts:
        for feature, cf_value in cf_as_dict.items():
            original_value = original_instance.get(feature, None)
            if original_value is not None and cf_value != original_value:  # Exclude features with no change
                filtered_data['Feature'].append(feature)
                filtered_data['Type'].append('Original')
                filtered_data['Value'].append(original_value)
                filtered_data['Feature'].append(feature)
                filtered_data['Type'].append('Counterfactual')
                filtered_data['Value'].append(cf_value)

    # Creating a DataFrame for Plotly
    df = pd.DataFrame(filtered_data)

    # Creating the bar chart
    fig = px.bar(df, x='Feature', y='Value', color='Type', barmode='group',
                 labels={'Type': 'Type', 'Value': 'Value'},
                 title='Counterfactual Explanations')
    fig.update_layout(xaxis_title='Feature', yaxis_title='Value')

    # Convert the figure to JSON for frontend
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json


def explain_counterfactual_percentage(original_instance, cf_as_dicts):
    narrative = "<p>To change the model's prediction, consider the following adjustments:</p>"

    for cf_as_dict in cf_as_dicts:
        narrative += "<ul>"
        for feature, new_value in cf_as_dict.items():
            original_value = original_instance[feature]
            if original_value != 0:
                percentage_change = ((new_value - original_value) / original_value) * 100
                if abs(percentage_change) > 0.01:  # Filter out negligible changes
                    narrative += f"<li>Change <strong>{feature}</strong> by {percentage_change:.2f}% (from {original_value} to {new_value}).</li>"
            elif new_value != 0:  # Handle cases where the original value is zero, but the new value is not
                narrative += f"<li>Set <strong>{feature}</strong> to {new_value} (currently zero or undefined).</li>"
        narrative += "</ul>"

    return narrative
    
    
    
def visualize_counterfactuals_radar_plotly_v2(cf_as_dict):
    # Prepare data for radar chart
    features = list(cf_as_dict.keys())
    values_original = [original_instance[feature] for feature in features]
    values_cf = [cf_as_dict[feature] for feature in features]

    df = pd.DataFrame(dict(
        r=values_original + values_cf,
        theta=features + features,
        type=['Original'] * len(features) + ['Counterfactual'] * len(features)
    ))

    # Create radar chart using Plotly
    fig = px.line_polar(df, r='r', theta='theta', color='type', line_close=True,
                        color_discrete_sequence=px.colors.sequential.Plasma[-2::-1])
    fig.update_layout(title='Counterfactual Explanations Radar Chart')

    # Display the radar chart in Streamlit
    st.plotly_chart(fig)


def visualize_counterfactuals_plotly_v2(original_instance, cf_as_dict):
    # Prepare data for visualization
    features = list(cf_as_dict.keys())
    filtered_data = {'Feature': [], 'Original': [], 'Counterfactual': []}

    for feature in features:
        original = original_instance.get(feature, None)
        counterfactual = cf_as_dict.get(feature, None)

        # Check if both original and counterfactual values are available
        if original is not None and counterfactual is not None:
            if original != counterfactual:  # Exclude features with no change
                filtered_data['Feature'].append(feature)
                filtered_data['Original'].append(original)
                filtered_data['Counterfactual'].append(counterfactual)

    # Creating a DataFrame for Plotly
    df = pd.DataFrame(filtered_data)

    # Creating the bar chart using Plotly
    fig = px.bar(df, x='Feature', y=['Original', 'Counterfactual'], barmode='group',
                 labels={'value': 'Value', 'variable': 'Type'},
                 title='Counterfactual Explanations')
    fig.update_layout(xaxis_title='Feature', yaxis_title='Value')

    # Display the bar chart in Streamlit
    st.plotly_chart(fig)


def explain_counterfactual_percentage_v2(original_instance, cf_as_dict):
    # Construct narrative, skipping no-change scenarios
    narrative = "To change the model's prediction, consider the following adjustments: \n"
    for feature, new_value in cf_as_dict.items():
        original_value = original_instance[feature]
        if original_value != 0:
            percentage_change = ((new_value - original_value) / original_value) * 100
            if abs(percentage_change) > 0.01:  # Filter out negligible changes
                narrative += f"- Change '{feature}' by {percentage_change:.2f}% (from {original_value} to {new_value}).\n"
        elif new_value != 0:  # Handle cases where the original value is zero, but the new value is not
            narrative += f"- Set '{feature}' to {new_value} (currently zero or undefined).\n"

    # Display the narrative in Streamlit
    st.write(narrative)    
    

def cfe_live_v2(pipeline, X_train, y_train, live_instance, model_name, total_CFs=2):
    # Prepare outcome data with labels
    outcome_X_train = X_train.copy()
    outcome_X_train['isFraud'] = y_train

    # Define features for DiCE
    features = ['step', 'type', 'amount', 'oldbalanceOrg', 'bal_chg', 'orig_zero',
                'amt_bal_ratio', 'chg_amt_ratio']

    # Create DiCE data object
    d = dice_ml.Data(dataframe=outcome_X_train, continuous_features=features,
                     outcome_name='isFraud')

    # Create DiCE model object
    backend = 'sklearn'
    m = dice_ml.Model(model=pipeline, backend=backend)

    # Convert live_instance to a DataFrame if it's not already
    if not isinstance(live_instance, pd.DataFrame):
        live_instance = pd.DataFrame([live_instance], columns=X_train.columns)

    st.write(f"Counterfactual Explanation for {model_name} using Random Method")

    exp_random = dice_ml.Dice(d, m, method='random')
    dice_exp_random = exp_random.generate_counterfactuals(live_instance, total_CFs=total_CFs, desired_class="opposite", verbose=False)
    
    # Display the counterfactual explanations as a DataFrame
    st.write(dice_exp_random.data)


def interpret_lime_results_v2(pipeline, X_train, live_instance, model_name):
    # Generate the LIME explanation using adapted_lime_Explainer
    lime_explanation_list = limeExplainer_live(pipeline, X_train, live_instance, model_name)

    # Check the model's prediction for the live instance
    predicted_class_index = pipeline.predict(live_instance)[0]
    class_map = {0: 'Legitimate', 1: 'Fraudulent'}
    predicted_class = class_map.get(predicted_class_index, 'Unknown')

    interpretations = []

    # Interpret each feature's effect from the explanation list
    for feature, effect in lime_explanation_list:
        # Remove any leading or trailing mathematical symbols and values
        feature_name = ''.join([i for i in feature if i.isalpha() or i == '_']).strip()

        # Determine the qualitative impact of the feature
        impact_qualitative = 'considered' if effect != 0 else 'not considered a significant factor'

        # Format the feature explanation without using specific thresholds
        explanation = f"The feature '{feature_name}' is {impact_qualitative} in the model's classification as '{predicted_class}'."
        
        interpretations.append(explanation)

    # Provide a qualitative summary instead of percentages
    interpretations.append("The model considers multiple features to assess each transaction, focusing on patterns that indicate potentially fraudulent or legitimate activity.")

    # Display the interpretations
    for interpretation in interpretations:
        st.write(interpretation)


def main_instance(cf_as_dict, prediction, model_name):
    """
    Generate an HTML representation of counterfactual explanations.

    Parameters:
    - cf_as_dict: A list of dictionaries representing counterfactual explanations.
    - prediction: A list containing model predictions.
    - model_name: The name of the machine learning model.

    Returns:
    A string containing HTML representation of counterfactual explanations.
    """
    # Check if cf_as_dict is a list and not empty
    if not isinstance(cf_as_dict, list) or not cf_as_dict:
        return "Counterfactual explanation not available."

    # Extract first element from the list
    data_list = cf_as_dict[0]

    message1 = f'<p><b>Counterfactual Explanation for {model_name}</b></p>'
    message2 = f'<p>Query instance (original outcome : {prediction})</p>'

    # Create a DataFrame with the data
    df = pd.DataFrame([data_list])
    df['isFraud'] = prediction

    # Convert the DataFrame to an HTML table
    html_table = df.to_html(classes='table table-striped', escape=False, index=False)

    # Combine the messages and the HTML table with a line break
    result_html = message1 + "<br>" + message2 + "<br>" + html_table

    return result_html




def cfe_instance(original_instance, prediction):
    """
    Generate an HTML representation of the main instance with a message.

    Parameters:
    - original_instance: A dictionary representing the original instance.
    - prediction: A list containing model predictions.
    - model_name: The name of the machine learning model.

    Returns:
    A string containing HTML representation of the main instance with a message.
    """
    # Create a list containing the original instance
    datalist = [original_instance]

    # Determine the new outcome based on the prediction
    if prediction == 1:
        new_outcome = 0
    elif prediction == 0:
        new_outcome = 1

    # Create a DataFrame from the data list
    df = pd.DataFrame(datalist)
    df['isFraud'] = new_outcome

    # Convert the DataFrame to an HTML table
    html_table = df.to_html(classes='table table-striped', escape=False, index=False)

    # Create a message indicating the new outcome
    message = f'<p>Diverse Counterfactual set (new outcome: {new_outcome})</p>'

    # Combine the message and the HTML table with a line break
    result_html = message + "<br>" + html_table

    return result_html
        
def interpret_lime_results_v5(pipeline, X_train, live_instance, model_name):
    # Generate the LIME explanation using adapted_lime_Explainer
    lime_explanation_list = limeExplainer_live(pipeline, X_train, live_instance, model_name)

    # Check the model's prediction for the live instance
    predicted_class_index = pipeline.predict(live_instance)[0]
    class_map = {0: 'Legitimate', 1: 'Fraudulent'}
    predicted_class = class_map.get(predicted_class_index, 'Unknown')

    interpretations = []

    # Interpret each feature's effect from the explanation list
    for feature, effect in lime_explanation_list:
        # Remove any leading or trailing mathematical symbols and values
        feature_name = ''.join([i for i in feature if i.isalpha() or i == '_']).strip()

        # Determine the qualitative impact of the feature
        impact_qualitative = 'considered' if effect != 0 else 'not considered a significant factor'

        # Format the feature explanation without using specific thresholds
        explanation = f"The feature '{feature_name}' is {impact_qualitative} in the model's classification as '{predicted_class}'."
        
        interpretations.append(explanation)

    # Provide a qualitative summary instead of percentages
    interpretations.append("The model considers multiple features to assess each transaction, focusing on patterns that indicate potentially fraudulent or legitimate activity.")

    # Convert the interpretations list to an HTML unordered list
    html_list = "<ul>"
    for interpretation in interpretations:
        html_list += f"<li>{interpretation}</li>"
    html_list += "</ul>"

    return html_list



def visualize_counterfactuals_radar_plotly_v5(cf_as_dict, original_instance):
    features = list(cf_as_dict[0].keys())
    values_original = [original_instance.get(feature, 0) for feature in features]
    values_cfs = [[cf.get(feature, 0) for feature in features] for cf in cf_as_dict]

    df_original = pd.DataFrame(dict(r=values_original, theta=features))
    df_original['type'] = 'Original'

    df_cfs = pd.concat(
        [pd.DataFrame(dict(r=values_cf, theta=features)) for values_cf in values_cfs],
        ignore_index=True
    )
    df_cfs['type'] = 'Counterfactual'

    df = pd.concat([df_original, df_cfs], ignore_index=True)

    fig = px.line_polar(df, r='r', theta='theta', color='type', line_close=True,
                        color_discrete_sequence=px.colors.sequential.Plasma[-2::-1])

    fig.update_layout(
        title='Counterfactual Explanations Radar Chart',
        autosize=True,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white",
        polar=dict(radialaxis=dict(visible=True))
    )

    # Convert the figure to HTML
    plotly_html = fig.to_html()

    return plotly_html


def visualize_counterfactuals_plotly_v5(original_instance, cf_as_dict):
    # Prepare data for visualization
    filtered_data = {'Feature': [], 'Type': [], 'Value': []}

    # Assuming cf_as_dicts is a list of counterfactual instances
    for cf_as_dict in cf_as_dicts:
        for feature, cf_value in cf_as_dict.items():
            original_value = original_instance.get(feature, None)
            if original_value is not None and cf_value != original_value:  # Exclude features with no change
                filtered_data['Feature'].append(feature)
                filtered_data['Type'].append('Original')
                filtered_data['Value'].append(original_value)
                filtered_data['Feature'].append(feature)
                filtered_data['Type'].append('Counterfactual')
                filtered_data['Value'].append(cf_value)

    # Creating a DataFrame for Plotly
    df = pd.DataFrame(filtered_data)

    # Creating the bar chart
    fig = px.bar(df, x='Feature', y='Value', color='Type', barmode='group',
                 labels={'Type': 'Type', 'Value': 'Value'},
                 title='Counterfactual Explanations')
    fig.update_layout(xaxis_title='Feature', yaxis_title='Value')

    # Convert the figure to HTML
    plotly_html = fig.to_html()

    return plotly_html

   