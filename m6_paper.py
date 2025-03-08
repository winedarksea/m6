#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 17:55:29 2023

@author: colincatlin
"""
import json
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

diverging_palette = sns.diverging_palette(220, 20, as_cmap=True)

primary_color = 'royalblue'
secondary_color = 'coral'
group_colors = {
    'motif': primary_color,
    'ML': secondary_color,
    'stat': 'mediumseagreen',
    'naive': 'orchid',
}
sns_colors = sns.color_palette("pastel")
group_colors = {
    'motif': sns_colors[0],
    'ML': sns_colors[1],
    'stat': sns_colors[2],
    'naive': sns_colors[3],
}
transformer_colors = {
    'filter': sns_colors[4],
    'scaler': sns_colors[5],
    'decomposition': sns_colors[6],
    'other': sns_colors[7],
}
primary_color = sns_colors[0]
secondary_color = sns_colors[1]


df = pd.read_csv("/users/colincatlin/Downloads/m6/a_m6_trial_results_3.csv")
id_col = df.columns[0]

df['mae+spl+rmse'] =  df['mae'] / df['mae'].max() / 2 + df['spl'] / df['spl'].max() + df['rmse'] / df['rmse'].max() / 2
if df['return_mrkt_basket'].abs().max() > 2:
    df['return_hinge_above_mrkt'] = (df['return_hinge_forecast'] - (df['return_mrkt_basket']) / 100)
else:
    df['return_hinge_above_mrkt'] = (df['return_hinge_forecast'] - (df['return_mrkt_basket']))

df_grouped = df.groupby('Unnamed: 0')['return_hinge_rounded'].mean()
top5 = df_grouped.sort_values(ascending=False).head(5).index.tolist()
top5_df = df[df['Unnamed: 0'].isin(top5)]

corr = df.select_dtypes("number").corr()

cols = ['return_hinge_rounded', 'return_agreement', 'RPS_Forecasts', 'smape', 'mae', 'spl', 'rmse', 'mae+spl+rmse']

correlation_matrix = corr[cols].loc[cols]
# Create a mask for the upper triangle to hide redundant information
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up the matplotlib figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Generate a diverging colormap
cmap = sns.diverging_palette(220, 20, as_cmap=True)

# Create the correlogram using a heatmap
sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=1, center=0,
            annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": 0.7})
sns.set_style("whitegrid")  # Add a grid for clarity

# Add a title
plt.title("Correlogram of Metric Correlations from Optimized Forecasts")
plt.savefig("correlogram_nocontour.png", dpi=300, bbox_inches="tight")

# Set up the matplotlib figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Add a vertical line to the plot
vertical_line_value = 20.0 * 0.8

# Create the histogram
# sns.histplot(df['RPS_Forecasts'], bins=20, kde=True, ax=ax, color='skyblue', edgecolor='k')
# ax.axvline(x=vertical_line_value, color='red', linestyle='--', label='Naive Forecast Accuracy')
sns.histplot(df['RPS_Forecasts'] * 0.8, bins=20, kde=True, ax=ax, color=primary_color, edgecolor='k')
sns.despine(left=True, bottom=True)  # Removes the top and right borders for cleaner look
ax.axvline(x=vertical_line_value, color=secondary_color, linestyle='--', label='Naive RPS Accuracy')
# Add labels and title
plt.xlabel('RPS, Smaller Values are Better', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Histogram of RPS Scores for Forecasts Optimized with Different AutoTS Parameters', fontsize=14)
# Add legend
ax.legend()
# Show the plot
plt.tight_layout()
plt.savefig("RPS_scores_adjusted.png", dpi=300, bbox_inches="tight")
# plt.show()


# Set up the matplotlib figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
# Add a vertical line to the plot
vertical_line_value = 0.0  # Adjust this value according to your data
# Create the histogram
sns.histplot(df['return_hinge_forecast'], bins=20, kde=True, ax=ax, color=primary_color, edgecolor='k')
sns.despine(left=True, bottom=True)  # Removes the top and right borders for cleaner look
ax.axvline(x=vertical_line_value, color=secondary_color, linestyle='--', label='Breakeven Return')
# Add labels and title
plt.xlabel('RPS, Smaller Values are Better', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Histogram of Hinge Returns for Forecasts Optimized with Different AutoTS Parameters', fontsize=14)
# Add legend
ax.legend()
# Show the plot
plt.tight_layout()
plt.show()

# Set up the matplotlib figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
# Add a vertical line to the plot
vertical_line_value = 0.0  # Adjust this value according to your data
# Create the histogram
sns.histplot(df['return_hinge_above_mrkt'] * 100, bins=20, kde=True, ax=ax, color=primary_color, edgecolor='k')
sns.despine(left=True, bottom=True)  # Removes the top and right borders for cleaner look
ax.axvline(x=vertical_line_value, color=secondary_color, linestyle='--', label='Market Return')
# Add labels and title
plt.xlabel('Percent Return, %, relative to market where 0 is market return', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Histogram of Hinge Returns Against Market for Forecasts Optimized with Different AutoTS Parameters', fontsize=14)
# Add legend
ax.legend()
# Show the plot
plt.tight_layout()
plt.savefig("returns_above_market.png", dpi=300, bbox_inches="tight")
# plt.show()


# Create a new figure
plt.figure(figsize=(10, 6))
data = df[['return_agreement', 'return_hinge_forecast', 'return_point_forecast']].clip(upper=0.2, lower=-0.2) * 100
# data = df.groupby('Unnamed: 0')[['return_hinge_forecast', 'return_point_forecast', 'return_agreement']].mean()
# Plot violin plot
sns.violinplot(data=data)
# Add title and labels
plt.title('Violin Plot of Returns from Different Decision Methods Across Differently Optimized Forecasts')
plt.xlabel('Method of Generating Decisions from Forecasts')
plt.ylabel('Percentage Return, %')
# Show the plot
plt.savefig("violin_return.png", dpi=300, bbox_inches="tight")
# plt.show()

plt.figure(figsize=(10, 6))
data = df[['return_agreement', 'return_hinge_forecast', 'return_point_forecast', 'return_hinge_above_mrkt']].clip(upper=0.2, lower=-0.2) * 100
# Plot violin plot
sns.violinplot(data=data)
# Add title and labels
plt.title('Violin Plot of Returns from Different Decision Methods Across Differently Optimized Forecasts')
plt.xlabel('Method of Generating Decisions from Forecasts')
plt.ylabel('Percentage Return, %')
# Show the plot
plt.show()

#########################################################
## METHODS DATA
methods = pd.read_csv("/users/colincatlin/Downloads/m6/m6_all_saved_best.csv")
# method1 = pd.read_csv("/users/colincatlin/Downloads/autots_forecast_template_m5_new2_20231030.csv")
# method2 = pd.read_csv("/users/colincatlin/Downloads/autots_forecast_template_daily_20231030.csv")
# methods = pd.concat([method1, method2])

all_transformers = []
all_models = []
for index, row in methods.iterrows():
    current_params = json.loads(row['ModelParameters'])
    ModelParameters = current_params.copy()
    if 'series' in ModelParameters.keys():
        series = ModelParameters['series']
        series = pd.DataFrame.from_dict(series, orient="index").reset_index(drop=False)
        if series.shape[1] > 2:
            # for mosaic style ensembles, choose the mode model id
            series.set_index(series.columns[0], inplace=True)
            series = series.mode(axis=1)[0].to_frame().reset_index(drop=False)
        series.columns = ['Series', 'ID']
        results = pd.Series(
            {
                x: current_params['models'][x]['Model']
                for x in current_params['models'].keys()
            }
        )
        results.name = "Model"
        series = series.merge(results, left_on="ID", right_index=True)
        # series = series.merge(self.results()[['ID', "Model"]].drop_duplicates(), on="ID")  # old
        # series = series.merge(self.df_wide_numeric.std().to_frame(), right_index=True, left_on="Series")
        # series = series.merge(self.df_wide_numeric.mean().to_frame(), right_index=True, left_on="Series")
        series.columns = ["Series", "ID", 'Model']  # , "Volatility", "Mean"]
        series['Transformers'] = series['ID'].copy()
        series['FillNA'] = series['ID'].copy()
        lookup = {}
        na_lookup = {}
        for k, v in ModelParameters['models'].items():
            try:
                trans_params = json.loads(v.get('TransformationParameters', '{}'))
                lookup[k] = ",".join(trans_params.get('transformations', {}).values())
                na_lookup[k] = trans_params.get('fillna', '')
            except Exception:
                lookup[k] = "None"
                na_lookup[k] = "None"
        series['Transformers'] = (
            series['Transformers'].replace(lookup).replace("", "None")
        )
        series['FillNA'] = series['FillNA'].replace(na_lookup).replace("", "None")
        models = series['Model'].value_counts().iloc[0:100]
        all_models.append(models)
        transformers = pd.Series(
           ",".join(series['Transformers']).split(",")
       ).value_counts()
        all_transformers.append(transformers)
    else:
        all_models.append(pd.Series({row['Model']: 1}, name="count"))
        transformers = pd.Series(
            json.loads(row['TransformationParameters'])['transformations']
        ).value_counts()
        all_transformers.append(transformers)

full_transformers = pd.concat(all_transformers).groupby(level=0).sum()
full_models = pd.concat(all_models).groupby(level=0).sum().drop(columns='Ensemble')


# full_models = methods.groupby('Model')['ID'].count()
# Sort the count_series by values in descending order
sorted_count_series = full_models.sort_values(ascending=False)
classes = {
 'ARDL': 'stat', 'AverageValueNaive': 'naive', 'ConstantNaive': 'naive', 'DatepartRegression': 'ML',
       'ETS': 'stat', 'FBProphet': 'stat', 'GLM': 'stat', 'GLS': 'stat', 'LastValueNaive': 'naive', 'MAR': 'stat',
       'MultivariateMotif': 'motif', 'MultivariateRegression': 'ML', 'NVAR': 'stat', 'RRVAR': 'stat',
       'SeasonalNaive': 'naive', 'SectionalMotif': 'motif', 'Theta': 'stat', 'UnivariateMotif': 'motif',
       'UnivariateRegression': 'ML', 'UnobservedComponents': 'stat', 'VAR': 'stat', 'VECM': 'stat',
       'WindowRegression': 'ML', 'ZeroesNaive': 'naive', 'BallTreeMultivariateMotif': 'motif',
       'SeasonalityMotif': 'motif', 'WindowRegression': 'ML', 'FFT': 'stat', 'MetricMotif': 'motif', 'ARCH': 'stat',
       'KalmanStateSpace': 'stat',
}
classes = {x: y for x, y in classes.items() if x in full_models.keys()}
classes = sorted(classes.items(), key=lambda pair: sorted_count_series.index.tolist().index(pair[0]))
class_values = [x[1] for x in classes]
# Map group colors to each method
colors = [group_colors[group] for group in class_values]

# fig.savefig('plot.tif', dpi=300, bbox_inches="tight")  # requires pillow


# Seaborn Methods Barplot
fig, ax = plt.subplots(figsize=(8, 6))

# Convert series to dataframe for seaborn compatibility
methods_df = sorted_count_series.reset_index()
methods_df.columns = ['Method', 'Count']

sns.barplot(x='Method', y='Count', data=methods_df, palette=colors, edgecolor='k', ax=ax)

# Rotate x-labels for better readability
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')

# Annotate bar heights
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., int(p.get_height())),
                ha='center', va='bottom', fontsize=10, color='black')

# Add labels, title, and grid for clarity
ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Count of Models Used in AutoTS by Type in M6', fontsize=14)
sns.despine(left=True, bottom=True)  # Removes the top and right borders for cleaner look
ax.grid(False, axis='x')  # Remove x-axis grid lines
ax.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.7)  # Add horizontal grid lines

# Create custom legend patches for each group
legend_patches = [Patch(color=color, label=group) for group, color in group_colors.items()]

# Add the legend with the custom patches
ax.legend(handles=legend_patches, loc='upper right', title='Groups', frameon=False)  # frameon=False removes the box around the legend

# Adjust layout
plt.tight_layout()
plt.savefig("count_models.png", dpi=300, bbox_inches="tight")
# plt.show()


top_n = 15
sorted_trans = full_transformers.sort_values(ascending=False)[0:top_n]
trans_classes_ov = {
    'SeasonalDifference': 'decomposition', 'DifferencedTransformer': 'scaler', 'Detrend': 'decomposition',
    'AlignLastValue': 'other',
    'MinMaxScaler': 'scaler', 'QuantileTransformer': 'scaler',
    'ScipyFilter': 'filter', 'StandardScaler': 'scaler', 'RobustScaler': 'scaler',
    'PctChangeTransformer': 'scaler',
    'ClipOutliers': 'filter', 'MaxAbsScaler': 'scaler', 'Round': 'filter', 'bkfilter': 'filter',
    'Slice': 'filter', 'cffilter': 'filter', 'PositiveShift': 'scaler', 'PowerTransformer': 'scaler',
    'SinTrend': 'decomposition', 'EWMAFilter': 'filter', 'Discretize': 'filter',
    'RollingMeanTransformer': 'filter',
    'DatepartRegression': 'decomposition', 'AnomalyRemoval': 'filter', 'IntermittentOccurrence': "scaler",
    "Log": 'scaler', 'HPFilter': 'filter', 'STLFilter': "filter", 'PCA': 'decomposition',
    "CenterLastValue": 'scaler',
    "RollingMean100thN": 'filter', 'CumSumTransformer': 'scaler', "MeanDifference": 'scaler',
    "FastICA": 'decomposition', 'convolution_filter': 'filter',
}
trans_classes_ov = {x: y for x, y in trans_classes_ov.items() if x in sorted_trans.index.tolist()}
trans_classes = sorted(trans_classes_ov.items(), key=lambda pair: sorted_trans.index.tolist().index(pair[0]))
trans_class_values = [x[1] for x in trans_classes]
# Map group colors to each method
trans_colors = [transformer_colors[group] for group in trans_class_values]

# Seaborn Methods Barplot
fig, ax = plt.subplots(figsize=(8, 6))

# Convert series to dataframe for seaborn compatibility
trans_methods_df = sorted_trans.reset_index()
trans_methods_df.columns = ['Method', 'Count']

sns.barplot(x='Method', y='Count', data=trans_methods_df, palette=trans_colors, edgecolor='k', ax=ax)

# Rotate x-labels for better readability
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')

# Annotate bar heights
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., int(p.get_height())),
                ha='center', va='bottom', fontsize=10, color='black')

# Add labels, title, and grid for clarity
ax.set_xlabel('Methods', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'Count of Transformer Methods Used in AutoTS by Type in M6, Top {top_n}', fontsize=14)
sns.despine(left=True, bottom=True)  # Removes the top and right borders for cleaner look
ax.grid(False, axis='x')  # Remove x-axis grid lines
ax.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.7)  # Add horizontal grid lines

# Create custom legend patches for each group
legend_patches = [Patch(color=color, label=group) for group, color in transformer_colors.items()]

# Add the legend with the custom patches
ax.legend(handles=legend_patches, loc='upper right', title='Groups', frameon=False)  # frameon=False removes the box around the legend

# Adjust layout
plt.tight_layout()
plt.savefig("count_transformers.png", dpi=300, bbox_inches="tight")
# plt.show()




#######################
### CHATGPT o1 preview generated


import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Read the CSV file
methods = pd.read_csv("/users/colincatlin/Downloads/m6/m6_all_saved_best.csv")

# Define the model classes
classes = {
    'ARDL': 'stat', 'DatepartRegression': 'ML',
    'ETS': 'stat', 'FBProphet': 'stat', 'GLM': 'stat', 'GLS': 'stat',
    'MAR': 'stat', 'MultivariateMotif': 'motif', 'MultivariateRegression': 'ML',
    'NVAR': 'stat', 'RRVAR': 'stat', 'SectionalMotif': 'motif', 'Theta': 'stat',
    'UnivariateMotif': 'motif', 'UnivariateRegression': 'ML',
    'VAR': 'stat', 'VECM': 'stat', 'WindowRegression': 'ML',
    'BallTreeMultivariateMotif': 'motif', 'MetricMotif': 'motif', 'ARCH': 'stat',
    'KalmanStateSpace': 'stat',
    # Include 'naive' models
    'AverageValueNaive': 'naive', 'ConstantNaive': 'naive', 'LastValueNaive': 'naive',
    'SeasonalNaive': 'naive', 'ZeroesNaive': 'naive',
}

# Initialize a list to collect transformer data
transformer_data = []

# Iterate over each method to extract model and transformer information
for index, row in methods.iterrows():
    current_params = json.loads(row['ModelParameters'])
    ModelParameters = current_params.copy()
    if 'series' in ModelParameters.keys():
        # Ensemble model
        series = ModelParameters['series']
        series = pd.DataFrame.from_dict(series, orient="index").reset_index(drop=False)
        if series.shape[1] > 2:
            # For mosaic style ensembles, choose the mode model id
            series.set_index(series.columns[0], inplace=True)
            series = series.mode(axis=1)[0].to_frame().reset_index(drop=False)
        series.columns = ['Series', 'ID']
        results = pd.Series(
            {
                x: current_params['models'][x]['Model']
                for x in current_params['models'].keys()
            }
        )
        results.name = "Model"
        series = series.merge(results, left_on="ID", right_index=True)
        series.columns = ["Series", "ID", 'Model']
        # Get transformers for each model
        lookup = {}
        for k, v in ModelParameters['models'].items():
            try:
                trans_params = json.loads(v.get('TransformationParameters', '{}'))
                transformations = trans_params.get('transformations', {})
                transformers_str = ",".join(transformations.values())
                lookup[k] = transformers_str
            except Exception:
                lookup[k] = "None"
        series['Transformers'] = series['ID'].replace(lookup).replace("", "None")
        # Collect data
        for idx, row_series in series.iterrows():
            model_name = row_series['Model']
            transformers_list = row_series['Transformers'].split(',')
            model_class = classes.get(model_name, 'naive')
            for transformer in transformers_list:
                if transformer == '':
                    transformer = 'None'
                transformer_data.append({'Model': model_name, 'ModelClass': model_class, 'Transformer': transformer})
    else:
        # Single model
        model_name = row['Model']
        model_class = classes.get(model_name, 'naive')
        try:
            trans_params = json.loads(row['TransformationParameters'])
            transformations = trans_params.get('transformations', {})
            transformers_list = transformations.values()
            for transformer in transformers_list:
                if transformer == '':
                    transformer = 'None'
                transformer_data.append({'Model': model_name, 'ModelClass': model_class, 'Transformer': transformer})
        except Exception:
            # No transformers
            transformer_data.append({'Model': model_name, 'ModelClass': model_class, 'Transformer': 'None'})

# Create a DataFrame from the collected data
transformer_df = pd.DataFrame(transformer_data)

# Calculate counts of transformers per model class
counts = transformer_df.groupby(['ModelClass', 'Transformer']).size().reset_index(name='Count')

# Get total counts per transformer to identify top transformers
total_transformer_counts = transformer_df.groupby('Transformer').size().reset_index(name='TotalCount')

# Select the top N transformers
top_n = 15
top_transformers = total_transformer_counts.sort_values(by='TotalCount', ascending=False).head(top_n)['Transformer'].tolist()

# Filter counts to include only top transformers
counts_top = counts[counts['Transformer'].isin(top_transformers)]

# Ensure the transformers are in the desired order
counts_top['Transformer'] = pd.Categorical(counts_top['Transformer'], categories=top_transformers, ordered=True)

# Define colors for model classes using a colorblind-friendly palette
model_class_colors = group_colors

# Update font sizes using rcParams for consistency
plt.rcParams.update({
    'font.size': 16,          # Base font size
    'axes.titlesize': 18,     # Title font size
    'axes.labelsize': 16,     # Axes labels font size
    'xtick.labelsize': 14,    # X-axis tick labels font size
    'ytick.labelsize': 14,    # Y-axis tick labels font size
    'legend.fontsize': 14,    # Legend font size
    'legend.title_fontsize': 16,  # Legend title font size
})

# Set the style for publication quality
sns.set_theme(style='whitegrid', context='paper', font_scale=1.8)

# Plot using seaborn
plt.figure(figsize=(12, 8))
barplot = sns.barplot(
    data=counts_top,
    x='Transformer',
    y='Count',
    hue='ModelClass',
    order=top_transformers,
    palette=model_class_colors,
    edgecolor='black'
)

# Customize plot appearance
plt.xticks(rotation=90)
plt.xlabel('Transformer')
plt.ylabel('Count')
plt.title('Count of Transformers Used by Model Class', fontweight='bold')
plt.legend(title='Model Class', fontsize=14, title_fontsize=16)

# Add faint horizontal grid lines
barplot.yaxis.grid(True, linestyle='--', linewidth=0.7, color='gray', alpha=0.7)
barplot.xaxis.grid(False)

# Remove top and right spines for a cleaner look
sns.despine(trim=True, left=True)

# Adjust layout to fit larger text
plt.tight_layout()

# Save the figure with high resolution
plt.savefig("transformer_counts_by_model_class.png", dpi=300, bbox_inches="tight")
# plt.show()




##########

import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Read the CSV file
methods = pd.read_csv("/users/colincatlin/Downloads/m6/m6_all_saved_best.csv")

# Define the model classes
classes = {
    'ARDL': 'stat', 'DatepartRegression': 'ML',
    'ETS': 'stat', 'FBProphet': 'stat', 'GLM': 'stat', 'GLS': 'stat',
    'MAR': 'stat', 'MultivariateMotif': 'motif', 'MultivariateRegression': 'ML',
    'NVAR': 'stat', 'RRVAR': 'stat', 'SectionalMotif': 'motif', 'Theta': 'stat',
    'UnivariateMotif': 'motif', 'UnivariateRegression': 'ML',
    'VAR': 'stat', 'VECM': 'stat', 'WindowRegression': 'ML',
    'BallTreeMultivariateMotif': 'motif', 'MetricMotif': 'motif', 'ARCH': 'stat',
    'KalmanStateSpace': 'stat',
    # Include 'naive' models
    'AverageValueNaive': 'naive', 'ConstantNaive': 'naive', 'LastValueNaive': 'naive',
    'SeasonalNaive': 'naive', 'ZeroesNaive': 'naive',
}

# Initialize a list to collect transformer data
transformer_data = []

# Variable to assign unique IDs to models
model_counter = 0

# Iterate over each method to extract model and transformer information
for index, row in methods.iterrows():
    current_params = json.loads(row['ModelParameters'])
    ModelParameters = current_params.copy()
    if 'series' in ModelParameters.keys():
        # Ensemble model
        series = ModelParameters['series']
        series = pd.DataFrame.from_dict(series, orient="index").reset_index(drop=False)
        if series.shape[1] > 2:
            # For mosaic style ensembles, choose the mode model id
            series.set_index(series.columns[0], inplace=True)
            series = series.mode(axis=1)[0].to_frame().reset_index(drop=False)
        series.columns = ['Series', 'ID']
        results = pd.Series(
            {
                x: current_params['models'][x]['Model']
                for x in current_params['models'].keys()
            }
        )
        results.name = "Model"
        series = series.merge(results, left_on="ID", right_index=True)
        series.columns = ["Series", "ID", 'Model']
        # Get transformers for each model
        lookup = {}
        for k, v in ModelParameters['models'].items():
            try:
                trans_params = json.loads(v.get('TransformationParameters', '{}'))
                transformations = trans_params.get('transformations', {})
                transformers_str = ",".join(transformations.values())
                lookup[k] = transformers_str
            except Exception:
                lookup[k] = "None"
        series['Transformers'] = series['ID'].replace(lookup).replace("", "None")
        # Collect data
        for idx, row_series in series.iterrows():
            model_name = row_series['Model']
            transformers_list = row_series['Transformers'].split(',')
            model_class = classes.get(model_name, 'naive')
            # Assign a unique model ID
            model_id = f"model_{model_counter}"
            model_counter += 1
            for transformer in transformers_list:
                if transformer == '':
                    transformer = 'None'
                transformer_data.append({
                    'ModelID': model_id,
                    'Model': model_name,
                    'ModelClass': model_class,
                    'Transformer': transformer
                })
    else:
        # Single model
        model_name = row['Model']
        model_class = classes.get(model_name, 'naive')
        # Assign a unique model ID
        model_id = f"model_{model_counter}"
        model_counter += 1
        try:
            trans_params = json.loads(row['TransformationParameters'])
            transformations = trans_params.get('transformations', {})
            transformers_list = transformations.values()
            for transformer in transformers_list:
                if transformer == '':
                    transformer = 'None'
                transformer_data.append({
                    'ModelID': model_id,
                    'Model': model_name,
                    'ModelClass': model_class,
                    'Transformer': transformer
                })
        except Exception:
            # No transformers
            transformer_data.append({
                'ModelID': model_id,
                'Model': model_name,
                'ModelClass': model_class,
                'Transformer': 'None'
            })

# Create a DataFrame from the collected data
transformer_df = pd.DataFrame(transformer_data)

# Calculate total unique models per model class
unique_models = transformer_df[['ModelClass', 'ModelID']].drop_duplicates()
model_class_counts = unique_models.groupby('ModelClass').size().reset_index(name='TotalModels')

# Calculate total models overall
total_models_overall = model_class_counts['TotalModels'].sum()

# Calculate counts of transformers per model class
counts = transformer_df.groupby(['ModelClass', 'Transformer']).size().reset_index(name='Count')

# Calculate proportion of models in each class that used each transformer
counts = counts.merge(model_class_counts, on='ModelClass')
counts['ProportionInClass'] = counts['Count'] / counts['TotalModels']

# Calculate proportion of each model class in the total models
model_class_counts['ClassProportion'] = model_class_counts['TotalModels'] / total_models_overall

# Merge ClassProportion into counts
counts = counts.merge(model_class_counts[['ModelClass', 'ClassProportion']], on='ModelClass', suffixes=('', '_y'))

# Calculate adjusted proportion
counts['AdjustedProportion'] = counts['ProportionInClass'] * counts['ClassProportion']

# For each transformer, normalize the adjusted proportions so that they sum to 1
counts['NormalizedProportion'] = counts.groupby('Transformer')['AdjustedProportion'].transform(lambda x: x / x.sum())

# Select the top N transformers based on total usage
top_n = 15
total_transformer_counts = transformer_df.groupby('Transformer')['ModelID'].nunique().reset_index(name='TotalCount')
top_transformers = total_transformer_counts.sort_values(by='TotalCount', ascending=False).head(top_n)['Transformer'].tolist()

# Filter counts to include only top transformers
counts_top = counts[counts['Transformer'].isin(top_transformers)]

# Ensure the transformers are in the desired order
counts_top['Transformer'] = pd.Categorical(counts_top['Transformer'], categories=top_transformers, ordered=True)

# Define colors for model classes using a colorblind-friendly palette
model_class_colors = group_colors
# Update font sizes using rcParams for consistency
plt.rcParams.update({
    'font.size': 16,          # Base font size
    'axes.titlesize': 18,     # Title font size
    'axes.labelsize': 16,     # Axes labels font size
    'xtick.labelsize': 14,    # X-axis tick labels font size
    'ytick.labelsize': 14,    # Y-axis tick labels font size
    'legend.fontsize': 14,    # Legend font size
    'legend.title_fontsize': 16,  # Legend title font size
})

# Set the style for publication quality
sns.set_theme(style='whitegrid', context='paper', font_scale=1.8)

# Pivot the data for plotting
plot_data = counts_top.pivot_table(
    index='Transformer',
    columns='ModelClass',
    values='NormalizedProportion',
    fill_value=0
).reindex(index=top_transformers)

# Plot using seaborn
fig, ax = plt.subplots(figsize=(12, 8))
bottoms = [0]*len(plot_data)
transformer_indices = range(len(plot_data))
for model_class in plot_data.columns:
    proportions = plot_data[model_class].values
    bars = ax.bar(
        transformer_indices,
        proportions,
        bottom=bottoms,
        color=model_class_colors.get(model_class, '#333333'),
        edgecolor='black',
        label=model_class
    )
    # Add percentage labels
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0.02:  # Only label if the segment is larger than 2%
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottoms[idx] + height / 2,
                f'{height*100:.1f}%',
                ha='center',
                va='center',
                fontsize=10,
                color='white',
                alpha=0.99
            )
    bottoms = [sum(x) for x in zip(bottoms, proportions)]

# Customize plot appearance
ax.set_xticks(transformer_indices)
ax.set_xticklabels(plot_data.index, rotation=90)
ax.set_xlabel('Transformer')
ax.set_ylabel('Normalized Proportion')
ax.set_title('Normalized Transformer Usage Adjusted by Model Class Occurrence', fontweight='bold')

# Add faint horizontal grid lines
ax.yaxis.grid(False, linestyle='--', linewidth=0.7, color='gray', alpha=0.7)
ax.xaxis.grid(False)

# Remove top and right spines for a cleaner look
sns.despine(trim=True, left=True)

# Adjust y-axis to show percentages
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:%}'.format(y)))

# Adjust legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles=handles,
    labels=labels,
    title='Model Class',
    fontsize=14,
    title_fontsize=16,
    loc='upper right'
)

# Adjust layout to fit larger text
plt.tight_layout()

# Save the figure with high resolution
plt.savefig("normalized_transformer_usage_with_percentages.png", dpi=300, bbox_inches="tight")
# plt.show()









###########
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Read the CSV file
methods = pd.read_csv("/users/colincatlin/Downloads/m6/m6_all_saved_best.csv")

# Define the model classes
classes = {
    'ARDL': 'stat', 'DatepartRegression': 'ML',
    'ETS': 'stat', 'FBProphet': 'stat', 'GLM': 'stat', 'GLS': 'stat',
    'MAR': 'stat', 'MultivariateMotif': 'motif', 'MultivariateRegression': 'ML',
    'NVAR': 'stat', 'RRVAR': 'stat', 'SectionalMotif': 'motif', 'Theta': 'stat',
    'UnivariateMotif': 'motif', 'UnivariateRegression': 'ML',
    'VAR': 'stat', 'VECM': 'stat', 'WindowRegression': 'ML',
    'BallTreeMultivariateMotif': 'motif', 'MetricMotif': 'motif', 'ARCH': 'stat',
    'KalmanStateSpace': 'stat',
    # Include 'naive' models
    'AverageValueNaive': 'naive', 'ConstantNaive': 'naive', 'LastValueNaive': 'naive',
    'SeasonalNaive': 'naive', 'ZeroesNaive': 'naive',
}

# Initialize a list to collect transformer data
transformer_data = []

# Variable to assign unique IDs to models
model_counter = 0

# Iterate over each method to extract model and transformer information
for index, row in methods.iterrows():
    current_params = json.loads(row['ModelParameters'])
    ModelParameters = current_params.copy()
    if 'series' in ModelParameters.keys():
        # Ensemble model
        series = ModelParameters['series']
        series = pd.DataFrame.from_dict(series, orient="index").reset_index(drop=False)
        if series.shape[1] > 2:
            # For mosaic style ensembles, choose the mode model id
            series.set_index(series.columns[0], inplace=True)
            series = series.mode(axis=1)[0].to_frame().reset_index(drop=False)
        series.columns = ['Series', 'ID']
        results = pd.Series(
            {
                x: current_params['models'][x]['Model']
                for x in current_params['models'].keys()
            }
        )
        results.name = "Model"
        series = series.merge(results, left_on="ID", right_index=True)
        series.columns = ["Series", "ID", 'Model']
        # Get transformers for each model
        lookup = {}
        for k, v in ModelParameters['models'].items():
            try:
                trans_params = json.loads(v.get('TransformationParameters', '{}'))
                transformations = trans_params.get('transformations', {})
                transformers_str = ",".join(transformations.values())
                lookup[k] = transformers_str
            except Exception:
                lookup[k] = "None"
        series['Transformers'] = series['ID'].replace(lookup).replace("", "None")
        # Collect data
        for idx, row_series in series.iterrows():
            model_name = row_series['Model']
            transformers_list = row_series['Transformers'].split(',')
            model_class = classes.get(model_name, 'naive')
            # Assign a unique model ID
            model_id = f"model_{model_counter}"
            model_counter += 1
            for transformer in transformers_list:
                if transformer == '':
                    transformer = 'None'
                transformer_data.append({
                    'ModelID': model_id,
                    'Model': model_name,
                    'ModelClass': model_class,
                    'Transformer': transformer
                })
    else:
        # Single model
        model_name = row['Model']
        model_class = classes.get(model_name, 'naive')
        # Assign a unique model ID
        model_id = f"model_{model_counter}"
        model_counter += 1
        try:
            trans_params = json.loads(row['TransformationParameters'])
            transformations = trans_params.get('transformations', {})
            transformers_list = transformations.values()
            for transformer in transformers_list:
                if transformer == '':
                    transformer = 'None'
                transformer_data.append({
                    'ModelID': model_id,
                    'Model': model_name,
                    'ModelClass': model_class,
                    'Transformer': transformer
                })
        except Exception:
            # No transformers
            transformer_data.append({
                'ModelID': model_id,
                'Model': model_name,
                'ModelClass': model_class,
                'Transformer': 'None'
            })

# Create a DataFrame from the collected data
transformer_df = pd.DataFrame(transformer_data)

# Calculate total unique models per model class
unique_models = transformer_df[['ModelClass', 'ModelID']].drop_duplicates()
model_class_counts = unique_models.groupby('ModelClass').size().reset_index(name='TotalModels')

# Calculate total models overall
total_models_overall = model_class_counts['TotalModels'].sum()

# Calculate counts of transformers per model class
counts = transformer_df.groupby(['ModelClass', 'Transformer']).size().reset_index(name='Count')

# Calculate proportion of models in each class that used each transformer
counts = counts.merge(model_class_counts, on='ModelClass')
counts['ProportionInClass'] = counts['Count'] / counts['TotalModels']

# Calculate proportion of each model class in the total models
model_class_counts['ClassProportion'] = model_class_counts['TotalModels'] / total_models_overall

# Merge ClassProportion into counts
counts = counts.merge(model_class_counts[['ModelClass', 'ClassProportion']], on='ModelClass', suffixes=('', '_y'))

# Calculate adjusted proportion
counts['AdjustedProportion'] = counts['ProportionInClass'] * counts['ClassProportion']

# For each transformer, normalize the adjusted proportions so that they sum to 1
counts['NormalizedProportion'] = counts.groupby('Transformer')['AdjustedProportion'].transform(lambda x: x / x.sum())

# Select the top N transformers based on total usage
top_n = 15
total_transformer_counts = transformer_df.groupby('Transformer')['ModelID'].nunique().reset_index(name='TotalCount')
top_transformers = total_transformer_counts.sort_values(by='TotalCount', ascending=False).head(top_n)['Transformer'].tolist()

# Filter counts to include only top transformers
counts_top = counts[counts['Transformer'].isin(top_transformers)]

# Ensure the transformers are in the desired order
counts_top['Transformer'] = pd.Categorical(counts_top['Transformer'], categories=top_transformers, ordered=True)

# Define pastel colors for model classes
model_class_colors = group_colors

# Update font sizes using rcParams for consistency
plt.rcParams.update({
    'font.size': 16,          # Base font size
    'axes.titlesize': 18,     # Title font size
    'axes.labelsize': 16,     # Axes labels font size
    'xtick.labelsize': 14,    # X-axis tick labels font size
    'ytick.labelsize': 14,    # Y-axis tick labels font size
    'legend.fontsize': 14,    # Legend font size
    'legend.title_fontsize': 16,  # Legend title font size
})

# Set the style for publication quality
sns.set_theme(style='whitegrid', context='paper', font_scale=1.8)

# Pivot the data for plotting
plot_data = counts_top.pivot_table(
    index='Transformer',
    columns='ModelClass',
    values='NormalizedProportion',
    fill_value=0
).reindex(index=top_transformers)

# Plot using custom bar plot to add percentage labels
fig, ax = plt.subplots(figsize=(12, 8))
bottoms = [0]*len(plot_data)
transformer_indices = range(len(plot_data))
for model_class in plot_data.columns:
    proportions = plot_data[model_class].values
    bars = ax.bar(
        transformer_indices,
        proportions,
        bottom=bottoms,
        color=model_class_colors.get(model_class, '#333333'),
        edgecolor='black',
        label=model_class
    )
    # Add percentage labels
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0.025:  # Only label if the segment is larger than 1.5%
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottoms[idx] + height / 2,
                f'{round(height*100)}%',
                ha='center',
                va='center',
                fontsize=12,
                color='black',
                alpha=0.5,
            )
    bottoms = [sum(x) for x in zip(bottoms, proportions)]

# Customize plot appearance
ax.set_xticks(transformer_indices)
ax.set_xticklabels(plot_data.index, rotation=90)
ax.set_xlabel('Transformer')
ax.set_ylabel('Frequency Adjusted Proportion')
ax.set_title('Transformer Usage by Model Class', fontweight='bold')

# Remove y-axis grid lines
ax.yaxis.grid(False)
ax.xaxis.grid(False)

# Remove top and right spines for a cleaner look
sns.despine(trim=True, left=True)

# Adjust y-axis to show percentages rounded to whole numbers
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(round(y*100)/100)))

# Adjust legend position to avoid blocking bars
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles=handles,
    labels=labels,
    title='Model Class',
    fontsize=14,
    title_fontsize=16,
    loc='center left',
    bbox_to_anchor=(1, 0.5),
    frameon=False
)

# Adjust layout to fit larger text and legend
plt.tight_layout(rect=[0, 0, 0.92, 1])  # Adjust right margin to make space for legend

# Save the figure with high resolution
plt.savefig("normalized_transformer_usage_with_percentages.png", dpi=300, bbox_inches="tight")
# plt.show()
