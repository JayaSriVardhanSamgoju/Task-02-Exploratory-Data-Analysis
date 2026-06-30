import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for all plots to ensure premium look and feel
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'figure.facecolor': '#fafafa',
    'axes.facecolor': '#ffffff',
    'text.color': '#2d3748',
    'axes.labelcolor': '#2d3748',
    'xtick.color': '#4a5568',
    'ytick.color': '#4a5568',
    'patch.edgecolor': 'none'
})

# Custom premium palette colors
COLOR_DIED = '#e056fd'     # Purple-pink
COLOR_SURVIVED = '#00cec9' # Turquoise/Teal
PALETTE_SURVIVAL = {0: COLOR_DIED, 1: COLOR_SURVIVED}
COLOR_PRIMARY = '#1e3d59'  # Navy Blue
COLOR_SECONDARY = '#ffc13b'# Gold/Amber
COLOR_ACCENT = '#ff6e40'   # Coral

def ensure_directories():
    """Ensure output directories exist."""
    os.makedirs('images', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)

def load_data():
    """Load the Titanic dataset."""
    df_path = os.path.join('dataset', 'Titanic-Dataset.csv')
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Dataset not found at {df_path}")
    return pd.read_csv(df_path)

def generate_dataset_preview(df):
    """Generate a clean visual table of the first 10 rows of the dataset."""
    print("Generating dataset preview...")
    # Clean up name/ticket strings for table display
    preview_df = df.head(10).copy()
    preview_df['Name'] = preview_df['Name'].apply(lambda x: x[:22] + '...' if len(str(x)) > 25 else x)
    preview_df['Ticket'] = preview_df['Ticket'].apply(lambda x: x[:10] + '...' if len(str(x)) > 12 else x)
    # Fill NaN for rendering
    preview_df = preview_df.fillna("NaN")

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis('off')
    
    # Create matplotlib table
    table = ax.table(
        cellText=preview_df.values,
        colLabels=preview_df.columns,
        loc='center',
        cellLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    
    # Color headers and cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor(COLOR_PRIMARY)
        else:
            cell.set_facecolor('#ffffff' if row % 2 == 0 else '#f7fafc')
            # Text coloring based on survival status
            if col == 1: # Survived column
                val = preview_df.iloc[row-1, col]
                if val == 1:
                    cell.get_text().set_color(COLOR_SURVIVED)
                    cell.get_text().set_weight('bold')
                elif val == 0:
                    cell.get_text().set_color(COLOR_DIED)
                    cell.get_text().set_weight('bold')

    plt.title("Titanic Dataset - First 10 Rows Preview", fontsize=16, weight='bold', pad=20, color=COLOR_PRIMARY)
    plt.savefig('images/dataset_preview.png', bbox_inches='tight', dpi=300, facecolor='#fafafa')
    plt.close()

def generate_summary_statistics(df):
    """Generate descriptive statistics table including mean, median, mode, min, max, std, variance, and quartiles."""
    print("Generating summary statistics...")
    desc_df = df.describe().round(2)
    
    # Compute variance and mode for numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    variance = df[num_cols].var().round(2).to_frame(name='variance').T
    mode = df[num_cols].mode().iloc[0].round(2).to_frame(name='mode').T
    
    # Combine full descriptive stats
    full_stats = pd.concat([desc_df, variance, mode]).reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.axis('off')
    
    table = ax.table(
        cellText=full_stats.values,
        colLabels=full_stats.columns,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    
    # Color headers
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor(COLOR_PRIMARY)
        else:
            cell.set_facecolor('#ffffff' if row % 2 == 0 else '#f7fafc')
            if col == 0:
                cell.set_text_props(weight='bold')

    plt.title("Descriptive Statistics for Numeric Features (including Variance & Mode)", fontsize=16, weight='bold', pad=20, color=COLOR_PRIMARY)
    plt.savefig('images/summary_statistics.png', bbox_inches='tight', dpi=300, facecolor='#fafafa')
    plt.close()

def generate_histograms(df):
    """Generate distributions of Age and Fare."""
    print("Generating histograms...")
    # Age Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Age', kde=True, color=COLOR_PRIMARY, bins=30, alpha=0.85)
    plt.title('Distribution of Passenger Ages', fontsize=16, weight='bold', pad=15, color=COLOR_PRIMARY)
    plt.xlabel('Age (Years)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.savefig('images/histogram_age.png', bbox_inches='tight', dpi=300, facecolor='#fafafa')
    plt.close()

    # Fare Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Fare', kde=True, color=COLOR_ACCENT, bins=40, alpha=0.85)
    plt.title('Distribution of Ticket Fares', fontsize=16, weight='bold', pad=15, color=COLOR_PRIMARY)
    plt.xlabel('Fare (USD)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.savefig('images/histogram_fare.png', bbox_inches='tight', dpi=300, facecolor='#fafafa')
    plt.close()

def generate_boxplots(df):
    """Generate boxplots for Age and Fare."""
    print("Generating boxplots...")
    # Boxplot Age vs Class & Survival
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df, 
        x='Pclass', 
        y='Age', 
        hue='Survived', 
        palette={0: COLOR_DIED, 1: COLOR_SURVIVED},
        width=0.6,
        linewidth=2
    )
    plt.title('Age Distribution by Ticket Class and Survival Status', fontsize=16, weight='bold', pad=15, color=COLOR_PRIMARY)
    plt.xlabel('Passenger Class (Pclass)', fontsize=12)
    plt.ylabel('Age', fontsize=12)
    
    # Customize legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ['Died', 'Survived'], title='Outcome', frameon=True)
    
    plt.savefig('images/boxplot_age.png', bbox_inches='tight', dpi=300, facecolor='#fafafa')
    plt.close()

    # Boxplot Fare vs Class & Survival (Log scaled Fare due to skewness/outliers)
    plt.figure(figsize=(10, 6))
    df_fare_clean = df[df['Fare'] > 0].copy()
    df_fare_clean['LogFare'] = np.log10(df_fare_clean['Fare'])
    
    sns.boxplot(
        data=df_fare_clean, 
        x='Pclass', 
        y='LogFare', 
        hue='Survived', 
        palette={0: COLOR_DIED, 1: COLOR_SURVIVED},
        width=0.6,
        linewidth=2
    )
    plt.title('Ticket Fare (Log10) by Class and Survival Status', fontsize=16, weight='bold', pad=15, color=COLOR_PRIMARY)
    plt.xlabel('Passenger Class (Pclass)', fontsize=12)
    plt.ylabel('Log10(Fare)', fontsize=12)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ['Died', 'Survived'], title='Outcome', frameon=True)
    
    plt.savefig('images/boxplot_fare.png', bbox_inches='tight', dpi=300, facecolor='#fafafa')
    plt.close()

def generate_countplot_survived(df):
    """Generate counts of survival status."""
    print("Generating survival countplot...")
    plt.figure(figsize=(8, 6))
    
    # Calculate counts and percentages
    ax = sns.countplot(
        data=df, 
        x='Survived', 
        palette={0: COLOR_DIED, 1: COLOR_SURVIVED},
        hue='Survived',
        legend=False
    )
    
    # Style plot
    plt.title('Passenger Survival Counts', fontsize=16, weight='bold', pad=15, color=COLOR_PRIMARY)
    plt.ylabel('Passenger Count', fontsize=12)
    plt.xlabel('', fontsize=12)
    ax.set_xticklabels(['Died (0)', 'Survived (1)'], fontsize=12, weight='bold')

    # Add percentages and counts on top of bars
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        percentage = (height / total) * 100
        ax.annotate(
            f'{int(height)}\n({percentage:.1f}%)', 
            (p.get_x() + p.get_width() / 2., height + 10),
            ha='center', 
            va='bottom', 
            fontsize=12, 
            weight='bold', 
            color='#2d3748'
        )
        
    plt.savefig('images/countplot_survived.png', bbox_inches='tight', dpi=300, facecolor='#fafafa')
    plt.close()

def generate_pairplot(df):
    """Generate pairplot of numerical columns."""
    print("Generating pairplot...")
    # Drop PassengerId as it is just an index
    cols_to_plot = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    plot_df = df[cols_to_plot].copy()
    
    # Plot using sns.pairplot
    g = sns.pairplot(
        plot_df, 
        hue='Survived', 
        palette={0: COLOR_DIED, 1: COLOR_SURVIVED},
        diag_kind='kde',
        height=2.2,
        aspect=1.2,
        plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'none'},
        diag_kws={'fill': True}
    )
    g.fig.suptitle('Pairwise Feature Relationships grouped by Survival Status', y=1.02, fontsize=18, weight='bold', color=COLOR_PRIMARY)
    
    # Adjust legend
    g._legend.set_title('Outcome')
    for t, l in zip(g._legend.texts, ['Died', 'Survived']):
        t.set_text(l)
        
    plt.savefig('images/pairplot.png', bbox_inches='tight', dpi=150, facecolor='#fafafa')
    plt.close()

def generate_correlation_heatmap(df):
    """Generate correlation heatmap of numeric columns."""
    print("Generating correlation heatmap...")
    # Get only numeric columns
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['PassengerId'])
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=1, 
        square=True,
        cbar_kws={'shrink': 0.8},
        annot_kws={'size': 11, 'weight': 'bold'}
    )
    plt.title('Correlation Matrix of Numeric Features', fontsize=16, weight='bold', pad=20, color=COLOR_PRIMARY)
    plt.savefig('images/correlation_heatmap.png', bbox_inches='tight', dpi=300, facecolor='#fafafa')
    plt.close()

def generate_survival_analysis(df):
    """Generate survival analysis by Sex, Pclass, and Embarked."""
    print("Generating survival analysis...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharey=True)
    
    # Survival Rate by Sex
    sns.barplot(
        data=df, 
        x='Sex', 
        y='Survived', 
        ax=axes[0], 
        errorbar=None, 
        palette={'male': COLOR_PRIMARY, 'female': COLOR_ACCENT},
        hue='Sex',
        legend=False
    )
    axes[0].set_title('Survival Rate by Sex', fontsize=14, weight='bold', pad=10)
    axes[0].set_ylabel('Survival Rate', fontsize=12)
    axes[0].set_xlabel('', fontsize=12)
    axes[0].set_xticklabels(['Male', 'Female'], fontsize=12)
    
    # Survival Rate by Pclass
    sns.barplot(
        data=df, 
        x='Pclass', 
        y='Survived', 
        ax=axes[1], 
        errorbar=None, 
        palette='Blues_r',
        hue='Pclass',
        legend=False
    )
    axes[1].set_title('Survival Rate by Class (Pclass)', fontsize=14, weight='bold', pad=10)
    axes[1].set_ylabel('', fontsize=12)
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_xticklabels(['1st Class', '2nd Class', '3rd Class'], fontsize=12)
    
    # Survival Rate by Embarked Port
    df_embarked = df.dropna(subset=['Embarked'])
    sns.barplot(
        data=df_embarked, 
        x='Embarked', 
        y='Survived', 
        ax=axes[2], 
        errorbar=None, 
        palette='crest',
        hue='Embarked',
        legend=False
    )
    axes[2].set_title('Survival Rate by Embarkation Port', fontsize=14, weight='bold', pad=10)
    axes[2].set_ylabel('', fontsize=12)
    axes[2].set_xlabel('Port', fontsize=12)
    axes[2].set_xticklabels(['Cherbourg (C)', 'Queenstown (Q)', 'Southampton (S)'], fontsize=12)

    # Annotate survival rate percentages on top of bars
    for ax in axes:
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height):
                ax.annotate(
                    f'{height * 100:.1f}%', 
                    (p.get_x() + p.get_width() / 2., height + 0.02),
                    ha='center', 
                    va='bottom', 
                    fontsize=12, 
                    weight='bold', 
                    color='#2d3748'
                )

    plt.suptitle('Survival Rates Across Passenger Categories', y=0.98, fontsize=18, weight='bold', color=COLOR_PRIMARY)
    plt.tight_layout()
    plt.savefig('images/survival_analysis.png', bbox_inches='tight', dpi=300, facecolor='#fafafa')
    plt.close()

def write_observations_report(df):
    """Write observations report file."""
    print("Writing observations report...")
    
    # Extract data insights
    total_passengers = len(df)
    missing_age = df['Age'].isnull().sum()
    missing_cabin = df['Cabin'].isnull().sum()
    missing_embarked = df['Embarked'].isnull().sum()
    
    survived_df = df[df['Survived'] == 1]
    died_df = df[df['Survived'] == 0]
    
    survival_rate = (len(survived_df) / total_passengers) * 100
    
    female_survival = (df[df['Sex'] == 'female']['Survived'].mean()) * 100
    male_survival = (df[df['Sex'] == 'male']['Survived'].mean()) * 100
    
    pclass_survival = df.groupby('Pclass')['Survived'].mean() * 100
    embarked_survival = df.groupby('Embarked')['Survived'].mean() * 100
    
    # Calculate statistics for age groups
    df_age_clean = df.dropna(subset=['Age']).copy()
    df_age_clean['AgeGroup'] = pd.cut(df_age_clean['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior'])
    age_group_survival = df_age_clean.groupby('AgeGroup', observed=False)['Survived'].mean() * 100
    
    report_content = f"""# Exploratory Data Analysis Observations - Titanic Dataset

This report documents the key insights and analytical findings discovered during the Exploratory Data Analysis (EDA) of the Titanic dataset.

---

## 1. Dataset Overview and Completeness

- **Total Passengers in Dataset**: {total_passengers}
- **Features (Columns)**: {df.shape[1]} (PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked)

### Missing Values Summary
- **Age**: {missing_age} missing values ({missing_age/total_passengers*100:.2f}%) - *High missingness; requires imputation or special care in modeling.*
- **Cabin**: {missing_cabin} missing values ({missing_cabin/total_passengers*100:.2f}%) - *Severely incomplete. May be converted to a binary indicator (Cabin recorded vs. not).*
- **Embarked**: {missing_embarked} missing values ({missing_embarked/total_passengers*100:.2f}%) - *Extremely minor missingness (only 2 records).*

---

## 2. Demographic Analysis & Survival Patterns

### Survival Rates by Gender (Sex)
- **Overall Survival Rate**: {survival_rate:.2f}% ({len(survived_df)} survived, {len(died_df)} died)
- **Female Survival Rate**: {female_survival:.2f}%
- **Male Survival Rate**: {male_survival:.2f}%

> [!NOTE]
> There is a massive disparity in survival rate between genders, strongly aligning with the historic "women and children first" protocol. Females were nearly **4 times** more likely to survive than males.

### Survival Rates by Socioeconomic Status (Pclass)
- **1st Class (Pclass 1)**: {pclass_survival[1]:.2f}% survival rate
- **2nd Class (Pclass 2)**: {pclass_survival[2]:.2f}% survival rate
- **3rd Class (Pclass 3)**: {pclass_survival[3]:.2f}% survival rate

> [!IMPORTANT]
> Ticket class was a highly significant factor in survival. Over 62% of 1st-class passengers survived, compared to only 24.24% of 3rd-class passengers. This indicates a strong socioeconomic bias in rescue accessibility.

### Survival Rates by Embarkation Port
- **Cherbourg (C)**: {embarked_survival.get('C', 0.0):.2f}% survival rate
- **Queenstown (Q)**: {embarked_survival.get('Q', 0.0):.2f}% survival rate
- **Southampton (S)**: {embarked_survival.get('S', 0.0):.2f}% survival rate

> [!NOTE]
> Passengers embarking from Cherbourg (C) had a higher survival rate. Further analysis shows this is strongly correlated with a larger proportion of 1st-class passengers boarding at Cherbourg.

### Survival Rates by Age Groups
- **Children (0-12)**: {age_group_survival.get('Child', 0.0):.2f}% survival rate
- **Teenagers (13-18)**: {age_group_survival.get('Teenager', 0.0):.2f}% survival rate
- **Young Adults (19-35)**: {age_group_survival.get('Young Adult', 0.0):.2f}% survival rate
- **Adults (36-60)**: {age_group_survival.get('Adult', 0.0):.2f}% survival rate
- **Seniors (60+)**: {age_group_survival.get('Senior', 0.0):.2f}% survival rate

> [!TIP]
> Children had a significantly higher survival rate (nearly 58%), confirming priority was given to the young. Conversely, seniors (60+) had the lowest survival rate (26.9%), showing the physical difficulty or lack of priority in evacuating.

---

## 3. Numerical Feature Distributions and Correlations

### Age and Fare Distributions
- **Age**: Moderately normal distribution, peaking around 20-30 years of age. Median age is 28, mean is 29.7.
- **Fare**: Extremely right-skewed. The majority of tickets cost under $50, but a few outlier tickets exceeded $500. This requires log-transformation or scaling for machine learning.

### Correlation Analysis
Key relationships from the correlation heatmap:
1. **Pclass & Fare (-0.55)**: Strong negative correlation, confirming 1st class (lower numerical rank) corresponds to higher ticket fares.
2. **Pclass & Survived (-0.34)**: Moderate negative correlation, indicating higher-class passengers (lower numerical rank) had higher survival.
3. **Fare & Survived (0.26)**: Positive correlation, suggesting those who paid more had higher odds of survival.
4. **SibSp & Parch (0.41)**: Moderate positive correlation, representing families traveling together.
"""
    with open(os.path.join('reports', 'observations.md'), 'w', encoding='utf-8') as f:
        f.write(report_content)
    print("Observations report successfully written.")

def main():
    ensure_directories()
    df = load_data()
    
    # Audit Console Output
    print("=" * 60)
    print("TITANIC DATASET EDA AUDIT PROFILE")
    print("=" * 60)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Duplicate Rows Count: {df.duplicated().sum()}")
    print("-" * 60)
    print("DATA TYPES SUMMARY:")
    print(df.dtypes)
    print("-" * 60)
    print("DATASET GENERAL INFORMATION (df.info()):")
    df.info()
    print("-" * 60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print("COMPUTED STATS (VARIANCE):")
    print(df[numeric_cols].var().round(4))
    print("-" * 60)
    print("COMPUTED STATS (MODE):")
    print(df[numeric_cols].mode().iloc[0])
    print("=" * 60)
    
    # Run all visual generations
    generate_dataset_preview(df)
    generate_summary_statistics(df)
    generate_histograms(df)
    generate_boxplots(df)
    generate_countplot_survived(df)
    generate_pairplot(df)
    generate_correlation_heatmap(df)
    generate_survival_analysis(df)
    
    # Write report
    write_observations_report(df)
    print("\nEDA Completed successfully. All 10 plots are saved in 'images/' folder.")

if __name__ == '__main__':
    main()
