from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('talk')

ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()


def find_project_root():
    candidates = [Path.cwd(), ROOT]
    for candidate in candidates:
        if (candidate / 'master_jobs_dataset.csv').exists():
            return candidate
        for parent in [candidate, *candidate.parents]:
            if (parent / 'master_jobs_dataset.csv').exists():
                return parent
    return ROOT


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text):
    return [token for token in normalize_text(text).split() if token]


def build_text_for_nlp(row):
    parts = [
        row.get('job_title', ''),
        row.get('category', ''),
        row.get('description', ''),
        row.get('tags', ''),
        row.get('location', ''),
        row.get('department', ''),
        row.get('workplace', ''),
        row.get('type', ''),
    ]
    cleaned = []
    seen = set()
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        for token in normalize_text(text).split():
            if token not in seen:
                seen.add(token)
                cleaned.append(token)
    return ' '.join(cleaned)


def remove_stopwords(tokens):
    stop_words = {
        'the', 'and', 'for', 'with', 'in', 'on', 'of', 'to', 'a', 'an', 'is', 'are', 'be',
        'this', 'that', 'our', 'you', 'your', 'will', 'work', 'jobs', 'job', 'remote', 'team',
        'company', 'role', 'skills', 'experience', 'using', 'developing', 'develop', 'data',
        'science', 'engineer', 'engineering', 'software', 'developer', 'analyst', 'manager',
        'product', 'business', 'technical', 'senior', 'junior', 'lead', 'principal', 'full',
        'time', 'at', 'we', 'can', 'help', 'new', 'up', 'from', 'into', 'their', 'about'
    }
    return [token for token in tokens if token not in stop_words and len(token) > 2]


def parse_salary_value(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if not text:
        return np.nan
    nums = re.findall(r'\d+(?:\.\d+)?', text)
    if not nums:
        return np.nan
    amount = float(nums[0])
    if 'k' in text:
        amount *= 1000
    if 'm' in text:
        amount *= 1000000
    return amount


def estimate_salary(row, index=None):
    title = str(row.get('job_title', '')).lower()
    category = str(row.get('category', '')).lower()
    description = str(row.get('description', '')).lower()
    location = str(row.get('location', '')).lower()
    tags = str(row.get('tags', '')).lower()
    text = f"{title} {category} {description} {location} {tags}"

    row_index = int(index) if index is not None else int(row.name)
    variation = ((row_index % 7) - 3) * 3500

    if 'business analyst' in text:
        base = 78000
        if any(keyword in text for keyword in ['senior', 'lead', 'principal', 'manager']):
            base += 18000
        elif any(keyword in text for keyword in ['junior', 'entry', 'associate']):
            base -= 12000
        if any(keyword in text for keyword in ['finance', 'strategy', 'product', 'operations']):
            base += 8000
        elif any(keyword in text for keyword in ['data', 'bi', 'reporting']):
            base += 4000
        if any(keyword in text for keyword in ['united states', 'usa', 'canada', 'london', 'uk', 'singapore', 'switzerland', 'netherlands', 'germany']):
            base += 6000
        elif any(keyword in text for keyword in ['india', 'pakistan', 'philippines', 'egypt', 'brazil', 'mexico']):
            base -= 5000
        if any(keyword in text for keyword in ['remote', 'hybrid', 'worldwide', 'anywhere']):
            base -= 3000
        return base + variation

    if any(keyword in text for keyword in ['data scientist', 'machine learning', 'ai', 'nlp', 'deep learning']):
        base = 140000
        if any(keyword in text for keyword in ['senior', 'lead', 'principal', 'manager']):
            base += 18000
        elif any(keyword in text for keyword in ['junior', 'entry', 'associate']):
            base -= 12000
        return base + variation

    if any(keyword in text for keyword in ['devops', 'cloud', 'platform', 'site reliability', 'infrastructure']):
        base = 125000
        if any(keyword in text for keyword in ['senior', 'lead', 'principal']):
            base += 14000
        elif any(keyword in text for keyword in ['junior', 'entry']):
            base -= 10000
        return base + variation

    if any(keyword in text for keyword in ['software engineer', 'backend', 'frontend', 'full stack', 'developer']):
        base = 115000
        if any(keyword in text for keyword in ['senior', 'lead', 'principal']):
            base += 16000
        elif any(keyword in text for keyword in ['junior', 'entry', 'associate']):
            base -= 12000
        return base + variation

    if any(keyword in text for keyword in ['product manager', 'design', 'ux']):
        base = 105000
        if any(keyword in text for keyword in ['senior', 'lead']):
            base += 14000
        elif any(keyword in text for keyword in ['junior', 'entry']):
            base -= 10000
        return base + variation

    if any(keyword in text for keyword in ['data analyst', 'analyst', 'operations', 'bi analyst']):
        base = 85000
        if any(keyword in text for keyword in ['senior', 'lead']):
            base += 12000
        elif any(keyword in text for keyword in ['junior', 'entry']):
            base -= 9000
        return base + variation

    return 75000 + variation


def extract_skills(text):
    text = str(text).lower()
    skill_keywords = ['python', 'sql', 'aws', 'docker', 'nlp', 'tensorflow', 'pytorch', 'kubernetes', 'azure', 'spark', 'tableau', 'power bi']
    return [skill for skill in skill_keywords if skill in text]


def classify_role(title):
    text = str(title).lower()
    if any(keyword in text for keyword in ['data scientist', 'machine learning', 'ai', 'nlp', 'deep learning']):
        return 'AI / ML'
    if any(keyword in text for keyword in ['devops', 'cloud', 'platform', 'site reliability', 'infrastructure']):
        return 'DevOps / Cloud'
    if any(keyword in text for keyword in ['software engineer', 'backend', 'frontend', 'full stack', 'developer']):
        return 'Software Engineering'
    if 'business analyst' in text:
        return 'Business Analysis'
    if any(keyword in text for keyword in ['data analyst', 'bi analyst', 'reporting', 'analytics']):
        return 'Data Analytics'
    if any(keyword in text for keyword in ['product manager', 'ux', 'design']):
        return 'Product / Design'
    if any(keyword in text for keyword in ['operations', 'strategy']):
        return 'Operations / Strategy'
    return 'Other'


def sentiment_score(text):
    text = str(text).lower()
    positive_words = ['growth', 'remote', 'flexible', 'innovative', 'learning', 'benefit', 'opportunity', 'great']
    demanding_words = ['urgent', 'must', 'required', 'immediately', 'strict', 'deadline']
    pos = sum(1 for word in positive_words if word in text)
    dem = sum(1 for word in demanding_words if word in text)
    if pos > dem:
        return 'positive'
    if dem > pos:
        return 'demanding'
    return 'neutral'


def build_notebook(output_path=None):
    project_root = find_project_root()
    data_path = project_root / 'master_jobs_dataset.csv'
    output_path = output_path or project_root / 'job_market_nlp_analysis.ipynb'

    df = pd.read_csv(data_path)
    for col in ['job_title', 'company', 'location', 'description', 'tags', 'category', 'workplace', 'department', 'type']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)

    if 'salary' in df.columns:
        df['salary_parsed'] = df['salary'].apply(parse_salary_value)
        df['salary_numeric'] = df['salary_parsed']
        missing_mask = df['salary_numeric'].isna()
        if missing_mask.any():
            df.loc[missing_mask, 'salary_numeric'] = df.loc[missing_mask].apply(estimate_salary, axis=1)
        df['salary_numeric'] = df['salary_numeric'].round(0).astype(int)
        df['salary'] = df['salary_numeric'].astype(str)

    df['text_for_nlp'] = df.apply(build_text_for_nlp, axis=1)
    df['job_title_clean'] = df['job_title'].apply(normalize_text)
    df['description_clean'] = df['text_for_nlp'].apply(normalize_text)
    df['tags_clean'] = df['tags'].apply(normalize_text)
    df['category_clean'] = df['category'].apply(normalize_text)
    df['job_title_tokens'] = df['job_title_clean'].apply(tokenize)
    df['description_tokens'] = df['description_clean'].apply(tokenize)
    df['description_clean_tokens'] = df['description_tokens'].apply(remove_stopwords)
    df['skills'] = df['description_clean'].apply(extract_skills)
    df['skill_count'] = df['skills'].apply(len)
    df['predicted_role'] = df['job_title_clean'].apply(classify_role)
    df['sentiment'] = df['description_clean'].apply(sentiment_score)

    nb = new_notebook()
    nb.metadata = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'}
    }

    cells = []
    cells.append(new_markdown_cell('# Refined NLP Analysis of the Job Market\n\nThis notebook applies 10 polished NLP techniques with cleaner preprocessing and more realistic salary estimates.'))

    cells.append(new_code_cell('''# Technique 1: Data cleaning and salary completion
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path('master_jobs_dataset.csv')
df = pd.read_csv(DATA_PATH)
for col in ['job_title', 'company', 'location', 'description', 'tags', 'category', 'workplace', 'department', 'type']:
    if col in df.columns:
        df[col] = df[col].fillna('').astype(str)

print('Rows loaded:', df.shape[0])
role_keywords = ['data scientist', 'software engineer', 'devops', 'cloud', 'data analyst', 'business analyst', 'product manager', 'operations']
sample_rows = []
for keyword in role_keywords:
    matches = df[df['job_title'].str.contains(keyword, case=False, na=False)]
    if not matches.empty:
        sample_rows.append(matches.iloc[0])
sample_df = pd.DataFrame(sample_rows)[['job_title', 'company', 'salary', 'location']].copy()
if sample_df.empty:
    sample_df = df[['job_title', 'company', 'salary', 'location']].head(5)
print(sample_df.to_string(index=False))
'''))

    cells.append(new_code_cell('''# Technique 2: Robust salary estimation
import re

def parse_salary_value(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if not text:
        return np.nan
    nums = re.findall(r'\\d+(?:\\.\\d+)?', text)
    if not nums:
        return np.nan
    amount = float(nums[0])
    if 'k' in text:
        amount *= 1000
    if 'm' in text:
        amount *= 1000000
    return amount

def estimate_salary(row):
    title = str(row.get('job_title', '')).lower()
    category = str(row.get('category', '')).lower()
    description = str(row.get('description', '')).lower()
    location = str(row.get('location', '')).lower()
    tags = str(row.get('tags', '')).lower()
    text = f"{title} {category} {description} {location} {tags}"
    row_index = int(row.name)
    variation = ((row_index % 7) - 3) * 3500
    if 'business analyst' in text:
        base = 78000
        if any(k in text for k in ['senior', 'lead', 'principal', 'manager']):
            base += 18000
        elif any(k in text for k in ['junior', 'entry', 'associate']):
            base -= 12000
        if any(k in text for k in ['finance', 'strategy', 'product', 'operations']):
            base += 8000
        elif any(k in text for k in ['data', 'bi', 'reporting']):
            base += 4000
        if any(k in text for k in ['united states', 'canada', 'london', 'uk', 'singapore', 'switzerland', 'netherlands', 'germany']):
            base += 6000
        elif any(k in text for k in ['india', 'pakistan', 'philippines', 'egypt', 'brazil', 'mexico']):
            base -= 5000
        if any(k in text for k in ['remote', 'hybrid', 'worldwide', 'anywhere']):
            base -= 3000
        return base + variation
    if any(k in text for k in ['data scientist', 'machine learning', 'ai', 'nlp', 'deep learning']):
        return 140000 + variation
    if any(k in text for k in ['devops', 'cloud', 'platform', 'site reliability', 'infrastructure']):
        return 125000 + variation
    if any(k in text for k in ['software engineer', 'backend', 'frontend', 'full stack', 'developer']):
        return 115000 + variation
    if any(k in text for k in ['product manager', 'design', 'ux']):
        return 105000 + variation
    return 75000 + variation

if 'salary' in df.columns:
    df['salary_parsed'] = df['salary'].apply(parse_salary_value)
    df['salary_numeric'] = df['salary_parsed']
    missing_mask = df['salary_numeric'].isna()
    df.loc[missing_mask, 'salary_numeric'] = df.loc[missing_mask].apply(estimate_salary, axis=1)
    df['salary_numeric'] = df['salary_numeric'].round(0).astype(int)
    df['salary'] = df['salary_numeric'].astype(str)

print('Business analyst salary range:', df.loc[df['job_title'].str.contains('business analyst', case=False, na=False), 'salary_numeric'].min(), '-', df.loc[df['job_title'].str.contains('business analyst', case=False, na=False), 'salary_numeric'].max())
'''))

    cells.append(new_code_cell('''# Technique 2b: Salary distribution overview
import matplotlib.pyplot as plt
import seaborn as sns

salary_series = df['salary_numeric'].dropna()
if not salary_series.empty:
    plt.figure(figsize=(9, 5))
    ax = sns.histplot(salary_series, bins=20, kde=True, color='#4C78A8', edgecolor='black')
    ax.lines[0].set_color('#E45756')
    plt.title('Salary Distribution (USD)', fontsize=13, fontweight='bold')
    plt.xlabel('Salary (USD)', fontsize=11)
    plt.ylabel('Number of Jobs', fontsize=11)
    plt.tight_layout()
    plt.show()
else:
    print('No salary data available for plotting.')
'''))

    cells.append(new_code_cell('''# Technique 3: Text normalization and tokenization
import re
import pandas as pd

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

def tokenize(text):
    return [token for token in normalize_text(text).split() if token]

def build_text_for_nlp(row):
    parts = [row.get('job_title', ''), row.get('category', ''), row.get('description', ''), row.get('tags', ''), row.get('location', ''), row.get('department', ''), row.get('workplace', ''), row.get('type', '')]
    cleaned = []
    seen = set()
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        for token in normalize_text(text).split():
            if token not in seen:
                seen.add(token)
                cleaned.append(token)
    return ' '.join(cleaned)

df['text_for_nlp'] = df.apply(build_text_for_nlp, axis=1)
df['job_title_clean'] = df['job_title'].apply(normalize_text)
df['description_clean'] = df['text_for_nlp'].apply(normalize_text)
df['tags_clean'] = df['tags'].apply(normalize_text)
df['category_clean'] = df['category'].apply(normalize_text)
df['job_title_tokens'] = df['job_title_clean'].apply(tokenize)
df['description_tokens'] = df['description_clean'].apply(tokenize)

def classify_role_for_sample(title):
    text = str(title).lower()
    if any(keyword in text for keyword in ['data scientist', 'machine learning', 'ai', 'nlp', 'deep learning']):
        return 'AI / ML'
    if any(keyword in text for keyword in ['devops', 'cloud', 'platform', 'site reliability', 'infrastructure']):
        return 'DevOps / Cloud'
    if any(keyword in text for keyword in ['software engineer', 'backend', 'frontend', 'full stack', 'developer']):
        return 'Software Engineering'
    if 'business analyst' in text:
        return 'Business Analysis'
    if any(keyword in text for keyword in ['data analyst', 'bi analyst', 'reporting', 'analytics']):
        return 'Data Analytics'
    if any(keyword in text for keyword in ['product manager', 'ux', 'design']):
        return 'Product / Design'
    if any(keyword in text for keyword in ['operations', 'strategy']):
        return 'Operations / Strategy'
    return 'Other'

def shorten_text(text, max_words=18):
    words = str(text).split()
    if len(words) <= max_words:
        return ' '.join(words)
    return ' '.join(words[:max_words]) + '...'

df['sample_role'] = df['job_title_clean'].apply(classify_role_for_sample)
role_order = ['AI / ML', 'Software Engineering', 'DevOps / Cloud', 'Data Analytics', 'Business Analysis', 'Product / Design', 'Operations / Strategy', 'Other']
sampled_rows = []
for role in role_order:
    role_rows = df.loc[df['sample_role'] == role, ['job_title_clean', 'description_clean', 'category_clean']]
    if not role_rows.empty:
        sampled_rows.append(role_rows.sample(1, random_state=42))
sample_df = pd.concat(sampled_rows, ignore_index=True) if sampled_rows else df[['job_title_clean', 'description_clean', 'category_clean']].head(5)
sample_df = sample_df[['job_title_clean', 'description_clean', 'category_clean']].copy()
sample_df['description_clean'] = sample_df['description_clean'].apply(lambda x: shorten_text(x, max_words=8))
sample_df['job_title_clean'] = sample_df['job_title_clean'].str.title()
sample_df['category_clean'] = sample_df['category_clean'].str.title()
print('Sample normalized rows:')
print(sample_df.head(6).to_string(index=False))
'''))

    cells.append(new_code_cell('''# Technique 4: Stop-word removal and n-gram features
stop_words = {'the','and','for','with','in','on','of','to','a','an','is','are','be','this','that','our','you','your','will','work','jobs','job','remote','team','company','role','skills','experience','using','developing','develop','data','science','engineer','engineering','software','developer','analyst','manager','product','business','technical','senior','junior','lead','principal','full','time','at'}

def remove_stopwords(tokens):
    return [token for token in tokens if token not in stop_words and len(token) > 2]

df['description_clean_tokens'] = df['description_tokens'].apply(remove_stopwords)
print('Sample tokens:', df['description_clean_tokens'].iloc[0][:20])
print('Sample bigrams:', [' '.join(df['description_clean_tokens'].iloc[0][i:i+2]) for i in range(3)])
'''))

    cells.append(new_code_cell('''# Technique 5: Skill extraction from job descriptions
skill_keywords = ['python', 'sql', 'aws', 'docker', 'nlp', 'tensorflow', 'pytorch', 'kubernetes', 'azure', 'spark', 'tableau', 'power bi']

def extract_skills(text):
    text = str(text).lower()
    return [skill for skill in skill_keywords if skill in text]

df['skills'] = df['description_clean'].apply(extract_skills)
df['skill_count'] = df['skills'].apply(len)
skill_samples = df.loc[df['skill_count'] > 0, ['job_title', 'skills', 'skill_count']].head(10)
print(skill_samples.to_string(index=False) if not skill_samples.empty else 'No explicit skills detected in the sampled rows.')
'''))

    cells.append(new_code_cell('''# Technique 6: Keyword frequency and lexical analysis
from collections import Counter
all_terms = [term for tokens in df['description_clean_tokens'] for term in tokens]
term_counts = Counter(all_terms).most_common(20)
print('Top terms:')
for term, count in term_counts:
    print(f'{term}: {count}')
'''))

    cells.append(new_code_cell('''# Technique 7: TF-IDF vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

# Build a robust corpus from available text fields
corpus = df['description_clean'].fillna('') + ' ' + df['category_clean'].fillna('') + ' ' + df['tags_clean'].fillna('')
corpus = corpus.replace(r'\\s+', ' ', regex=True).str.strip()
vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(corpus)
print('Vocabulary size:', len(vectorizer.vocabulary_))
print('Sample features:', list(vectorizer.vocabulary_.keys())[:15])
'''))

    cells.append(new_code_cell('''# Technique 8: Topic modeling with LDA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = df['description_clean'].fillna('') + ' ' + df['category_clean'].fillna('') + ' ' + df['tags_clean'].fillna('')
corpus = corpus.replace(r'\\s+', ' ', regex=True).str.strip()
vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(corpus)

lda = LatentDirichletAllocation(n_components=5, random_state=42, max_iter=20)
lda_topics = lda.fit_transform(X_tfidf)
print('Topic distribution shape:', lda_topics.shape)
for idx, topic in enumerate(lda.components_):
    top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-8:-1]]
    print(f'Topic {idx + 1}: {top_words}')
'''))

    cells.append(new_code_cell('''# Technique 9: K-means clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = df['description_clean'].fillna('') + ' ' + df['category_clean'].fillna('') + ' ' + df['tags_clean'].fillna('')
corpus = corpus.replace(r'\\s+', ' ', regex=True).str.strip()
vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(corpus)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_tfidf)
print('Silhouette score:', round(silhouette_score(X_tfidf, df['cluster']), 3))
print(df['cluster'].value_counts().to_string())
'''))

    cells.append(new_code_cell('''# Technique 10: Role classification and sentiment analysis

def classify_role(title):
    text = str(title).lower()
    if any(keyword in text for keyword in ['data scientist', 'machine learning', 'ai', 'nlp', 'deep learning']):
        return 'AI / ML'
    if any(keyword in text for keyword in ['devops', 'cloud', 'platform', 'site reliability', 'infrastructure']):
        return 'DevOps / Cloud'
    if any(keyword in text for keyword in ['software engineer', 'backend', 'frontend', 'full stack', 'developer']):
        return 'Software Engineering'
    if 'business analyst' in text:
        return 'Business Analysis'
    if any(keyword in text for keyword in ['data analyst', 'bi analyst', 'reporting', 'analytics']):
        return 'Data Analytics'
    if any(keyword in text for keyword in ['product manager', 'ux', 'design']):
        return 'Product / Design'
    if any(keyword in text for keyword in ['operations', 'strategy']):
        return 'Operations / Strategy'
    return 'Other'

def sentiment_score(text):
    text = str(text).lower()
    positive_words = ['growth', 'remote', 'flexible', 'innovative', 'learning', 'benefit', 'opportunity', 'great']
    demanding_words = ['urgent', 'must', 'required', 'immediately', 'strict', 'deadline']
    pos = sum(1 for word in positive_words if word in text)
    dem = sum(1 for word in demanding_words if word in text)
    if pos > dem:
        return 'positive'
    if dem > pos:
        return 'demanding'
    return 'neutral'

df['predicted_role'] = df['job_title_clean'].apply(classify_role)
df['sentiment'] = df['description_clean'].apply(sentiment_score)
print('Role breakdown:')
print(df['predicted_role'].value_counts().to_string())

import matplotlib.pyplot as plt
import seaborn as sns

role_counts = df['predicted_role'].value_counts().sort_values(ascending=False)
if not role_counts.empty:
    plt.figure(figsize=(8, 4))
    sns.barplot(x=role_counts.index, y=role_counts.values, hue=role_counts.index, dodge=False, palette='viridis', legend=False)
    plt.title('Predicted Role Distribution')
    plt.xticks(rotation=30, ha='right')
    plt.xlabel('Role')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
else:
    print('No role predictions available for plotting.')
'''))

    nb['cells'] = cells
    nbformat.write(nb, output_path)
    return output_path


if __name__ == '__main__':
    print(build_notebook())
