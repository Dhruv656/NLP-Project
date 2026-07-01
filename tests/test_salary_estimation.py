import pandas as pd
from build_notebook import estimate_salary


def test_business_analyst_salary_varies_by_context():
    rows = [
        pd.Series({'job_title': 'Business Analyst', 'category': 'Business Analyst', 'description': 'Senior finance analyst for strategy and planning', 'location': 'London, UK', 'tags': 'finance, strategy'}),
        pd.Series({'job_title': 'Business Analyst', 'category': 'Business Analyst', 'description': 'Junior data analyst supporting operations', 'location': 'Delhi, India', 'tags': 'operations, reporting'}),
        pd.Series({'job_title': 'Business Analyst', 'category': 'Business Analyst', 'description': 'Remote business analyst for product analytics', 'location': 'Remote', 'tags': 'product, analytics'}),
    ]
    salaries = [estimate_salary(row, index=i) for i, row in enumerate(rows)]
    assert len(set(salaries)) > 2, salaries
    assert min(salaries) < 80000, salaries
    assert max(salaries) > 90000, salaries
