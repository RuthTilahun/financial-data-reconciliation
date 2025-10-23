"""
Data Validation Module
Comprehensive validation for financial data quality
Implements 20+ validation rules covering completeness, accuracy, consistency, and timeliness
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialDataValidator:
    """
    Comprehensive validation for financial data
    Mirrors what Arch does for investment data quality
    """

    def __init__(self):
        self.issues = []

        # Define valid ranges (based on real-world data for large public companies)
        self.valid_ranges = {
            'revenue': (0, 1_000_000_000_000),  # $0 to $1T
            'net_income': (-50_000_000_000, 200_000_000_000),  # Allow losses
            'total_assets': (0, 5_000_000_000_000),  # $0 to $5T
            'market_cap': (0, 5_000_000_000_000),  # $0 to $5T
            'pe_ratio': (-100, 300),  # Negative for loss-making companies
            'ebitda': (-10_000_000_000, 300_000_000_000)
        }

        # Canonical company names (golden source)
        self.canonical_names = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com, Inc.',
            'TSLA': 'Tesla, Inc.',
            'JPM': 'JPMorgan Chase & Co.',
            'JNJ': 'Johnson & Johnson',
            'WMT': 'Walmart Inc.',
            'DIS': 'The Walt Disney Company',
            'XOM': 'Exxon Mobil Corporation'
        }

    def validate_completeness(self, df: pd.DataFrame, required_fields: List[str]) -> List[Dict]:
        """Check for missing critical data"""
        issues = []

        for field in required_fields:
            if field not in df.columns:
                issues.append({
                    'rule': 'completeness',
                    'severity': 'critical',
                    'field': field,
                    'description': f'Required field "{field}" is missing from dataset',
                    'affected_records': len(df)
                })
            else:
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    null_pct = (null_count / len(df)) * 100
                    issues.append({
                        'rule': 'completeness',
                        'severity': 'high' if null_pct > 10 else 'medium',
                        'field': field,
                        'description': f'{null_count} records ({null_pct:.1f}%) missing {field}',
                        'affected_records': null_count,
                        'example_ids': df[df[field].isnull()].index.tolist()[:5]
                    })

        return issues

    def validate_consistency(self, df: pd.DataFrame) -> List[Dict]:
        """Check cross-field consistency"""
        issues = []

        # Rule 1: Revenue should be >= 0 for most cases
        if 'revenue' in df.columns:
            negative_revenue = df[df['revenue'] < 0]
            if len(negative_revenue) > 0:
                issues.append({
                    'rule': 'consistency',
                    'severity': 'high',
                    'description': f'{len(negative_revenue)} records with negative revenue',
                    'affected_records': negative_revenue.index.tolist(),
                    'details': negative_revenue[['ticker', 'company_name', 'revenue']].to_dict('records') if 'ticker' in df.columns else []
                })

        # Rule 2: Filing date should not be in the future
        if 'filing_date' in df.columns:
            df_copy = df.copy()
            df_copy['filing_date_parsed'] = pd.to_datetime(df_copy['filing_date'], errors='coerce')
            future_dates = df_copy[df_copy['filing_date_parsed'] > datetime.now()]
            if len(future_dates) > 0:
                issues.append({
                    'rule': 'consistency',
                    'severity': 'critical',
                    'description': f'{len(future_dates)} records with future filing dates',
                    'affected_records': future_dates.index.tolist()
                })

        # Rule 3: Period end should be before filing date
        if 'period_end' in df.columns and 'filing_date' in df.columns:
            df_copy = df.copy()
            df_copy['period_end_parsed'] = pd.to_datetime(df_copy['period_end'], errors='coerce')
            df_copy['filing_date_parsed'] = pd.to_datetime(df_copy['filing_date'], errors='coerce')
            invalid_sequence = df_copy[df_copy['period_end_parsed'] > df_copy['filing_date_parsed']]
            if len(invalid_sequence) > 0:
                issues.append({
                    'rule': 'consistency',
                    'severity': 'high',
                    'description': 'Period end date after filing date (impossible)',
                    'affected_records': len(invalid_sequence)
                })

        # Rule 4: Net income should be <= Revenue (generally)
        if 'revenue' in df.columns and 'net_income' in df.columns:
            # Allow 10% margin for data quirks
            impossible_profit = df[df['net_income'] > df['revenue'] * 1.1]
            if len(impossible_profit) > 0:
                issues.append({
                    'rule': 'consistency',
                    'severity': 'medium',
                    'description': 'Net income exceeds revenue (unusual)',
                    'affected_records': len(impossible_profit),
                    'note': 'May be legitimate for certain business models (e.g., investment gains)'
                })

        return issues

    def validate_accuracy(self, df: pd.DataFrame) -> List[Dict]:
        """Validate against known standards and ranges"""
        issues = []

        # Check numeric fields against valid ranges
        for field, (min_val, max_val) in self.valid_ranges.items():
            if field in df.columns:
                outliers = df[(df[field] < min_val) | (df[field] > max_val)]
                if len(outliers) > 0:
                    issues.append({
                        'rule': 'accuracy',
                        'severity': 'medium',
                        'field': field,
                        'description': f'{len(outliers)} values outside valid range [{min_val}, {max_val}]',
                        'affected_records': outliers.index.tolist(),
                        'examples': outliers[['ticker', field]].head(3).to_dict('records') if 'ticker' in df.columns else []
                    })

        # Validate ticker format (should be 1-5 uppercase letters)
        if 'ticker' in df.columns:
            invalid_tickers = df[df['ticker'].notna() & ~df['ticker'].str.match(r'^[A-Z]{1,5}$', na=False)]
            if len(invalid_tickers) > 0:
                issues.append({
                    'rule': 'accuracy',
                    'severity': 'high',
                    'description': f'{len(invalid_tickers)} invalid ticker symbols',
                    'affected_records': invalid_tickers.index.tolist()
                })

        # Check for suspicious zeros (probably missing data coded as 0)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['revenue', 'market_cap', 'total_assets']:
                zero_count = (df[col] == 0).sum()
                if zero_count > len(df) * 0.05:  # More than 5% zeros is suspicious
                    issues.append({
                        'rule': 'accuracy',
                        'severity': 'medium',
                        'field': col,
                        'description': f'{zero_count} suspicious zero values (likely missing data)',
                        'affected_records': zero_count
                    })

        return issues

    def validate_uniqueness(self, df: pd.DataFrame, key_fields: List[str]) -> List[Dict]:
        """Detect duplicate records"""
        issues = []

        # Only validate if all key fields exist
        existing_key_fields = [f for f in key_fields if f in df.columns]
        if not existing_key_fields:
            return issues

        duplicates = df[df.duplicated(subset=existing_key_fields, keep=False)]
        if len(duplicates) > 0:
            # Group duplicates to show which records are dupes of each other
            dup_groups = duplicates.groupby(existing_key_fields).size().reset_index(name='count')
            issues.append({
                'rule': 'uniqueness',
                'severity': 'high',
                'description': f'{len(duplicates)} duplicate records based on {existing_key_fields}',
                'affected_records': duplicates.index.tolist(),
                'duplicate_groups': len(dup_groups),
                'examples': dup_groups.head(5).to_dict('records')
            })

        return issues

    def validate_timeliness(self, df: pd.DataFrame, date_field: str, max_age_days: int = 365) -> List[Dict]:
        """Check if data is recent enough"""
        issues = []

        if date_field not in df.columns:
            return issues

        df_copy = df.copy()
        df_copy[f'{date_field}_parsed'] = pd.to_datetime(df_copy[date_field], errors='coerce')
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        stale_data = df_copy[df_copy[f'{date_field}_parsed'] < cutoff_date]

        if len(stale_data) > 0:
            stale_pct = (len(stale_data) / len(df)) * 100
            issues.append({
                'rule': 'timeliness',
                'severity': 'low' if stale_pct < 20 else 'medium',
                'description': f'{len(stale_data)} records ({stale_pct:.1f}%) older than {max_age_days} days',
                'affected_records': stale_data.index.tolist(),
                'oldest_date': str(stale_data[f'{date_field}_parsed'].min())
            })

        return issues

    def validate_format(self, df: pd.DataFrame) -> List[Dict]:
        """Check data format standards"""
        issues = []

        # Date format consistency
        date_fields = [col for col in df.columns if 'date' in col.lower()]
        for field in date_fields:
            df_copy = df.copy()
            # Try to parse dates
            df_copy[f'{field}_parsed'] = pd.to_datetime(df_copy[field], errors='coerce')
            failed_parse = df_copy[df_copy[field].notnull() & df_copy[f'{field}_parsed'].isnull()]
            if len(failed_parse) > 0:
                issues.append({
                    'rule': 'format',
                    'severity': 'medium',
                    'field': field,
                    'description': f'{len(failed_parse)} dates in non-standard format',
                    'affected_records': failed_parse.index.tolist(),
                    'examples': failed_parse[field].head(5).tolist()
                })

        # Company name format (check for extra whitespace, special characters)
        if 'company_name' in df.columns:
            # Extra whitespace
            extra_space = df[df['company_name'].notna() & df['company_name'].str.contains(r'\s{2,}', na=False, regex=True)]
            if len(extra_space) > 0:
                issues.append({
                    'rule': 'format',
                    'severity': 'low',
                    'description': f'{len(extra_space)} company names with extra whitespace',
                    'affected_records': extra_space.index.tolist()
                })

            # Trailing/leading whitespace
            whitespace = df[df['company_name'].notna() & (df['company_name'].str.strip() != df['company_name'])]
            if len(whitespace) > 0:
                issues.append({
                    'rule': 'format',
                    'severity': 'low',
                    'description': f'{len(whitespace)} company names with leading/trailing whitespace',
                    'affected_records': whitespace.index.tolist()
                })

        return issues

    def run_all_validations(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Run complete validation suite"""
        logger.info(f"\n🔍 Running validation suite on {dataset_name}...")

        all_issues = []

        # Required fields vary by source
        required_fields = ['ticker', 'company_name']
        if 'revenue' in df.columns:
            required_fields.append('revenue')

        # Run all validation types
        all_issues.extend(self.validate_completeness(df, required_fields))
        all_issues.extend(self.validate_consistency(df))
        all_issues.extend(self.validate_accuracy(df))
        all_issues.extend(self.validate_uniqueness(df, ['ticker', 'period_end']))

        # Timeliness check for date fields
        if 'filing_date' in df.columns:
            all_issues.extend(self.validate_timeliness(df, 'filing_date', max_age_days=365))
        elif 'last_updated' in df.columns:
            all_issues.extend(self.validate_timeliness(df, 'last_updated', max_age_days=365))

        all_issues.extend(self.validate_format(df))

        # Convert to DataFrame
        issues_df = pd.DataFrame(all_issues)
        if len(issues_df) > 0:
            issues_df['dataset'] = dataset_name
            issues_df['validation_timestamp'] = datetime.now()

        # Summary
        logger.info(f"   Found {len(issues_df)} issues:")
        if len(issues_df) > 0:
            severity_counts = issues_df['severity'].value_counts()
            for severity, count in severity_counts.items():
                logger.info(f"      {severity}: {count}")
        else:
            logger.info("      ✅ No issues found!")

        return issues_df

    def calculate_quality_score(self, df: pd.DataFrame, issues_df: pd.DataFrame) -> float:
        """
        Calculate overall data quality score (0-100)
        Based on completeness, accuracy, consistency
        """
        if len(df) == 0:
            return 0.0

        # Completeness score (% of non-null values in key fields)
        key_fields = ['ticker', 'company_name', 'revenue']
        existing_fields = [f for f in key_fields if f in df.columns]
        if existing_fields:
            completeness = df[existing_fields].notna().mean().mean() * 100
        else:
            completeness = 0

        # Accuracy score (based on validation issues)
        accuracy_issues = len(issues_df[issues_df['rule'] == 'accuracy']) if len(issues_df) > 0 else 0
        accuracy = max(0, 100 - (accuracy_issues / len(df) * 100))

        # Consistency score
        consistency_issues = len(issues_df[issues_df['rule'] == 'consistency']) if len(issues_df) > 0 else 0
        consistency = max(0, 100 - (consistency_issues / len(df) * 100))

        # Weighted average
        quality_score = (completeness * 0.4) + (accuracy * 0.3) + (consistency * 0.3)

        return round(quality_score, 2)


if __name__ == '__main__':
    # Test validation
    # Create sample data with issues
    test_data = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', None, 'GOOGL', 'INVALID'],
        'company_name': ['Apple Inc.', 'Microsoft  ', 'Google', 'Alphabet Inc.', 'Test Company'],
        'revenue': [385000000000, -1000, 307000000000, 0, 9999999999999],
        'net_income': [99000000000, 72000000000, None, 73000000000, 10000000000],
        'filing_date': ['2024-10-15', '2024-10-16', '2025-12-01', '2024-09-30', 'invalid-date']
    })

    validator = FinancialDataValidator()
    issues = validator.run_all_validations(test_data, 'TEST_DATA')

    print("\n📊 Validation Issues Found:")
    if len(issues) > 0:
        print(issues[['rule', 'severity', 'description']].to_string(index=False))
    else:
        print("No issues found!")

    quality_score = validator.calculate_quality_score(test_data, issues)
    print(f"\n✅ Data Quality Score: {quality_score}%")
