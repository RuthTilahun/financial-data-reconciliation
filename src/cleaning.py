"""
Data Cleaning Module
Intelligent standardization and cleaning of financial data
Handles unit normalization, date standardization, and format correction
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialDataCleaner:
    """
    Clean and standardize financial data
    """

    def __init__(self):
        pass

    def detect_unit(self, value: float, field_name: str) -> str:
        """
        Detect if value is in thousands, millions, or billions
        Based on expected ranges for field type
        """
        if pd.isna(value):
            return 'unknown'

        # Expected ranges for major public companies (in dollars)
        expected_ranges = {
            'revenue': (1_000_000_000, 1_000_000_000_000),  # $1B to $1T
            'net_income': (100_000_000, 200_000_000_000),    # $100M to $200B
            'total_assets': (1_000_000_000, 5_000_000_000_000), # $1B to $5T
            'market_cap': (1_000_000_000, 5_000_000_000_000),
            'ebitda': (100_000_000, 300_000_000_000)
        }

        min_expected, max_expected = expected_ranges.get(field_name, (0, np.inf))

        # Check what unit makes sense
        if min_expected <= value <= max_expected:
            return 'dollars'  # Already in dollars
        elif min_expected <= value * 1000 <= max_expected:
            return 'thousands'
        elif min_expected <= value * 1_000_000 <= max_expected:
            return 'millions'
        elif min_expected <= value * 1_000_000_000 <= max_expected:
            return 'billions'
        else:
            return 'unknown'

    def normalize_to_dollars(self, df: pd.DataFrame, numeric_fields: List[str]) -> pd.DataFrame:
        """
        Convert all financial values to standard unit (dollars)
        """
        df_clean = df.copy()

        for field in numeric_fields:
            if field not in df.columns:
                continue

            logger.info(f"🔧 Normalizing {field}...")

            # Detect unit for each value
            df_clean[f'{field}_detected_unit'] = df_clean[field].apply(
                lambda x: self.detect_unit(x, field)
            )

            # Apply conversion
            def convert_to_dollars(row):
                value = row[field]
                unit = row.get(f'{field}_detected_unit', 'unknown')

                if pd.isna(value):
                    return value

                if unit == 'thousands':
                    return value * 1000
                elif unit == 'millions':
                    return value * 1_000_000
                elif unit == 'billions':
                    return value * 1_000_000_000
                elif unit == 'dollars':
                    return value
                else:
                    # Unknown unit - flag for review
                    return np.nan

            df_clean[field] = df_clean.apply(convert_to_dollars, axis=1)

            # Report conversions
            unit_counts = df_clean[f'{field}_detected_unit'].value_counts()
            logger.info(f"   Conversions applied:")
            for unit, count in unit_counts.items():
                logger.info(f"      {unit}: {count} values")

        return df_clean

    def standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all dates to ISO 8601 format (YYYY-MM-DD)
        """
        df_clean = df.copy()

        date_columns = [col for col in df.columns if 'date' in col.lower()]

        for col in date_columns:
            if col not in df_clean.columns:
                continue

            logger.info(f"📅 Standardizing {col}...")

            # Parse dates (pandas is smart about formats)
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

            # Convert to ISO format string
            df_clean[col] = df_clean[col].dt.strftime('%Y-%m-%d')

            # Report any failed conversions
            failed = df_clean[col].isna().sum() - df[col].isna().sum()
            if failed > 0:
                logger.info(f"   ⚠️  {failed} dates could not be parsed")

        return df_clean

    def handle_missing_data(self, df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
        """
        Handle missing data based on field-specific strategies

        Strategies:
        - 'drop': Remove rows with missing data
        - 'zero': Fill with 0
        - 'median': Fill with median
        - 'flag': Keep as missing but add flag column
        - 'forward_fill': Use previous value
        """
        df_clean = df.copy()

        for field, action in strategy.items():
            if field not in df.columns:
                continue

            missing_count = df_clean[field].isna().sum()
            if missing_count == 0:
                continue

            logger.info(f"🔧 Handling missing {field} ({missing_count} records)...")

            if action == 'drop':
                df_clean = df_clean.dropna(subset=[field])
                logger.info(f"   Dropped {missing_count} rows")

            elif action == 'zero':
                df_clean[field].fillna(0, inplace=True)
                logger.info(f"   Filled with 0")

            elif action == 'median':
                median_val = df_clean[field].median()
                df_clean[field].fillna(median_val, inplace=True)
                logger.info(f"   Filled with median: ${median_val:,.0f}")

            elif action == 'flag':
                df_clean[f'{field}_missing'] = df_clean[field].isna()
                logger.info(f"   Flagged missing values (kept as NaN)")

            elif action == 'forward_fill':
                df_clean[field].fillna(method='ffill', inplace=True)
                logger.info(f"   Forward filled missing values")

        return df_clean

    def remove_outliers(self, df: pd.DataFrame, field: str, method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove or flag statistical outliers
        """
        df_clean = df.copy()

        if field not in df.columns:
            return df_clean

        logger.info(f"🔍 Detecting outliers in {field}...")

        if method == 'iqr':
            Q1 = df_clean[field].quantile(0.25)
            Q3 = df_clean[field].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = df_clean[(df_clean[field] < lower_bound) | (df_clean[field] > upper_bound)]

            logger.info(f"   Found {len(outliers)} outliers")
            logger.info(f"   Bounds: [{lower_bound:,.0f}, {upper_bound:,.0f}]")

            # Flag rather than remove (safer)
            df_clean[f'{field}_is_outlier'] = (
                (df_clean[field] < lower_bound) | (df_clean[field] > upper_bound)
            )

        elif method == 'zscore':
            z_scores = np.abs((df_clean[field] - df_clean[field].mean()) / df_clean[field].std())
            outliers = df_clean[z_scores > threshold]

            logger.info(f"   Found {len(outliers)} outliers (Z-score > {threshold})")

            df_clean[f'{field}_is_outlier'] = z_scores > threshold

        return df_clean

    def normalize_company_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize company names (remove extra spaces, standardize punctuation)
        """
        df_clean = df.copy()

        if 'company_name' not in df.columns:
            return df_clean

        logger.info("🏢 Normalizing company names...")

        # Remove leading/trailing whitespace
        df_clean['company_name'] = df_clean['company_name'].str.strip()

        # Remove extra internal whitespace
        df_clean['company_name'] = df_clean['company_name'].str.replace(r'\s+', ' ', regex=True)

        # Standardize common abbreviations
        replacements = {
            r'\bInc\.?\b': 'Inc.',
            r'\bCorp\.?\b': 'Corporation',
            r'\bCo\.?\b': 'Company',
            r'\bLtd\.?\b': 'Limited',
        }

        for pattern, replacement in replacements.items():
            df_clean['company_name'] = df_clean['company_name'].str.replace(
                pattern, replacement, regex=True
            )

        logger.info("   ✅ Company names normalized")

        return df_clean

    def full_cleaning_pipeline(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """
        Run complete cleaning workflow
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"CLEANING PIPELINE: {source_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Input: {len(df)} records")

        df_clean = df.copy()

        # Step 1: Normalize company names
        if 'company_name' in df_clean.columns:
            df_clean = self.normalize_company_names(df_clean)

        # Step 2: Standardize dates
        df_clean = self.standardize_dates(df_clean)

        # Step 3: Normalize financial units
        numeric_fields = ['revenue', 'net_income', 'total_assets', 'market_cap', 'ebitda']
        existing_numeric = [f for f in numeric_fields if f in df_clean.columns]
        if existing_numeric:
            df_clean = self.normalize_to_dollars(df_clean, existing_numeric)

        # Step 4: Handle missing data
        missing_strategy = {
            'revenue': 'flag',  # Critical field - flag but don't drop
            'net_income': 'flag',
            'company_name': 'drop',  # Must have name
            'ticker': 'drop'  # Must have ticker
        }
        # Only apply strategy for fields that exist
        existing_strategy = {k: v for k, v in missing_strategy.items() if k in df_clean.columns}
        if existing_strategy:
            df_clean = self.handle_missing_data(df_clean, existing_strategy)

        # Step 5: Detect outliers (flag, don't remove)
        for field in ['revenue', 'net_income']:
            if field in df_clean.columns:
                df_clean = self.remove_outliers(df_clean, field, method='iqr')

        # Step 6: Remove extra whitespace from text columns
        text_columns = df_clean.select_dtypes(include=['object']).columns
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].str.strip() if df_clean[col].dtype == 'object' else df_clean[col]

        logger.info(f"\n✅ Cleaning complete!")
        logger.info(f"Output: {len(df_clean)} records")

        return df_clean


if __name__ == '__main__':
    # Test cleaning
    test_data = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL'],
        'company_name': ['Apple Inc.  ', '  Microsoft Corp', 'Google Inc'],
        'revenue': [385.7, 211915, 307.4],  # Mixed units: billions and millions
        'net_income': [99.8, 72.4, 73.8],
        'report_date': ['2024-10-15', '10/15/2024', '15-10-2024'],
        'filing_date': ['2024-10-20', '2024-10-21', '2024-10-22']
    })

    print("📊 Original Data:")
    print(test_data)

    cleaner = FinancialDataCleaner()
    clean_data = cleaner.full_cleaning_pipeline(test_data, 'TEST_DATA')

    print("\n✨ Cleaned Data:")
    print(clean_data[['ticker', 'company_name', 'revenue', 'report_date']])
