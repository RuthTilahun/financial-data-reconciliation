"""
Anomaly Detection Module
Detects unusual patterns and outliers in financial data
Critical for catching data errors before they reach investors
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialAnomalyDetector:
    """
    Detect unusual patterns and outliers in financial data
    Critical for catching data errors before they reach investors
    """

    def __init__(self):
        self.anomalies_found = []

    def detect_statistical_outliers(self, df: pd.DataFrame,
                                   field: str,
                                   method: str = 'zscore',
                                   threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers using statistical methods
        """
        logger.info(f"\n🔍 Detecting outliers in {field} using {method} method...")

        if field not in df.columns:
            logger.warning(f"   Field {field} not found in dataframe")
            return pd.DataFrame()

        outliers = pd.DataFrame()

        # Filter out null values
        df_valid = df[df[field].notna()].copy()

        if len(df_valid) == 0:
            logger.warning(f"   No valid data for {field}")
            return pd.DataFrame()

        if method == 'zscore':
            # Z-score method: (x - mean) / std
            mean = df_valid[field].mean()
            std = df_valid[field].std()

            if std == 0:
                logger.warning(f"   Standard deviation is 0, cannot compute Z-scores")
                return pd.DataFrame()

            z_scores = np.abs((df_valid[field] - mean) / std)

            outliers = df_valid[z_scores > threshold].copy()
            outliers['z_score'] = z_scores[z_scores > threshold]
            outliers['method'] = 'zscore'

        elif method == 'iqr':
            # Interquartile Range method
            Q1 = df_valid[field].quantile(0.25)
            Q3 = df_valid[field].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = df_valid[(df_valid[field] < lower_bound) | (df_valid[field] > upper_bound)].copy()
            outliers['lower_bound'] = lower_bound
            outliers['upper_bound'] = upper_bound
            outliers['method'] = 'iqr'

        elif method == 'modified_zscore':
            # Modified Z-score (more robust to outliers)
            median = df_valid[field].median()
            mad = np.median(np.abs(df_valid[field] - median))

            if mad == 0:
                logger.warning(f"   MAD is 0, cannot compute modified Z-scores")
                return pd.DataFrame()

            modified_z = 0.6745 * (df_valid[field] - median) / mad

            outliers = df_valid[np.abs(modified_z) > threshold].copy()
            outliers['modified_z_score'] = modified_z[np.abs(modified_z) > threshold]
            outliers['method'] = 'modified_zscore'

        if len(outliers) > 0:
            logger.info(f"   🚨 Found {len(outliers)} outliers")

            # Log anomalies
            for idx, row in outliers.iterrows():
                severity = 'high'
                if method == 'zscore' and 'z_score' in row and row['z_score'] > 5:
                    severity = 'critical'

                self.anomalies_found.append({
                    'anomaly_type': 'statistical_outlier',
                    'field': field,
                    'ticker': row.get('ticker'),
                    'company_name': row.get('company_name'),
                    'value': row[field],
                    'method': method,
                    'severity': severity,
                    'detected_at': pd.Timestamp.now()
                })
        else:
            logger.info(f"   ✅ No outliers detected")

        return outliers

    def detect_business_rule_violations(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect violations of business logic rules
        """
        logger.info(f"\n🔍 Checking business rule violations...")

        violations = []

        # Rule 1: Revenue should be positive for most companies
        if 'revenue' in df.columns:
            negative_revenue = df[df['revenue'] < 0]
            if len(negative_revenue) > 0:
                logger.info(f"   🚨 Rule Violation: {len(negative_revenue)} companies with negative revenue")
                for idx, row in negative_revenue.iterrows():
                    violations.append({
                        'rule': 'revenue_must_be_positive',
                        'ticker': row.get('ticker'),
                        'company_name': row.get('company_name'),
                        'value': row['revenue'],
                        'severity': 'high',
                        'description': f"Revenue of ${row['revenue']:,.0f} is negative"
                    })

        # Rule 2: Net income should be less than revenue
        if 'revenue' in df.columns and 'net_income' in df.columns:
            profit_exceeds_revenue = df[(df['net_income'] > df['revenue'] * 1.1) & df['revenue'].notna() & df['net_income'].notna()]
            if len(profit_exceeds_revenue) > 0:
                logger.info(f"   🚨 Rule Violation: {len(profit_exceeds_revenue)} companies with net income > revenue")
                for idx, row in profit_exceeds_revenue.iterrows():
                    violations.append({
                        'rule': 'net_income_exceeds_revenue',
                        'ticker': row.get('ticker'),
                        'company_name': row.get('company_name'),
                        'revenue': row['revenue'],
                        'net_income': row['net_income'],
                        'severity': 'medium',
                        'description': f"Net income (${row['net_income']:,.0f}) > Revenue (${row['revenue']:,.0f})",
                        'note': 'May be legitimate for holding companies with investment gains'
                    })

        # Rule 3: Check for suspiciously round numbers (possible estimates)
        if 'revenue' in df.columns:
            # Revenue ending in many zeros might be estimated
            suspiciously_round = df[df['revenue'].notna() & (df['revenue'] % 1_000_000_000 == 0) & (df['revenue'] > 0)]
            if len(suspiciously_round) > 0:
                logger.info(f"   ⚠️  Warning: {len(suspiciously_round)} companies with very round revenue numbers")
                for idx, row in suspiciously_round.iterrows():
                    violations.append({
                        'rule': 'suspiciously_round_number',
                        'ticker': row.get('ticker'),
                        'value': row['revenue'],
                        'severity': 'low',
                        'description': f"Revenue of exactly ${row['revenue']:,.0f} (suspiciously round - may be estimate)"
                    })

        logger.info(f"   Total violations: {len(violations)}")
        return violations

    def detect_cross_metric_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect anomalies by comparing related metrics
        e.g., revenue up 50% but expenses flat (suspicious)
        """
        logger.info(f"\n🔍 Checking cross-metric consistency...")

        anomalies = []

        # Check profit margin (net income / revenue)
        if 'revenue' in df.columns and 'net_income' in df.columns:
            df_copy = df.copy()
            df_copy['profit_margin'] = (df_copy['net_income'] / df_copy['revenue']) * 100

            # Flag unusually high margins (>50%) or very negative (<-50%)
            high_margin = df_copy[df_copy['profit_margin'] > 50]
            low_margin = df_copy[df_copy['profit_margin'] < -50]

            for idx, row in high_margin.iterrows():
                anomalies.append({
                    'anomaly_type': 'unusual_profit_margin',
                    'ticker': row.get('ticker'),
                    'profit_margin': row['profit_margin'],
                    'severity': 'medium',
                    'description': f"Profit margin of {row['profit_margin']:.1f}% is unusually high"
                })

            for idx, row in low_margin.iterrows():
                anomalies.append({
                    'anomaly_type': 'unusual_profit_margin',
                    'ticker': row.get('ticker'),
                    'profit_margin': row['profit_margin'],
                    'severity': 'medium',
                    'description': f"Profit margin of {row['profit_margin']:.1f}% indicates major losses"
                })

        logger.info(f"   Found {len(anomalies)} cross-metric anomalies")
        return anomalies

    def generate_anomaly_report(self) -> pd.DataFrame:
        """
        Compile all detected anomalies into a report
        """
        if not self.anomalies_found:
            logger.info("\n✅ No anomalies detected!")
            return pd.DataFrame()

        anomalies_df = pd.DataFrame(self.anomalies_found)

        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        anomalies_df['severity_rank'] = anomalies_df['severity'].map(severity_order)
        anomalies_df = anomalies_df.sort_values('severity_rank')

        # Add unique ID
        anomalies_df['anomaly_id'] = [f"ANOM_{i:04d}" for i in range(len(anomalies_df))]

        # Save report
        anomalies_df.to_csv('data/reconciliation/anomaly_report.csv', index=False)

        logger.info(f"\n📊 ANOMALY DETECTION SUMMARY")
        logger.info(f"="*60)
        logger.info(f"Total anomalies detected: {len(anomalies_df)}")
        logger.info(f"\nBy severity:")
        severity_counts = anomalies_df['severity'].value_counts()
        for severity, count in severity_counts.items():
            logger.info(f"   {severity}: {count}")

        if 'anomaly_type' in anomalies_df.columns:
            logger.info(f"\nBy type:")
            type_counts = anomalies_df['anomaly_type'].value_counts()
            for atype, count in type_counts.items():
                logger.info(f"   {atype}: {count}")

        return anomalies_df


if __name__ == '__main__':
    # Test anomaly detection
    test_data = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'OUTLIER'],
        'company_name': ['Apple', 'Microsoft', 'Google', 'Amazon', 'Test Co'],
        'revenue': [385000000000, 212000000000, 307000000000, 575000000000, 9999999999999],  # Last one is outlier
        'net_income': [99000000000, 72000000000, 73000000000, 30000000000, 10000000000]
    })

    detector = FinancialAnomalyDetector()

    # Test statistical outliers
    outliers = detector.detect_statistical_outliers(test_data, 'revenue', method='zscore', threshold=2.0)
    print("\n📊 Statistical Outliers:")
    if len(outliers) > 0:
        print(outliers[['ticker', 'revenue', 'z_score']])

    # Test business rule violations
    violations = detector.detect_business_rule_violations(test_data)
    print(f"\n📋 Business Rule Violations: {len(violations)}")

    # Test cross-metric anomalies
    cross_anomalies = detector.detect_cross_metric_anomalies(test_data)
    print(f"\n🔗 Cross-Metric Anomalies: {len(cross_anomalies)}")

    # Generate report
    report = detector.generate_anomaly_report()
    print("\n✅ Anomaly report generated!")
