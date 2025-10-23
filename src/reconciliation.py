"""
Reconciliation Module
Reconciles company entities across multiple data sources
Core skill for Arch: matching same company across different data portals
"""

import pandas as pd
from fuzzywuzzy import fuzz, process
from typing import Dict, List, Tuple, Optional
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompanyReconciliation:
    """
    Reconcile company entities across multiple data sources
    Core skill for Arch: matching same fund/company across different portals
    """

    def __init__(self):
        # Golden source of canonical company names
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

        # Common variations to normalize
        self.normalization_rules = {
            r'\bInc\.?\b': 'Inc.',
            r'\bCorp\.?\b': 'Corporation',
            r'\bCo\.?\b': 'Company',
            r'\bLtd\.?\b': 'Limited',
            r'\s+': ' ',  # Multiple spaces to single
            r',\s*Inc\.?$': ', Inc.',  # Standardize comma before Inc
        }

    def normalize_name(self, name: str) -> str:
        """Apply normalization rules to company name"""
        if pd.isna(name):
            return name

        normalized = str(name).strip()

        # Apply regex rules
        for pattern, replacement in self.normalization_rules.items():
            normalized = re.sub(pattern, replacement, normalized)

        return normalized

    def fuzzy_match(self, name: str, candidates: List[str], threshold: int = 85) -> Tuple[Optional[str], int]:
        """
        Find best matching name from candidates using fuzzy string matching
        Returns: (best_match, confidence_score)
        """
        if pd.isna(name) or not candidates:
            return None, 0

        # Use fuzzywuzzy for similarity matching
        result = process.extractOne(name, candidates)
        if result is None:
            return None, 0

        best_match, score = result[0], result[1]

        if score >= threshold:
            return best_match, score
        return None, score

    def reconcile_by_ticker(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Reconcile two dataframes using ticker as primary key
        This is the "easy" case - when you have a common identifier
        """
        logger.info(f"\n🔗 Reconciling by ticker...")
        logger.info(f"   Source 1: {len(df1)} records")
        logger.info(f"   Source 2: {len(df2)} records")

        merged = df1.merge(
            df2,
            on='ticker',
            how='outer',
            suffixes=('_src1', '_src2'),
            indicator=True
        )

        # Analyze match quality
        both = len(merged[merged['_merge'] == 'both'])
        left_only = len(merged[merged['_merge'] == 'left_only'])
        right_only = len(merged[merged['_merge'] == 'right_only'])

        logger.info(f"\n   ✅ Matched: {both} ({both/len(df1)*100:.1f}%)")
        logger.info(f"   ⚠️  Only in source 1: {left_only}")
        logger.info(f"   ⚠️  Only in source 2: {right_only}")

        return merged

    def reconcile_by_name(self, df1: pd.DataFrame, df2: pd.DataFrame,
                         name_col1: str = 'company_name',
                         name_col2: str = 'company_name',
                         threshold: int = 85) -> pd.DataFrame:
        """
        Reconcile when ticker is missing - use fuzzy name matching
        This is the "hard" case - mirrors Arch matching funds with slightly different names
        """
        logger.info(f"\n🔗 Reconciling by company name (fuzzy matching)...")

        results = []

        for idx1, row1 in df1.iterrows():
            name1 = self.normalize_name(row1[name_col1])

            # Find best match in df2
            candidates = df2[name_col2].apply(self.normalize_name).tolist()
            best_match, score = self.fuzzy_match(name1, candidates, threshold)

            if best_match:
                # Find the original row in df2
                matched_rows = df2[df2[name_col2].apply(self.normalize_name) == best_match]
                if len(matched_rows) > 0:
                    matched_row = matched_rows.iloc[0]

                    results.append({
                        'name_src1': row1[name_col1],
                        'name_src2': matched_row[name_col2],
                        'match_score': score,
                        'match_type': 'exact' if score == 100 else 'fuzzy',
                        'normalized_name': self.normalize_name(row1[name_col1]),
                        'data_src1': row1.to_dict(),
                        'data_src2': matched_row.to_dict()
                    })
            else:
                results.append({
                    'name_src1': row1[name_col1],
                    'name_src2': None,
                    'match_score': score,
                    'match_type': 'no_match',
                    'normalized_name': self.normalize_name(row1[name_col1]),
                    'data_src1': row1.to_dict(),
                    'data_src2': None
                })

        results_df = pd.DataFrame(results)

        # Statistics
        if len(results_df) > 0:
            exact_matches = len(results_df[results_df['match_type'] == 'exact'])
            fuzzy_matches = len(results_df[results_df['match_type'] == 'fuzzy'])
            no_matches = len(results_df[results_df['match_type'] == 'no_match'])

            logger.info(f"   ✅ Exact matches: {exact_matches}")
            logger.info(f"   🔍 Fuzzy matches: {fuzzy_matches}")
            logger.info(f"   ❌ No match found: {no_matches}")

        return results_df

    def create_golden_record(self, ticker: str, sources: List[Dict]) -> Dict:
        """
        When multiple sources have data for same company, create single "golden record"
        Choose most complete and recent data
        """
        golden = {'ticker': ticker}

        # Use canonical name if we have it
        if ticker in self.canonical_names:
            golden['company_name'] = self.canonical_names[ticker]
        else:
            # Choose name from most reliable source (SEC > Alpha Vantage > Analyst)
            for source in sources:
                if source.get('source') == 'SEC_EDGAR' and source.get('company_name'):
                    golden['company_name'] = source['company_name']
                    break

        # For numeric fields, prefer more recent data
        # Also average when discrepancies are small
        numeric_fields = ['revenue', 'net_income', 'total_assets', 'market_cap', 'ebitda']

        for field in numeric_fields:
            values = [s.get(field) for s in sources if s.get(field) is not None and not pd.isna(s.get(field))]
            if values:
                # If values are very close (<5% difference), average them
                if len(values) > 1:
                    max_val = max(values)
                    min_val = min(values)
                    diff_pct = ((max_val - min_val) / max_val) * 100 if max_val > 0 else 0

                    if diff_pct < 5:
                        golden[field] = sum(values) / len(values)
                        golden[f'{field}_reconciliation'] = 'averaged'
                    else:
                        # Significant difference - use most recent (first in list)
                        golden[field] = values[0]
                        golden[f'{field}_reconciliation'] = 'conflict_used_latest'
                        golden[f'{field}_discrepancy_pct'] = diff_pct
                else:
                    golden[field] = values[0]
                    golden[f'{field}_reconciliation'] = 'single_source'

        # Copy other important fields from first source
        for source in sources:
            if source.get('period_end') and 'period_end' not in golden:
                golden['period_end'] = source.get('period_end')
            break

        # Track data lineage
        golden['sources_used'] = ','.join([s.get('source', 'unknown') for s in sources])
        golden['num_sources'] = len(sources)
        golden['created_at'] = pd.Timestamp.now().isoformat()

        return golden

    def full_reconciliation_workflow(self, sec_df: pd.DataFrame,
                                     av_df: pd.DataFrame,
                                     analyst_df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete reconciliation workflow across all 3 sources
        """
        logger.info("="*60)
        logger.info("STARTING FULL RECONCILIATION WORKFLOW")
        logger.info("="*60)

        # Step 1: Reconcile SEC and Alpha Vantage (both have ticker)
        if 'ticker' in sec_df.columns and 'ticker' in av_df.columns:
            sec_av_merged = self.reconcile_by_ticker(sec_df, av_df)
        else:
            sec_av_merged = sec_df.copy()
            logger.warning("⚠️  Ticker field missing, skipping ticker-based reconciliation")

        # Step 2: Normalize company names across all sources
        logger.info("\n📝 Normalizing company names...")
        for df in [sec_df, av_df, analyst_df]:
            if 'company_name' in df.columns:
                df['company_name_normalized'] = df['company_name'].apply(self.normalize_name)

        # Step 3: Create golden records for matched companies
        logger.info("\n🏆 Creating golden records...")
        golden_records = []

        # Get unique tickers from SEC data (most authoritative)
        if 'ticker' in sec_df.columns:
            matched_tickers = sec_df['ticker'].unique()

            for ticker in matched_tickers:
                if pd.isna(ticker):
                    continue

                sources = []

                # Get SEC data
                sec_records = sec_df[sec_df['ticker'] == ticker]
                if len(sec_records) > 0:
                    sec_record = sec_records.iloc[0].to_dict()
                    sec_record['source'] = 'SEC_EDGAR'
                    sources.append(sec_record)

                # Get Alpha Vantage data
                if 'ticker' in av_df.columns:
                    av_records = av_df[av_df['ticker'] == ticker]
                    if len(av_records) > 0:
                        av_record = av_records.iloc[0].to_dict()
                        av_record['source'] = 'ALPHA_VANTAGE'
                        sources.append(av_record)

                # Try to match analyst data
                if 'ticker' in analyst_df.columns:
                    analyst_matches = analyst_df[analyst_df['ticker'] == ticker]
                    if len(analyst_matches) > 0:
                        analyst_record = analyst_matches.iloc[0].to_dict()
                        analyst_record['source'] = 'ANALYST_REPORT'
                        sources.append(analyst_record)

                # Create golden record
                if sources:
                    golden = self.create_golden_record(ticker, sources)
                    golden_records.append(golden)

        golden_df = pd.DataFrame(golden_records)

        logger.info(f"\n✅ Created {len(golden_df)} golden records")

        # Calculate quality metrics
        if len(golden_df) > 0:
            avg_sources = golden_df['num_sources'].mean()
            logger.info(f"   Average sources per record: {avg_sources:.1f}")

            # Check for discrepancies
            discrepancy_fields = [col for col in golden_df.columns if 'discrepancy_pct' in col]
            if discrepancy_fields:
                for field in discrepancy_fields:
                    avg_discrepancy = golden_df[field].mean()
                    logger.info(f"   Average {field}: {avg_discrepancy:.2f}%")

        return golden_df


if __name__ == '__main__':
    # Test reconciliation
    # Create sample data with variations
    sec_data = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL'],
        'company_name': ['Apple Inc.', 'Microsoft Corporation', 'Alphabet Inc.'],
        'revenue': [385700000000, 211900000000, 307400000000]
    })

    av_data = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL'],
        'company_name': ['Apple Inc', 'Microsoft Corp.', 'Alphabet Inc.'],
        'revenue': [383900000000, 212100000000, 307000000000]
    })

    analyst_data = pd.DataFrame({
        'ticker': ['AAPL', None, 'GOOGL'],
        'company_name': ['Apple, Inc.', 'Microsoft Corporation', 'Google Inc.'],
        'revenue': [385.7, 211.9, 299.8]  # In billions
    })

    print("📊 Original Data:")
    print("\nSEC:")
    print(sec_data)
    print("\nAlpha Vantage:")
    print(av_data)
    print("\nAnalyst:")
    print(analyst_data)

    reconciler = CompanyReconciliation()
    golden_data = reconciler.full_reconciliation_workflow(sec_data, av_data, analyst_data)

    print("\n🏆 Golden Records:")
    print(golden_data[['ticker', 'company_name', 'revenue', 'sources_used']])
