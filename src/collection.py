"""
Data Collection Module
Collects financial data from multiple sources:
- SEC EDGAR API
- Alpha Vantage API
- Simulated Analyst Reports (intentionally messy)
"""

import requests
import pandas as pd
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialDataCollector:
    """Collect financial data from multiple sources"""

    def __init__(self, alpha_vantage_key: Optional[str] = None, fmp_key: Optional[str] = None):
        self.av_key = alpha_vantage_key
        self.fmp_key = fmp_key
        self.sec_headers = {
            'User-Agent': 'Financial-Research research@example.com'
        }

        # Company CIK mapping (SEC identifier)
        self.ticker_to_cik = {
            'AAPL': '0000320193',
            'MSFT': '0000789019',
            'GOOGL': '0001652044',
            'AMZN': '0001018724',
            'TSLA': '0001318605',
            'JPM': '0000019617',
            'JNJ': '0000200406',
            'WMT': '0000104169',
            'DIS': '0001744489',
            'XOM': '0000034088'
        }

    def collect_sec_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Collect data from SEC EDGAR
        Note: This is simplified - real implementation would parse XBRL
        """
        logger.info(f"📊 Collecting SEC EDGAR data for {len(tickers)} companies...")

        results = []

        # Sample SEC data (in production, would fetch from actual API)
        sec_sample_data = {
            'AAPL': {'revenue': 385706000000, 'net_income': 99803000000, 'total_assets': 365725000000},
            'MSFT': {'revenue': 211915000000, 'net_income': 72361000000, 'total_assets': 411976000000},
            'GOOGL': {'revenue': 307394000000, 'net_income': 73795000000, 'total_assets': 402392000000},
            'AMZN': {'revenue': 574785000000, 'net_income': 30425000000, 'total_assets': 527854000000},
            'TSLA': {'revenue': 96773000000, 'net_income': 14974000000, 'total_assets': 106618000000},
            'JPM': {'revenue': 158096000000, 'net_income': 48334000000, 'total_assets': 3875000000000},
            'JNJ': {'revenue': 85159000000, 'net_income': 17941000000, 'total_assets': 187378000000},
            'WMT': {'revenue': 648125000000, 'net_income': 11680000000, 'total_assets': 252399000000},
            'DIS': {'revenue': 88898000000, 'net_income': 2354000000, 'total_assets': 205580000000},
            'XOM': {'revenue': 344582000000, 'net_income': 36010000000, 'total_assets': 376317000000}
        }

        for ticker in tickers:
            if ticker not in self.ticker_to_cik:
                logger.warning(f"Ticker {ticker} not in CIK mapping, skipping")
                continue

            try:
                cik = self.ticker_to_cik[ticker]

                # In production, would fetch from:
                # url = f"https://data.sec.gov/submissions/CIK{cik}.json"
                # For this demo, using sample data

                data = sec_sample_data.get(ticker, {})

                results.append({
                    'cik': cik,
                    'ticker': ticker,
                    'company_name': self._get_company_name(ticker),
                    'filing_date': (datetime.now() - timedelta(days=random.randint(10, 60))).strftime('%Y-%m-%d'),
                    'period_end': '2024-09-30',
                    'filing_type': '10-K',
                    'revenue': data.get('revenue', 0),
                    'net_income': data.get('net_income', 0),
                    'total_assets': data.get('total_assets', 0),
                    'source': 'SEC_EDGAR',
                    'collected_at': datetime.now().isoformat()
                })

                time.sleep(0.1)  # Be nice to servers

            except Exception as e:
                logger.error(f"Error collecting SEC data for {ticker}: {e}")

        df = pd.DataFrame(results)
        logger.info(f"✅ Collected {len(df)} SEC records")
        return df

    def collect_alpha_vantage_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Collect data from Alpha Vantage API
        Note: Requires API key. Using sample data for demo.
        """
        logger.info(f"📈 Collecting Alpha Vantage data for {len(tickers)} companies...")

        results = []

        # Sample Alpha Vantage data
        av_sample_data = {
            'AAPL': {'market_cap': 3450000000000, 'pe_ratio': 35.2, 'revenue': 383933000000, 'ebitda': 123498000000},
            'MSFT': {'market_cap': 3100000000000, 'pe_ratio': 36.8, 'revenue': 212051000000, 'ebitda': 109438000000},
            'GOOGL': {'market_cap': 2150000000000, 'pe_ratio': 28.4, 'revenue': 307039000000, 'ebitda': 98502000000},
            'AMZN': {'market_cap': 1900000000000, 'pe_ratio': 62.3, 'revenue': 575146000000, 'ebitda': 71654000000},
            'TSLA': {'market_cap': 850000000000, 'pe_ratio': 56.8, 'revenue': 97154000000, 'ebitda': 13656000000},
            'JPM': {'market_cap': 620000000000, 'pe_ratio': 12.8, 'revenue': 158338000000, 'ebitda': 65230000000},
            'JNJ': {'market_cap': 390000000000, 'pe_ratio': 21.7, 'revenue': 85294000000, 'ebitda': 26742000000},
            'WMT': {'market_cap': 525000000000, 'pe_ratio': 45.0, 'revenue': 648398000000, 'ebitda': 35600000000},
            'DIS': {'market_cap': 210000000000, 'pe_ratio': 89.2, 'revenue': 91361000000, 'ebitda': 14289000000},
            'XOM': {'market_cap': 515000000000, 'pe_ratio': 14.3, 'revenue': 344281000000, 'ebitda': 73512000000}
        }

        for ticker in tickers:
            try:
                # In production with API key:
                # url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={self.av_key}'
                # response = requests.get(url)
                # data = response.json()

                data = av_sample_data.get(ticker, {})

                results.append({
                    'ticker': ticker,
                    'company_name': self._get_company_name(ticker),
                    'market_cap': data.get('market_cap', 0),
                    'pe_ratio': data.get('pe_ratio', 0),
                    'revenue': data.get('revenue', 0),
                    'net_income': data.get('revenue', 0) * 0.15,  # Estimated
                    'ebitda': data.get('ebitda', 0),
                    'period_end': '2024-09-30',
                    'last_updated': datetime.now().strftime('%Y-%m-%d'),
                    'source': 'ALPHA_VANTAGE',
                    'collected_at': datetime.now().isoformat()
                })

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error collecting Alpha Vantage data for {ticker}: {e}")

        df = pd.DataFrame(results)
        logger.info(f"✅ Collected {len(df)} Alpha Vantage records")
        return df

    def generate_messy_analyst_data(self, base_tickers: List[str]) -> pd.DataFrame:
        """
        Generate intentionally messy analyst reports to simulate real-world data quality issues
        This is the "secret weapon" - demonstrates understanding of common data problems
        """
        logger.info(f"📝 Generating messy analyst reports for {len(base_tickers)} companies...")

        results = []
        analyst_firms = ['Morgan Stanley', 'Goldman Sachs', 'Barclays', 'Credit Suisse', 'JP Morgan']

        # Base data similar to other sources
        base_data = {
            'AAPL': {'revenue': 385.7, 'net_income': 99.8},
            'MSFT': {'revenue': 211.9, 'net_income': 72.4},
            'GOOGL': {'revenue': 307.4, 'net_income': 73.8},
            'AMZN': {'revenue': 574.8, 'net_income': 30.4},
            'TSLA': {'revenue': 96.8, 'net_income': 15.0},
            'JPM': {'revenue': 158.1, 'net_income': 48.3},
            'JNJ': {'revenue': 85.2, 'net_income': 17.9},
            'WMT': {'revenue': 648.1, 'net_income': 11.7},
            'DIS': {'revenue': 88.9, 'net_income': 2.4},
            'XOM': {'revenue': 344.6, 'net_income': 36.0}
        }

        for ticker in base_tickers:
            # Create 1-2 analyst reports per company
            for i in range(random.randint(1, 2)):
                data = base_data.get(ticker, {'revenue': 100, 'net_income': 10})

                # Introduce realistic errors
                company_name = self._get_company_name(ticker)
                revenue = data['revenue']
                net_income = data['net_income']

                # Error 1: Name variations (50% chance)
                if random.random() < 0.5:
                    company_name = self._introduce_name_variation(company_name)

                # Error 2: Unit inconsistency (30% chance - mix billions and millions)
                if random.random() < 0.3:
                    revenue = revenue * 1000  # Convert to millions
                    net_income = net_income * 1000

                # Error 3: Date format variations
                date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y']
                report_date = datetime.now() - timedelta(days=random.randint(5, 45))
                date_format = random.choice(date_formats)

                # Error 4: Slight value variations (simulating different reporting periods)
                revenue_variation = random.uniform(0.98, 1.02)
                revenue = revenue * revenue_variation

                results.append({
                    'ticker': ticker if random.random() > 0.1 else None,  # 10% missing ticker
                    'company_name': company_name,
                    'report_date': report_date.strftime(date_format),
                    'analyst_firm': random.choice(analyst_firms),
                    'revenue': revenue,
                    'net_income': net_income,
                    'recommendation': random.choice(['Buy', 'Hold', 'Sell', 'Overweight', 'Underweight']),
                    'target_price': random.randint(100, 300),
                    'source': 'ANALYST_REPORT',
                    'collected_at': datetime.now().isoformat()
                })

        # Error 5: Add some duplicate entries with slight variations
        if len(results) > 3:
            dupes = random.sample(results, min(3, len(results)))
            for dupe in dupes:
                dupe_copy = dupe.copy()
                dupe_copy['revenue'] = dupe_copy['revenue'] * 1.01  # Slight variation
                dupe_copy['analyst_firm'] = random.choice(analyst_firms)
                results.append(dupe_copy)

        # Error 6: Add an obvious outlier/typo
        if results:
            outlier_idx = random.randint(0, len(results)-1)
            results[outlier_idx]['revenue'] = 9999999999999  # Obvious typo

        df = pd.DataFrame(results)
        logger.info(f"✅ Generated {len(df)} analyst reports (with intentional data quality issues)")
        return df

    def _get_company_name(self, ticker: str) -> str:
        """Get canonical company name"""
        names = {
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
        return names.get(ticker, ticker)

    def _introduce_name_variation(self, name: str) -> str:
        """Introduce realistic name variations"""
        variations = {
            'Apple Inc.': ['Apple Inc', 'Apple, Inc.', 'Apple Incorporated'],
            'Microsoft Corporation': ['Microsoft Corp.', 'Microsoft Corp', 'MSFT Corporation'],
            'Alphabet Inc.': ['Google Inc.', 'Alphabet', 'Google'],
            'Amazon.com, Inc.': ['Amazon', 'Amazon Inc', 'Amazon.com Inc'],
            'Tesla, Inc.': ['Tesla Inc', 'Tesla Motors', 'Tesla'],
            'JPMorgan Chase & Co.': ['JP Morgan Chase', 'JPMorgan Chase', 'J.P. Morgan Chase & Co.'],
            'Walmart Inc.': ['Walmart', 'Wal-Mart Inc.', 'WMT Inc'],
            'The Walt Disney Company': ['Disney', 'Walt Disney Co.', 'Disney Company']
        }

        if name in variations:
            return random.choice(variations[name])
        return name

    def run_full_collection(self, tickers: List[str]) -> tuple:
        """
        Orchestrate all data collection
        Returns: (sec_df, av_df, analyst_df)
        """
        logger.info("="*60)
        logger.info("STARTING DATA COLLECTION")
        logger.info("="*60)

        # Collect from all sources
        sec_df = self.collect_sec_data(tickers)
        av_df = self.collect_alpha_vantage_data(tickers)
        analyst_df = self.generate_messy_analyst_data(tickers)

        # Save raw data
        sec_df.to_csv('data/raw/sec_filings.csv', index=False)
        av_df.to_csv('data/raw/alpha_vantage.csv', index=False)
        analyst_df.to_csv('data/raw/analyst_reports.csv', index=False)

        logger.info("\n✅ Data collection complete!")
        logger.info(f"   SEC: {len(sec_df)} records → data/raw/sec_filings.csv")
        logger.info(f"   Alpha Vantage: {len(av_df)} records → data/raw/alpha_vantage.csv")
        logger.info(f"   Analyst Reports: {len(analyst_df)} records → data/raw/analyst_reports.csv")

        return sec_df, av_df, analyst_df


if __name__ == '__main__':
    # Test data collection
    collector = FinancialDataCollector()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    sec_data, av_data, analyst_data = collector.run_full_collection(tickers)

    print("\n📊 Sample SEC Data:")
    print(sec_data.head())
    print("\n📈 Sample Alpha Vantage Data:")
    print(av_data.head())
    print("\n📝 Sample Analyst Data:")
    print(analyst_data.head())
