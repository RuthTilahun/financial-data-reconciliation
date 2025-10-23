"""
Pipeline Orchestration Module
End-to-end automated data quality pipeline
Orchestrates all data operations from collection to alert generation
"""

import pandas as pd
from datetime import datetime
import logging
import sys
import os

# Import custom modules
from collection import FinancialDataCollector
from validation import FinancialDataValidator
from cleaning import FinancialDataCleaner
from reconciliation import CompanyReconciliation
from anomaly_detection import FinancialAnomalyDetector
from alerts import AlertSystem

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FinancialDataPipeline:
    """
    End-to-end automated data quality pipeline
    Orchestrates all data operations
    """

    def __init__(self, alpha_vantage_key: str = None):
        self.av_key = alpha_vantage_key
        self.collector = FinancialDataCollector(alpha_vantage_key)
        self.validator = FinancialDataValidator()
        self.cleaner = FinancialDataCleaner()
        self.reconciler = CompanyReconciliation()
        self.detector = FinancialAnomalyDetector()
        self.alerts = AlertSystem()

        self.pipeline_start_time = None
        self.pipeline_metrics = {}

    def run_full_pipeline(self, tickers: list) -> pd.DataFrame:
        """
        Execute complete data quality pipeline
        Returns: golden_df (reconciled data)
        """
        self.pipeline_start_time = datetime.now()
        logger.info("="*60)
        logger.info("STARTING FINANCIAL DATA QUALITY PIPELINE")
        logger.info("="*60)

        try:
            # Step 1: Data Collection
            logger.info("\n🔄 STEP 1: Data Collection")
            sec_df, av_df, analyst_df = self._collect_data(tickers)
            self.pipeline_metrics['records_collected'] = len(sec_df) + len(av_df) + len(analyst_df)

            # Step 2: Initial Validation
            logger.info("\n✅ STEP 2: Data Validation")
            validation_issues = self._validate_data(sec_df, av_df, analyst_df)
            self.pipeline_metrics['validation_issues'] = len(validation_issues)

            # Step 3: Data Cleaning
            logger.info("\n🧹 STEP 3: Data Cleaning")
            sec_clean, av_clean, analyst_clean = self._clean_data(sec_df, av_df, analyst_df)

            # Step 4: Reconciliation
            logger.info("\n🔗 STEP 4: Cross-Source Reconciliation")
            golden_df = self._reconcile_data(sec_clean, av_clean, analyst_clean)
            self.pipeline_metrics['golden_records'] = len(golden_df)

            # Step 5: Anomaly Detection
            logger.info("\n🔍 STEP 5: Anomaly Detection")
            anomalies = self._detect_anomalies(golden_df)
            self.pipeline_metrics['anomalies_detected'] = len(anomalies)

            # Step 6: Generate Alerts
            logger.info("\n📢 STEP 6: Alert Generation")
            self._generate_alerts(anomalies, validation_issues)

            # Step 7: Save Results
            logger.info("\n💾 STEP 7: Saving Results")
            self._save_results(golden_df, anomalies, validation_issues)

            # Step 8: Generate Report
            logger.info("\n📊 STEP 8: Generating Report")
            self._generate_pipeline_report()

            logger.info("\n✅ PIPELINE COMPLETED SUCCESSFULLY")
            self._log_pipeline_metrics()

            return golden_df

        except Exception as e:
            logger.error(f"❌ PIPELINE FAILED: {str(e)}")
            self._handle_pipeline_failure(e)
            raise

    def _collect_data(self, tickers: list) -> tuple:
        """Step 1: Collect data from all sources"""
        logger.info(f"Collecting data for {len(tickers)} companies...")

        try:
            sec_df, av_df, analyst_df = self.collector.run_full_collection(tickers)

            logger.info(f"✅ Collected:")
            logger.info(f"   - SEC: {len(sec_df)} records")
            logger.info(f"   - Alpha Vantage: {len(av_df)} records")
            logger.info(f"   - Analyst Reports: {len(analyst_df)} records")

            return sec_df, av_df, analyst_df

        except Exception as e:
            logger.error(f"Data collection failed: {str(e)}")
            raise

    def _validate_data(self, sec_df, av_df, analyst_df) -> pd.DataFrame:
        """Step 2: Validate all data sources"""
        logger.info("Running validation checks...")

        sec_issues = self.validator.run_all_validations(sec_df, 'SEC_EDGAR')
        av_issues = self.validator.run_all_validations(av_df, 'ALPHA_VANTAGE')
        analyst_issues = self.validator.run_all_validations(analyst_df, 'ANALYST_REPORTS')

        all_issues = pd.concat([sec_issues, av_issues, analyst_issues], ignore_index=True)

        if len(all_issues) > 0:
            logger.warning(f"⚠️  Found {len(all_issues)} validation issues")

            # Log critical issues
            critical = all_issues[all_issues['severity'] == 'critical']
            if len(critical) > 0:
                logger.error(f"🚨 {len(critical)} CRITICAL issues require immediate attention")
        else:
            logger.info("✅ All validation checks passed")

        return all_issues

    def _clean_data(self, sec_df, av_df, analyst_df) -> tuple:
        """Step 3: Clean and standardize data"""
        logger.info("Cleaning data...")

        sec_clean = self.cleaner.full_cleaning_pipeline(sec_df, 'SEC_EDGAR')
        av_clean = self.cleaner.full_cleaning_pipeline(av_df, 'ALPHA_VANTAGE')
        analyst_clean = self.cleaner.full_cleaning_pipeline(analyst_df, 'ANALYST_REPORTS')

        logger.info("✅ Data cleaning complete")

        return sec_clean, av_clean, analyst_clean

    def _reconcile_data(self, sec_df, av_df, analyst_df) -> pd.DataFrame:
        """Step 4: Reconcile across sources"""
        logger.info("Reconciling data sources...")

        golden_df = self.reconciler.full_reconciliation_workflow(sec_df, av_df, analyst_df)

        if len(sec_df) > 0:
            match_rate = len(golden_df) / len(sec_df) * 100
            logger.info(f"✅ Reconciliation complete: {match_rate:.1f}% match rate")
        else:
            logger.warning("⚠️  No SEC data available for reconciliation")

        return golden_df

    def _detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 5: Detect anomalies"""
        logger.info("Running anomaly detection...")

        # Statistical outliers
        if 'revenue' in df.columns:
            self.detector.detect_statistical_outliers(df, 'revenue', method='zscore', threshold=3.0)
        if 'net_income' in df.columns:
            self.detector.detect_statistical_outliers(df, 'net_income', method='iqr', threshold=1.5)

        # Business rule violations
        violations = self.detector.detect_business_rule_violations(df)

        # Cross-metric anomalies
        cross_metric_anomalies = self.detector.detect_cross_metric_anomalies(df)

        # Generate report
        anomalies_df = self.detector.generate_anomaly_report()

        return anomalies_df

    def _generate_alerts(self, anomalies: pd.DataFrame, validation_issues: pd.DataFrame):
        """Step 6: Generate alerts for issues"""
        logger.info("Generating alerts...")

        # Alert on anomalies
        if len(anomalies) > 0:
            self.alerts.check_anomaly_thresholds(anomalies)

        # Alert on validation issues
        if len(validation_issues) > 0:
            self.alerts.check_validation_thresholds(validation_issues)

        # Send alerts
        self.alerts.send_alerts()

    def _save_results(self, golden_df: pd.DataFrame, anomalies_df: pd.DataFrame, issues_df: pd.DataFrame):
        """Step 7: Save all outputs"""
        logger.info("Saving pipeline outputs...")

        # Create directories if they don't exist
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('data/reconciliation', exist_ok=True)

        # Save golden records
        golden_df.to_csv('data/processed/master_financials.csv', index=False)
        logger.info("   ✅ Saved master_financials.csv")

        # Save anomalies
        if len(anomalies_df) > 0:
            anomalies_df.to_csv('data/reconciliation/anomaly_report.csv', index=False)
            logger.info("   ✅ Saved anomaly_report.csv")

        # Save validation issues
        if len(issues_df) > 0:
            issues_df.to_csv('data/reconciliation/validation_issues.csv', index=False)
            logger.info("   ✅ Saved validation_issues.csv")

        # Save pipeline metadata
        metadata = {
            'pipeline_run_date': self.pipeline_start_time,
            'records_processed': self.pipeline_metrics.get('records_collected', 0),
            'golden_records_created': self.pipeline_metrics.get('golden_records', 0),
            'anomalies_detected': self.pipeline_metrics.get('anomalies_detected', 0),
            'validation_issues': self.pipeline_metrics.get('validation_issues', 0)
        }
        pd.DataFrame([metadata]).to_csv('data/processed/pipeline_metadata.csv', index=False)
        logger.info("   ✅ Saved pipeline_metadata.csv")

        # Calculate and save quality scores
        if len(golden_df) > 0:
            quality_scores = []
            for ticker in golden_df['ticker'].unique():
                if pd.notna(ticker):
                    company_data = golden_df[golden_df['ticker'] == ticker]
                    # Simple quality score based on data completeness
                    completeness = company_data.notna().mean().mean() * 100
                    quality_scores.append({
                        'ticker': ticker,
                        'company_name': company_data['company_name'].iloc[0] if 'company_name' in company_data.columns else '',
                        'quality_score': round(completeness, 2),
                        'num_sources': company_data['num_sources'].iloc[0] if 'num_sources' in company_data.columns else 1
                    })

            if quality_scores:
                pd.DataFrame(quality_scores).to_csv('data/reconciliation/company_quality_scores.csv', index=False)
                logger.info("   ✅ Saved company_quality_scores.csv")

    def _generate_pipeline_report(self):
        """Step 8: Generate summary report"""
        end_time = datetime.now()
        duration = (end_time - self.pipeline_start_time).total_seconds()

        report = f"""
{'='*60}
PIPELINE EXECUTION REPORT
{'='*60}
Start Time: {self.pipeline_start_time.strftime('%Y-%m-%d %H:%M:%S')}
End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration:.1f} seconds

METRICS:
- Records Collected: {self.pipeline_metrics.get('records_collected', 0)}
- Golden Records Created: {self.pipeline_metrics.get('golden_records', 0)}
- Validation Issues: {self.pipeline_metrics.get('validation_issues', 0)}
- Anomalies Detected: {self.pipeline_metrics.get('anomalies_detected', 0)}
- Alerts Generated: {len(self.alerts.alerts)}

STATUS: ✅ SUCCESS
{'='*60}
"""

        logger.info(report)

        # Save report to file
        os.makedirs('logs', exist_ok=True)
        report_filename = f'logs/pipeline_report_{self.pipeline_start_time.strftime("%Y%m%d_%H%M%S")}.txt'
        with open(report_filename, 'w') as f:
            f.write(report)

        logger.info(f"   ✅ Saved pipeline report to {report_filename}")

    def _log_pipeline_metrics(self):
        """Log final metrics"""
        logger.info("\n📊 PIPELINE METRICS:")
        for metric, value in self.pipeline_metrics.items():
            logger.info(f"   {metric}: {value}")

    def _handle_pipeline_failure(self, error):
        """Handle pipeline failures gracefully"""
        logger.error("Pipeline failed - generating error report...")

        # Send critical alert
        self.alerts.generate_alert(
            severity='critical',
            title='Pipeline Execution Failed',
            description=f'Error: {str(error)}',
            action_required='Investigate pipeline logs and fix root cause'
        )
        self.alerts.send_alerts()

        logger.info("Partial results may have been saved")


def main():
    """Main execution function"""
    # Configuration
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
               'JPM', 'JNJ', 'WMT', 'DIS', 'XOM']

    # Run pipeline
    pipeline = FinancialDataPipeline()

    try:
        golden_data = pipeline.run_full_pipeline(TICKERS)
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 Created {len(golden_data)} golden records")

        # Display sample results
        if len(golden_data) > 0:
            print("\n🏆 Sample Golden Records:")
            display_cols = ['ticker', 'company_name', 'revenue', 'net_income', 'sources_used']
            existing_cols = [col for col in display_cols if col in golden_data.columns]
            print(golden_data[existing_cols].head())

    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
