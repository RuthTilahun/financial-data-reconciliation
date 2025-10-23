"""
Alert System Module
Generates alerts for data quality issues
Simulates what Arch's monitoring system would do
"""

import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSystem:
    """
    Generate alerts for data quality issues
    Simulates what Arch's monitoring system would do
    """

    def __init__(self):
        self.alerts = []

    def generate_alert(self, severity: str, title: str, description: str,
                      affected_records: Optional[List[str]] = None,
                      action_required: Optional[str] = None) -> Dict:
        """
        Create an alert
        """
        alert = {
            'alert_id': f"ALERT_{len(self.alerts)+1:04d}",
            'timestamp': datetime.now(),
            'severity': severity,
            'title': title,
            'description': description,
            'affected_records': affected_records or [],
            'action_required': action_required,
            'status': 'open'
        }
        self.alerts.append(alert)
        return alert

    def check_anomaly_thresholds(self, anomalies_df: pd.DataFrame):
        """
        Generate alerts based on anomaly counts and severity
        """
        if len(anomalies_df) == 0:
            return

        # Alert 1: Critical anomalies
        critical = anomalies_df[anomalies_df['severity'] == 'critical']
        if len(critical) > 0:
            affected = critical['ticker'].tolist() if 'ticker' in critical.columns else []
            self.generate_alert(
                severity='critical',
                title=f'{len(critical)} CRITICAL Data Anomalies Detected',
                description=f"Found {len(critical)} critical anomalies requiring immediate attention",
                affected_records=affected,
                action_required='Investigate and resolve immediately before data release'
            )

        # Alert 2: High volume of issues
        if len(anomalies_df) > 10:
            self.generate_alert(
                severity='high',
                title='High Volume of Anomalies',
                description=f"Detected {len(anomalies_df)} total anomalies (threshold: 10)",
                action_required='Review data source quality and collection process'
            )

        # Alert 3: Specific ticker with multiple issues
        if 'ticker' in anomalies_df.columns:
            ticker_counts = anomalies_df['ticker'].value_counts()
            problematic_tickers = ticker_counts[ticker_counts > 3]
            if len(problematic_tickers) > 0:
                for ticker, count in problematic_tickers.items():
                    if pd.notna(ticker):
                        self.generate_alert(
                            severity='medium',
                            title=f'Multiple Issues for {ticker}',
                            description=f"{ticker} has {count} anomalies - may indicate systematic data problem",
                            affected_records=[ticker],
                            action_required=f'Deep dive investigation into {ticker} data sources'
                        )

    def check_validation_thresholds(self, issues_df: pd.DataFrame):
        """
        Generate alerts based on validation issues
        """
        if len(issues_df) == 0:
            return

        # Alert on critical validation issues
        critical = issues_df[issues_df['severity'] == 'critical']
        if len(critical) > 0:
            self.generate_alert(
                severity='critical',
                title=f'{len(critical)} Critical Validation Failures',
                description='Data quality checks failed for critical fields',
                action_required='Review and fix data sources before proceeding'
            )

        # Alert on high completeness issues
        completeness_issues = issues_df[issues_df['rule'] == 'completeness']
        if len(completeness_issues) > 5:
            self.generate_alert(
                severity='high',
                title='High Number of Missing Data Issues',
                description=f'Found {len(completeness_issues)} fields with missing data',
                action_required='Investigate data collection process for gaps'
            )

    def format_alert_message(self, alert: Dict) -> str:
        """
        Format alert for display/email
        """
        severity_emoji = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }

        affected_str = ', '.join(str(x) for x in alert['affected_records'][:5])
        if len(alert['affected_records']) > 5:
            affected_str += '...'

        message = f"""
{severity_emoji.get(alert['severity'], '⚪')} {alert['severity'].upper()} ALERT
Alert ID: {alert['alert_id']}
Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

{alert['title']}
{'-' * 60}
{alert['description']}

Affected Records: {len(alert['affected_records'])}
{affected_str}

Action Required:
{alert['action_required'] or 'Review and acknowledge'}

Status: {alert['status'].upper()}
"""
        return message

    def send_alerts(self):
        """
        Send all pending alerts
        In production: would email/Slack, here we just print
        """
        if not self.alerts:
            logger.info("\n✅ No alerts to send")
            return

        logger.info(f"\n📧 SENDING {len(self.alerts)} ALERTS")
        logger.info("="*60)

        for alert in sorted(self.alerts, key=lambda x: {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x['severity']]):
            print(self.format_alert_message(alert))
            print("="*60)

        # Save to file
        alerts_df = pd.DataFrame(self.alerts)
        alerts_df.to_csv('data/reconciliation/alerts_log.csv', index=False)
        logger.info(f"\n💾 Alerts saved to data/reconciliation/alerts_log.csv")

    def get_alert_summary(self) -> Dict:
        """
        Get summary statistics of alerts
        """
        if not self.alerts:
            return {
                'total_alerts': 0,
                'by_severity': {},
                'open_alerts': 0
            }

        alerts_df = pd.DataFrame(self.alerts)

        return {
            'total_alerts': len(alerts_df),
            'by_severity': alerts_df['severity'].value_counts().to_dict(),
            'open_alerts': len(alerts_df[alerts_df['status'] == 'open']),
            'affected_companies': len(set([r for alert in self.alerts for r in alert['affected_records']]))
        }


if __name__ == '__main__':
    # Test alert system
    alert_system = AlertSystem()

    # Generate some test alerts
    alert_system.generate_alert(
        severity='critical',
        title='Revenue Data Anomaly Detected',
        description='Company X shows revenue of $99T (obviously incorrect)',
        affected_records=['TEST'],
        action_required='Verify data source and correct value'
    )

    alert_system.generate_alert(
        severity='medium',
        title='Date Format Inconsistency',
        description='Found 15 records with non-standard date formats',
        action_required='Standardize all dates to ISO 8601'
    )

    # Send alerts
    alert_system.send_alerts()

    # Get summary
    summary = alert_system.get_alert_summary()
    print("\n📊 Alert Summary:")
    print(f"   Total alerts: {summary['total_alerts']}")
    print(f"   By severity: {summary['by_severity']}")
    print(f"   Open alerts: {summary['open_alerts']}")
