"""
Synthetic Banking Transaction Data Generator
Generates realistic banking transaction datasets with intentional quality issues
for testing data reconciliation pipelines.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


class SyntheticBankingDataGenerator:
    """Generate synthetic banking transaction datasets with realistic issues"""

    def __init__(self):
        self.currencies = ['USD', 'EUR', 'GBP', 'CAD', 'AUD']
        self.transaction_types = ['DEPOSIT', 'WITHDRAWAL', 'TRANSFER', 'PAYMENT', 'FEE']
        self.statuses = ['COMPLETED', 'PENDING', 'FAILED', 'REVERSED']
        self.descriptions = [
            'ATM Withdrawal', 'Online Purchase', 'Wire Transfer', 'Direct Deposit',
            'Bill Payment', 'Check Deposit', 'Card Purchase', 'Transfer In',
            'Transfer Out', 'Monthly Fee', 'Interest Payment', 'Refund'
        ]

        # Generate consistent account numbers pool (to ensure some overlap across datasets)
        self.account_numbers = [f"ACC{str(i).zfill(8)}" for i in range(1000, 11000)]

        # Generate customer IDs
        self.customer_ids = [f"CUST{str(i).zfill(6)}" for i in range(5000, 10000)]

        # Branch codes
        self.branch_codes = [f"BR{str(i).zfill(3)}" for i in range(1, 51)]

        # Merchant codes
        self.merchant_codes = [f"MER{str(i).zfill(5)}" for i in range(1000, 2000)]

        # Teller IDs
        self.teller_ids = [f"T{str(i).zfill(4)}" for i in range(100, 250)]

    def generate_transaction_id(self, prefix='TXN', length=12):
        """Generate a unique transaction ID"""
        suffix = ''.join(random.choices(string.digits, k=length-len(prefix)))
        return f"{prefix}{suffix}"

    def generate_date_range(self, start_date, days_back=365):
        """Generate random datetime within date range"""
        start = start_date - timedelta(days=days_back)
        random_days = random.randint(0, days_back)
        random_seconds = random.randint(0, 86400)
        return start - timedelta(days=random_days, seconds=random_seconds)

    def generate_cbs_transactions(self, num_records=50000):
        """
        Generate Core Banking System transactions
        Fields: transaction_id, account_number, transaction_date, amount,
                transaction_type, status, branch_code, customer_id
        Issues: NULL values, duplicate entries
        """
        print(f"Generating {num_records} CBS transactions...")

        data = []
        base_date = datetime(2024, 10, 23)

        for i in range(num_records):
            transaction_id = self.generate_transaction_id('CBS')
            account_number = random.choice(self.account_numbers)
            transaction_date = self.generate_date_range(base_date, days_back=180)
            amount = round(random.uniform(1.0, 50000.0), 2)
            transaction_type = random.choice(self.transaction_types)
            status = random.choice(self.statuses)
            branch_code = random.choice(self.branch_codes)
            customer_id = random.choice(self.customer_ids)

            data.append({
                'transaction_id': transaction_id,
                'account_number': account_number,
                'transaction_date': transaction_date.strftime('%Y-%m-%d %H:%M:%S'),
                'amount': amount,
                'transaction_type': transaction_type,
                'status': status,
                'branch_code': branch_code,
                'customer_id': customer_id
            })

        df = pd.DataFrame(data)

        # Introduce NULL values (3% of records)
        null_indices = np.random.choice(df.index, size=int(len(df) * 0.03), replace=False)
        null_columns = ['status', 'branch_code', 'customer_id', 'transaction_type']

        for idx in null_indices:
            col = random.choice(null_columns)
            df.loc[idx, col] = None

        # Introduce duplicate entries (2% of records)
        duplicate_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
        duplicates = df.loc[duplicate_indices].copy()

        # Modify transaction_id for duplicates but keep other fields same
        duplicates['transaction_id'] = duplicates['transaction_id'].apply(
            lambda x: self.generate_transaction_id('CBS')
        )

        df = pd.concat([df, duplicates], ignore_index=True)

        # Shuffle the dataframe
        df = df.sample(frac=1).reset_index(drop=True)

        print(f"✅ Generated {len(df)} CBS transactions (including {len(duplicates)} duplicates)")
        return df

    def generate_pgs_payments(self, num_records=35000):
        """
        Generate Payment Gateway System transactions
        Fields: payment_ref, account_no, payment_datetime, payment_amount,
                currency, merchant_code, auth_code, settlement_date
        Issues: Different date formats, currency conversions needed, missing auth_codes
        """
        print(f"Generating {num_records} PGS payments...")

        data = []
        base_date = datetime(2024, 10, 23)

        # Different date format styles
        date_formats = [
            '%Y-%m-%d %H:%M:%S',  # Standard
            '%m/%d/%Y %I:%M %p',  # US format with AM/PM
            '%d-%b-%Y %H:%M',     # Day-Month-Year
            '%Y%m%d%H%M%S',       # Compact format
            '%d/%m/%Y %H:%M:%S'   # European format
        ]

        for i in range(num_records):
            payment_ref = self.generate_transaction_id('PGS', length=15)
            account_no = random.choice(self.account_numbers)
            payment_datetime = self.generate_date_range(base_date, days_back=150)

            # Use different currencies with different amounts
            currency = random.choice(self.currencies)
            if currency == 'USD':
                payment_amount = round(random.uniform(1.0, 10000.0), 2)
            elif currency == 'EUR':
                payment_amount = round(random.uniform(1.0, 9000.0), 2)
            elif currency == 'GBP':
                payment_amount = round(random.uniform(1.0, 8000.0), 2)
            else:
                payment_amount = round(random.uniform(1.0, 12000.0), 2)

            merchant_code = random.choice(self.merchant_codes)
            auth_code = self.generate_transaction_id('AUTH', length=8)

            # Settlement date is typically 1-3 days after payment
            settlement_days = random.randint(1, 3)
            settlement_date = payment_datetime + timedelta(days=settlement_days)

            # Apply different date format
            date_format = random.choice(date_formats)
            formatted_datetime = payment_datetime.strftime(date_format)
            formatted_settlement = settlement_date.strftime(date_format)

            data.append({
                'payment_ref': payment_ref,
                'account_no': account_no,
                'payment_datetime': formatted_datetime,
                'payment_amount': payment_amount,
                'currency': currency,
                'merchant_code': merchant_code,
                'auth_code': auth_code,
                'settlement_date': formatted_settlement
            })

        df = pd.DataFrame(data)

        # Introduce missing auth_codes (15% of records)
        missing_auth_indices = np.random.choice(df.index, size=int(len(df) * 0.15), replace=False)
        df.loc[missing_auth_indices, 'auth_code'] = None

        # Some records with missing settlement_date (5%)
        missing_settlement_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
        df.loc[missing_settlement_indices, 'settlement_date'] = None

        print(f"✅ Generated {len(df)} PGS payments with mixed date formats and missing values")
        return df

    def generate_legacy_transactions(self, num_records=20000):
        """
        Generate Legacy System transactions
        Fields: trans_code, acct_num, trans_dt, debit_amt, credit_amt,
                balance, description, teller_id
        Issues: Separate debit/credit columns, inconsistent account formats,
                corrupted amounts (negative values where impossible)
        """
        print(f"Generating {num_records} Legacy transactions...")

        data = []
        base_date = datetime(2024, 10, 23)

        # Different account number formats (inconsistent)
        def format_account_inconsistently(account):
            """Apply inconsistent formatting to account numbers"""
            format_type = random.choice(['normal', 'dashed', 'no_prefix', 'spaces'])
            if format_type == 'normal':
                return account
            elif format_type == 'dashed':
                # ACC12345678 -> ACC-1234-5678
                return f"{account[:3]}-{account[3:7]}-{account[7:]}"
            elif format_type == 'no_prefix':
                # ACC12345678 -> 12345678
                return account[3:]
            else:  # spaces
                # ACC12345678 -> ACC 12345678
                return f"{account[:3]} {account[3:]}"

        for i in range(num_records):
            trans_code = self.generate_transaction_id('LEG', length=10)
            account = random.choice(self.account_numbers)
            acct_num = format_account_inconsistently(account)
            trans_dt = self.generate_date_range(base_date, days_back=200)

            # Randomly decide if debit or credit
            is_debit = random.choice([True, False])
            amount = round(random.uniform(1.0, 25000.0), 2)

            if is_debit:
                debit_amt = amount
                credit_amt = 0.0
            else:
                debit_amt = 0.0
                credit_amt = amount

            # Balance between 0 and 100,000
            balance = round(random.uniform(0.0, 100000.0), 2)
            description = random.choice(self.descriptions)
            teller_id = random.choice(self.teller_ids)

            # Legacy date format (various old formats)
            legacy_date_formats = ['%m/%d/%y', '%d-%m-%Y', '%Y%m%d', '%m-%d-%Y']
            formatted_date = trans_dt.strftime(random.choice(legacy_date_formats))

            data.append({
                'trans_code': trans_code,
                'acct_num': acct_num,
                'trans_dt': formatted_date,
                'debit_amt': debit_amt,
                'credit_amt': credit_amt,
                'balance': balance,
                'description': description,
                'teller_id': teller_id
            })

        df = pd.DataFrame(data)

        # Introduce corrupted negative values (3% of records)
        corrupt_indices = np.random.choice(df.index, size=int(len(df) * 0.03), replace=False)
        for idx in corrupt_indices:
            # Make balance or amounts negative where they shouldn't be
            corruption_type = random.choice(['balance', 'debit', 'credit'])
            if corruption_type == 'balance':
                df.loc[idx, 'balance'] = -abs(df.loc[idx, 'balance'])
            elif corruption_type == 'debit':
                df.loc[idx, 'debit_amt'] = -abs(df.loc[idx, 'debit_amt'])
            else:
                df.loc[idx, 'credit_amt'] = -abs(df.loc[idx, 'credit_amt'])

        # Some NULL teller_ids (2%)
        null_teller_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
        df.loc[null_teller_indices, 'teller_id'] = None

        print(f"✅ Generated {len(df)} Legacy transactions with inconsistent formats and corrupted data")
        return df


def main():
    """Generate all synthetic datasets"""
    print("=" * 80)
    print("SYNTHETIC BANKING TRANSACTION DATA GENERATOR")
    print("=" * 80)
    print()

    generator = SyntheticBankingDataGenerator()

    # Generate datasets
    print("\n📊 Generating datasets...\n")

    # 1. Core Banking System
    cbs_df = generator.generate_cbs_transactions(num_records=50000)
    cbs_path = 'data/raw/cbs_transactions.csv'
    cbs_df.to_csv(cbs_path, index=False)
    print(f"   Saved to: {cbs_path}\n")

    # 2. Payment Gateway System
    pgs_df = generator.generate_pgs_payments(num_records=35000)
    pgs_path = 'data/raw/pgs_payments.csv'
    pgs_df.to_csv(pgs_path, index=False)
    print(f"   Saved to: {pgs_path}\n")

    # 3. Legacy System
    legacy_df = generator.generate_legacy_transactions(num_records=20000)
    legacy_path = 'data/raw/legacy_transactions.csv'
    legacy_df.to_csv(legacy_path, index=False)
    print(f"   Saved to: {legacy_path}\n")

    # Summary
    print("=" * 80)
    print("✅ GENERATION COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  • CBS Transactions:     {len(cbs_df):>7,} records (with NULLs and duplicates)")
    print(f"  • PGS Payments:         {len(pgs_df):>7,} records (mixed date formats, missing auth codes)")
    print(f"  • Legacy Transactions:  {len(legacy_df):>7,} records (inconsistent formats, corrupted data)")
    print(f"  • Total:                {len(cbs_df) + len(pgs_df) + len(legacy_df):>7,} records")
    print()
    print("Data Quality Issues Introduced:")
    print("  ✓ NULL values in critical fields")
    print("  ✓ Duplicate transaction entries")
    print("  ✓ Inconsistent date formats across datasets")
    print("  ✓ Multiple currency types")
    print("  ✓ Missing authorization codes")
    print("  ✓ Inconsistent account number formats")
    print("  ✓ Corrupted negative amounts")
    print("  ✓ Separate debit/credit columns (requires consolidation)")
    print()
    print("Ready for reconciliation pipeline testing!")
    print("=" * 80)


if __name__ == '__main__':
    main()
