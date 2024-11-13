import requests
import pandas as pd
from datetime import datetime
import time
from enum import Enum
import os

class MarketCapCategory(Enum):
    MEGA_CAP = (100e9, float('inf'), 'Mega Cap')     
    LARGE_CAP = (20e9, 100e9, 'Large Cap')           
    MEDIUM_CAP = (5e9, 20e9, 'Medium Cap')           
    SMALL_CAP = (1e9, 5e9, 'Small Cap')              
    MICRO_CAP = (100e6, 1e9, 'Micro Cap')            
    NANO_CAP = (10e6, 100e6, 'Nano Cap')             
    PICO_CAP = (0, 10e6, 'Pico Cap')                 

class ATHCategory(Enum):
    NEAR_ATH = (0, 20, 'Near ATH')           
    SLIGHT_DIP = (20, 40, 'Slight Dip')      
    CORRECTION = (40, 60, 'Correction')       
    BEAR_MARKET = (60, 80, 'Bear Market')     
    DEEP_BEAR = (80, 90, 'Deep Bear')         
    EXTREME_BEAR = (90, float('inf'), 'Extreme Bear')

class CryptoClassifier:
    def __init__(self):
        self.base_url = 'https://api.coingecko.com/api/v3'
        self.data_dir = 'crypto_data'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def _make_request(self, url, params=None):
        """Make API request with retry logic"""
        try:
            response = requests.get(url, params=params)
            if response.status_code == 429:
                print("Rate limit reached. Waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(url, params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            return None

    def fetch_crypto_data(self, limit=50):
        """Fetch data for multiple cryptocurrencies including their categories"""
        markets_url = f"{self.base_url}/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': limit,
            'page': 1,
            'sparkline': False,
            'price_change_percentage': '24h,7d,30d',
            'locale': 'en'
        }
        
        market_data = self._make_request(markets_url, params)
        if not market_data:
            return pd.DataFrame()

        detailed_data = []
        for coin in market_data:
            # Fetch detailed data including categories
            coin_info_url = f"{self.base_url}/coins/{coin['id']}"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'false',
                'community_data': 'false',
                'developer_data': 'false'
            }
            coin_info = self._make_request(coin_info_url, params)
            
            if not coin_info:
                continue
                
            ath_date = datetime.strptime(coin['ath_date'], '%Y-%m-%dT%H:%M:%S.%fZ')
            days_since_ath = (datetime.now() - ath_date).days
            
            crypto_info = {
                'id': coin['id'],
                'name': coin['name'],
                'symbol': coin['symbol'].upper(),
                'current_price': coin['current_price'],
                'market_cap': coin['market_cap'],
                'ath': coin['ath'],
                'ath_date': ath_date,
                'days_since_ath': days_since_ath,
                'volume_24h': coin['total_volume'],
                'price_change_24h': coin['price_change_percentage_24h'],
                'price_change_7d': coin.get('price_change_percentage_7d_in_currency'),
                'price_change_30d': coin.get('price_change_percentage_30d_in_currency'),
                'market_cap_rank': coin['market_cap_rank'],
                'categories': coin_info.get('categories', []),
                'category': ', '.join(coin_info.get('categories', ['Unknown'])[:2])  # Take top 2 categories
            }
            detailed_data.append(crypto_info)
            time.sleep(0.6)  # Rate limiting
            
        return pd.DataFrame(detailed_data)

    def calculate_ath_metrics(self, current_price, ath):
        """Calculate ATH-related metrics"""
        if ath == 0 or pd.isna(ath):
            return 0, 0
        down_from_ath = ((ath - current_price) / ath) * 100
        price_to_recover = ((ath / current_price) - 1) * 100
        return down_from_ath, price_to_recover

    def classify_ath_status(self, down_from_ath):
        """Classify based on ATH distance"""
        for category in ATHCategory:
            if category.value[0] <= down_from_ath < category.value[1]:
                return category.value[2]
        return ATHCategory.EXTREME_BEAR.value[2]

    def classify_market_cap(self, market_cap):
        """Classify based on market cap"""
        for category in MarketCapCategory:
            if category.value[0] <= market_cap < category.value[1]:
                return category.value[2]
        return MarketCapCategory.PICO_CAP.value[2]

    def save_to_csv(self, df, filename_prefix='crypto_classification'):
        """Save DataFrame to CSV with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}_{timestamp}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        # Save both formatted and raw versions
        df.to_csv(filepath, index=False)
        
        # Create raw version with numeric values
        raw_filename = f"{filename_prefix}_raw_{timestamp}.csv"
        raw_filepath = os.path.join(self.data_dir, raw_filename)
        
        numeric_columns = [
            'current_price', 'market_cap', 'ath', 'volume_24h',
            'price_change_24h', 'price_change_7d', 'price_change_30d',
            'down_from_ath', 'price_to_recover'
        ]
        
        raw_df = df.copy()
        for col in numeric_columns:
            if col in raw_df.columns:
                raw_df[col] = pd.to_numeric(raw_df[col].str.rstrip('%').str.replace('$', '').str.replace(',', ''), errors='ignore')
        
        raw_df.to_csv(raw_filepath, index=False)
        
        return filepath, raw_filepath

    def classify_cryptocurrencies(self, limit=50):
        """Main function to classify cryptocurrencies"""
        try:
            print(f"Fetching data for top {limit} cryptocurrencies...")
            df = self.fetch_crypto_data(limit)
            if df.empty:
                return df
            
            print("Processing classifications...")
            # Calculate ATH metrics
            ath_metrics = df.apply(lambda x: pd.Series(
                self.calculate_ath_metrics(x['current_price'], x['ath'])), axis=1)
            df['down_from_ath'] = ath_metrics[0]
            df['price_to_recover'] = ath_metrics[1]
            df['ath_status'] = df['down_from_ath'].apply(self.classify_ath_status)
            
            # Market cap classifications
            df['market_cap_category'] = df['market_cap'].apply(self.classify_market_cap)
            
            # Format numeric columns
            df['current_price'] = df['current_price'].apply(lambda x: f"${x:,.4f}")
            df['ath'] = df['ath'].apply(lambda x: f"${x:,.4f}")
            df['market_cap'] = df['market_cap'].apply(lambda x: f"${x:,.0f}")
            df['volume_24h'] = df['volume_24h'].apply(lambda x: f"${x:,.0f}")
            df['down_from_ath'] = df['down_from_ath'].apply(lambda x: f"{x:.2f}%")
            df['price_to_recover'] = df['price_to_recover'].apply(lambda x: f"{x:.2f}%")
            
            # Format price changes
            for period in ['24h', '7d', '30d']:
                col = f'price_change_{period}'
                df[col] = df[col].apply(
                    lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A"
                )
            
            # Select and reorder columns
            columns = [
                'market_cap_rank',
                'name',
                'symbol',
                'category',
                'current_price',
                'ath',
                'down_from_ath',
                'ath_status',
                'days_since_ath',
                'market_cap',
                'market_cap_category',
                'volume_24h',
                'price_change_24h',
                'price_change_7d',
                'price_change_30d'
            ]
            
            df = df[columns].sort_values('market_cap_rank')
            
            # Save to CSV
            filepath, raw_filepath = self.save_to_csv(df)
            print(f"\nData saved to:")
            print(f"Formatted CSV: {filepath}")
            print(f"Raw CSV: {raw_filepath}")
            
            # Print summary of categories found
            print("\nCategory Distribution:")
            category_counts = df['category'].value_counts()
            for category, count in category_counts.items():
                print(f"{category}: {count} coins")
            
            return df
            
        except Exception as e:
            print(f"Error occurred: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    classifier = CryptoClassifier()
    
    print("\nFetching and classifying cryptocurrencies...")
    results = classifier.classify_cryptocurrencies(limit=50)
    
    if not results.empty:
        print("\nCryptocurrency Classification Results:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(results.to_string(index=False))
    else:
        print("No data retrieved. Please check your internet connection or try again later.")