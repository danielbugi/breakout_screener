import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
import json
import os
from symbol_scraper import SymbolLoader  # Import our symbol loader

warnings.filterwarnings('ignore')


class EnhancedDonchianBreakoutScreener:
    def __init__(self, period=20, lookback_days=252, symbol_file_dir='stock_lists'):
        """
        Enhanced Donchian Channel Breakout Screener with symbol file integration

        Args:
            period (int): Donchian channel period (default: 20 days)
            lookback_days (int): How many days of historical data to fetch (default: 252 = 1 year)
            symbol_file_dir (str): Directory containing scraped symbol files
        """
        self.period = period
        self.lookback_days = lookback_days
        self.symbol_file_dir = symbol_file_dir
        self.symbol_loader = SymbolLoader()

        # Load available symbol files
        self._load_available_symbol_lists()

    def _load_available_symbol_lists(self):
        """Load available symbol lists from scraped files with proper error handling"""
        self.available_lists = {}

        if not os.path.exists(self.symbol_file_dir):
            raise FileNotFoundError(
                f"❌ Symbol directory '{self.symbol_file_dir}' not found.\n"
                f"   Please ensure the symbol scraper has been run and the directory exists.\n"
                f"   Expected path: {os.path.abspath(self.symbol_file_dir)}"
            )

        try:
            # Get latest symbol files
            latest_files = self.symbol_loader.get_latest_symbol_files(self.symbol_file_dir)

            if not latest_files:
                raise ValueError(
                    f"❌ No symbol files found in '{self.symbol_file_dir}'.\n"
                    f"   Please run the symbol scraper first to generate symbol lists."
                )

            # Load each available list
            loaded_count = 0
            failed_lists = []

            for index_name, file_path in latest_files.items():
                try:
                    symbols = self.symbol_loader.load_symbols_from_file(file_path)
                    if symbols:
                        self.available_lists[index_name] = symbols
                        loaded_count += 1
                        print(f"✅ Loaded {len(symbols)} symbols for {index_name.upper()}")
                    else:
                        failed_lists.append(f"{index_name} (empty file)")
                except Exception as e:
                    failed_lists.append(f"{index_name} ({str(e)})")
                    print(f"⚠️  Error loading {index_name}: {str(e)}")

            if loaded_count == 0:
                raise ValueError(
                    f"❌ Failed to load any symbol lists.\n"
                    f"   Failed lists: {', '.join(failed_lists)}\n"
                    f"   Please check the symbol files and run the symbol scraper again."
                )

            print(f"📊 Successfully loaded {loaded_count} symbol lists: {list(self.available_lists.keys())}")

            if failed_lists:
                print(f"⚠️  Failed to load {len(failed_lists)} lists: {', '.join(failed_lists)}")

        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            else:
                raise RuntimeError(
                    f"❌ Unexpected error loading symbol lists: {str(e)}\n"
                    f"   Please check the symbol_scraper module and file permissions."
                )

    def get_stock_list(self, list_name):
        """
        Get stock list by name from scraped files

        Args:
            list_name (str): Name of the stock list

        Returns:
            list: List of stock symbols

        Raises:
            ValueError: If list_name is not found
        """
        if not self.available_lists:
            raise ValueError(
                "❌ No symbol lists available. Please run the symbol scraper first."
            )

        # Direct match
        if list_name in self.available_lists:
            return self.available_lists[list_name]

        # Special handling for variations
        list_mapping = {
            'sp500': 'sp500',
            's&p500': 'sp500',
            'nasdaq': 'nasdaq100',
            'nasdaq100': 'nasdaq100',
            'russell': 'russell1000',
            'russell1000': 'russell1000',
            'all': 'all_indices',
            'combined': 'all_indices'
        }

        mapped_name = list_mapping.get(list_name.lower())
        if mapped_name and mapped_name in self.available_lists:
            return self.available_lists[mapped_name]

        # Generate helpful error message
        available_names = list(self.available_lists.keys())
        suggestions = [name for name in available_names if list_name.lower() in name.lower()]

        error_msg = f"❌ List '{list_name}' not found.\n"
        error_msg += f"   Available lists: {', '.join(available_names)}"

        if suggestions:
            error_msg += f"\n   Did you mean: {', '.join(suggestions)}?"

        raise ValueError(error_msg)

    def list_available_stock_lists(self):
        """Display all available stock lists"""
        print("\n📋 AVAILABLE STOCK LISTS:")
        print("=" * 50)

        if self.available_lists:
            print("🔍 Available symbol lists:")
            for name, symbols in self.available_lists.items():
                print(f"   {name:<20} | {len(symbols):>4} symbols")

            print(f"\n📊 Total lists: {len(self.available_lists)}")
            print(f"💡 Usage: screener.screen_multiple_stocks(list_name='list_name')")
        else:
            print("❌ No symbol lists available!")
            print("   Please run the symbol scraper first to generate symbol lists.")
            print("   Expected directory: stock_lists/")
            raise ValueError("No symbol lists available for screening.")

    def calculate_donchian_channel(self, df):
        """
        Calculate Donchian Channel for given price data

        Args:
            df (DataFrame): DataFrame with OHLCV data

        Returns:
            DataFrame: Original data with Donchian channel columns added
        """
        # Calculate Donchian Channel
        df['donchian_high'] = df['High'].rolling(window=self.period).max()
        df['donchian_low'] = df['Low'].rolling(window=self.period).min()
        df['donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2

        # Calculate channel width (useful for volatility analysis)
        df['channel_width'] = df['donchian_high'] - df['donchian_low']
        df['channel_width_pct'] = (df['channel_width'] / df['donchian_mid']) * 100

        return df

    def detect_breakouts(self, df):
        """
        Detect Donchian channel breakouts

        Args:
            df (DataFrame): DataFrame with price data and Donchian channels

        Returns:
            dict: Breakout information
        """
        if len(df) < self.period + 1:
            return None

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        # Detect breakouts
        # bullish_breakout = (latest['Close'] > latest['donchian_high'] and
        #                     previous['Close'] <= previous['donchian_high'])
        bullish_breakout = (latest['Close'] > previous['donchian_high'] and
                            previous['Close'] <= previous['donchian_high'])

        # bearish_breakout = (latest['Close'] < latest['donchian_low'] and
        #                     previous['Close'] >= previous['donchian_low'])
        bearish_breakout = (latest['Close'] < previous['donchian_low'] and
                            previous['Close'] >= previous['donchian_low'])

        # Calculate additional metrics
        price_position = ((latest['Close'] - latest['donchian_low']) /
                          (latest['donchian_high'] - latest['donchian_low'])) * 100

        # Volume analysis (if volume data is available)
        avg_volume = df['Volume'].tail(20).mean() if 'Volume' in df.columns else 0
        current_volume = latest.get('Volume', 0)
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # Calculate price change and volatility
        price_change_pct = ((latest['Close'] - previous['Close']) / previous['Close']) * 100

        # Calculate ATR (Average True Range) for volatility
        df['tr'] = np.maximum(df['High'] - df['Low'],
                              np.maximum(abs(df['High'] - df['Close'].shift()),
                                         abs(df['Low'] - df['Close'].shift())))
        atr = df['tr'].tail(14).mean()
        atr_pct = (atr / latest['Close']) * 100

        return {
            'symbol': None,  # Will be set by caller
            'date': latest.name.strftime('%Y-%m-%d'),
            'close_price': round(latest['Close'], 2),
            'donchian_high': round(latest['donchian_high'], 2),
            'donchian_low': round(latest['donchian_low'], 2),
            'donchian_mid': round(latest['donchian_mid'], 2),
            'bullish_breakout': bullish_breakout,
            'bearish_breakout': bearish_breakout,
            'price_position_pct': round(price_position, 1),
            'channel_width_pct': round(latest['channel_width_pct'], 2),
            'volume_ratio': round(volume_ratio, 2),
            'price_change_pct': round(price_change_pct, 2),
            'atr_pct': round(atr_pct, 2),
            'days_since_high': self._days_since_extreme(df, 'High'),
            'days_since_low': self._days_since_extreme(df, 'Low')
        }

    def _days_since_extreme(self, df, column):
        """Calculate days since highest high or lowest low"""
        if column == 'High':
            max_idx = df['High'].tail(self.period).idxmax()
        else:
            max_idx = df['Low'].tail(self.period).idxmin()

        return len(df) - df.index.get_loc(max_idx) - 1

    def fetch_stock_data(self, symbol):
        """
        Fetch stock data for a given symbol

        Args:
            symbol (str): Stock symbol

        Returns:
            DataFrame: Stock price data or None if failed
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)

            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                return None

            return df

        except Exception as e:
            return None

    def screen_stock(self, symbol):
        """
        Screen a single stock for Donchian breakouts

        Args:
            symbol (str): Stock symbol

        Returns:
            dict: Breakout analysis or None
        """
        df = self.fetch_stock_data(symbol)
        if df is None:
            return None

        df = self.calculate_donchian_channel(df)
        result = self.detect_breakouts(df)

        if result:
            result['symbol'] = symbol

        return result

    def screen_multiple_stocks(self, stock_list=None, list_name='all_indices', batch_size=50, max_stocks=None):
        """
        Screen multiple stocks for breakouts with enhanced features

        Args:
            stock_list (list): Custom list of stock symbols
            list_name (str): Name of predefined list
            batch_size (int): Number of stocks to process in each batch
            max_stocks (int): Maximum number of stocks to screen (for testing)

        Returns:
            dict: Screening results
        """
        try:
            if stock_list is None:
                stock_list = self.get_stock_list(list_name)
        except ValueError as e:
            print(str(e))
            return {
                'bullish_breakouts': [],
                'bearish_breakouts': [],
                'near_breakouts': [],
                'all_results': [],
                'failed_symbols': [],
                'error': str(e)
            }

        if not stock_list:
            error_msg = f"❌ No stocks found for list '{list_name}'"
            print(error_msg)
            return {
                'bullish_breakouts': [],
                'bearish_breakouts': [],
                'near_breakouts': [],
                'all_results': [],
                'failed_symbols': [],
                'error': error_msg
            }

        # Limit stocks for testing if specified
        if max_stocks:
            stock_list = stock_list[:max_stocks]
            print(f"🔬 Testing mode: Limited to {max_stocks} stocks")

        results = {
            'bullish_breakouts': [],
            'bearish_breakouts': [],
            'near_breakouts': [],
            'all_results': [],
            'failed_symbols': []
        }

        print(f"🔍 Screening {len(stock_list)} stocks from '{list_name}' for Donchian channel breakouts...")
        print(f"📊 Using {self.period}-day Donchian channels")
        print(f"⚙️  Processing in batches of {batch_size}")
        print("-" * 80)

        total_processed = 0

        # Process in batches
        for batch_start in range(0, len(stock_list), batch_size):
            batch_end = min(batch_start + batch_size, len(stock_list))
            batch_symbols = stock_list[batch_start:batch_end]

            print(f"\n📦 Processing batch {batch_start // batch_size + 1} "
                  f"({batch_start + 1}-{batch_end} of {len(stock_list)})")

            for symbol in batch_symbols:
                print(f"   {symbol:<6}", end=' ... ')

                result = self.screen_stock(symbol)
                total_processed += 1

                if result:
                    results['all_results'].append(result)

                    if result['bullish_breakout']:
                        results['bullish_breakouts'].append(result)
                        print("🚀 BULLISH BREAKOUT!")
                    elif result['bearish_breakout']:
                        results['bearish_breakouts'].append(result)
                        print("📉 BEARISH BREAKOUT!")
                    elif result['price_position_pct'] > 90:
                        results['near_breakouts'].append(result)
                        print("⬆️  Near bullish breakout")
                    elif result['price_position_pct'] < 10:
                        results['near_breakouts'].append(result)
                        print("⬇️  Near bearish breakout")
                    else:
                        print("✓")
                else:
                    results['failed_symbols'].append(symbol)
                    print("❌ Failed")

                # Small delay to avoid overwhelming the API
                time.sleep(0.05)

            # Progress update
            progress = (total_processed / len(stock_list)) * 100
            print(f"\n📈 Progress: {total_processed}/{len(stock_list)} ({progress:.1f}%)")

            # Longer delay between batches
            if batch_end < len(stock_list):
                time.sleep(1)

        # Final summary
        print(f"\n✅ Screening completed!")
        print(f"   📊 Processed: {total_processed}/{len(stock_list)} stocks")
        print(f"   🚀 Bullish breakouts: {len(results['bullish_breakouts'])}")
        print(f"   📉 Bearish breakouts: {len(results['bearish_breakouts'])}")
        print(f"   ⚠️  Near breakouts: {len(results['near_breakouts'])}")
        print(f"   ❌ Failed: {len(results['failed_symbols'])}")

        return results

    def export_for_nextjs(self, results, filename=None):
        """
        Export results in a format optimized for Next.js consumption

        Args:
            results (dict): Results from screen_multiple_stocks
            filename (str): Optional filename for JSON export

        Returns:
            dict: Formatted data for Next.js
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'donchian_breakouts_{timestamp}.json'

        # Structure data for frontend consumption
        export_data = {
            'metadata': {
                'scan_date': datetime.now().isoformat(),
                'period': self.period,
                'total_scanned': len(results['all_results']),
                'bullish_count': len(results['bullish_breakouts']),
                'bearish_count': len(results['bearish_breakouts']),
                'near_breakout_count': len(results['near_breakouts']),
                'failed_count': len(results['failed_symbols'])
            },
            'breakouts': {
                'bullish': sorted(results['bullish_breakouts'],
                                  key=lambda x: x['volume_ratio'], reverse=True),
                'bearish': sorted(results['bearish_breakouts'],
                                  key=lambda x: x['volume_ratio'], reverse=True),
                'near_bullish': [r for r in results['near_breakouts'] if r['price_position_pct'] > 90],
                'near_bearish': [r for r in results['near_breakouts'] if r['price_position_pct'] < 10]
            },
            'summary_stats': {
                'avg_volume_ratio_bullish': np.mean([r['volume_ratio'] for r in results['bullish_breakouts']]) if
                results['bullish_breakouts'] else 0,
                'avg_volume_ratio_bearish': np.mean([r['volume_ratio'] for r in results['bearish_breakouts']]) if
                results['bearish_breakouts'] else 0,
                'top_performers': sorted(results['all_results'],
                                         key=lambda x: x['price_change_pct'], reverse=True)[:10],
                'worst_performers': sorted(results['all_results'],
                                           key=lambda x: x['price_change_pct'])[:10]
            }
        }

        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"\n💾 Next.js data exported to: {filename}")
        print(f"   📊 Structure: {list(export_data.keys())}")

        return export_data

    def print_results(self, results):
        """
        Print formatted screening results with enhanced display

        Args:
            results (dict): Results from screen_multiple_stocks
        """
        print("\n" + "=" * 90)
        print(f"DONCHIAN CHANNEL BREAKOUT SCREENING RESULTS")
        print(f"Period: {self.period} days | Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 90)

        # Bullish breakouts
        if results['bullish_breakouts']:
            print(f"\n🚀 BULLISH BREAKOUTS ({len(results['bullish_breakouts'])} found):")
            print("-" * 80)
            print(
                f"{'Symbol':<8} {'Price':<10} {'Change%':<10} {'Vol Ratio':<10} {'ATR%':<8} {'Days Since Low':<15}")
            print("-" * 80)
            for stock in sorted(results['bullish_breakouts'], key=lambda x: x['volume_ratio'], reverse=True):
                print(f"{stock['symbol']:<8} ${stock['close_price']:<9.2f} {stock['price_change_pct']:<9.1f}% "
                      f"{stock['volume_ratio']:<9.1f}x {stock['atr_pct']:<7.1f}% {stock['days_since_low']:<15}")

        # Bearish breakouts
        if results['bearish_breakouts']:
            print(f"\n📉 BEARISH BREAKOUTS ({len(results['bearish_breakouts'])} found):")
            print("-" * 80)
            print(
                f"{'Symbol':<8} {'Price':<10} {'Change%':<10} {'Vol Ratio':<10} {'ATR%':<8} {'Days Since High':<15}")
            print("-" * 80)
            for stock in sorted(results['bearish_breakouts'], key=lambda x: x['volume_ratio'], reverse=True):
                print(f"{stock['symbol']:<8} ${stock['close_price']:<9.2f} {stock['price_change_pct']:<9.1f}% "
                      f"{stock['volume_ratio']:<9.1f}x {stock['atr_pct']:<7.1f}% {stock['days_since_high']:<15}")

        # Near breakouts (top 10)
        if results['near_breakouts']:
            near_sorted = sorted(results['near_breakouts'],
                                 key=lambda x: max(x['price_position_pct'], 100 - x['price_position_pct']),
                                 reverse=True)[:10]

            print(f"\n⚠️  TOP NEAR BREAKOUTS ({len(near_sorted)} of {len(results['near_breakouts'])} shown):")
            print("-" * 80)
            print(f"{'Symbol':<8} {'Price':<10} {'Position':<10} {'Direction':<12} {'Change%':<10}")
            print("-" * 80)
            for stock in near_sorted:
                direction = "Bullish" if stock['price_position_pct'] > 50 else "Bearish"
                print(f"{stock['symbol']:<8} ${stock['close_price']:<9.2f} {stock['price_position_pct']:<9.1f}% "
                      f"{direction:<12} {stock['price_change_pct']:<9.1f}%")

        if not any([results['bullish_breakouts'], results['bearish_breakouts'], results['near_breakouts']]):
            print("\nNo breakouts or near-breakouts found in the screened stocks.")

        # Show failed symbols if any
        if results.get('failed_symbols'):
            print(f"\n❌ FAILED TO ANALYZE ({len(results['failed_symbols'])}):")
            failed_str = ', '.join(results['failed_symbols'][:20])
            if len(results['failed_symbols']) > 20:
                failed_str += f" ... and {len(results['failed_symbols']) - 20} more"
            print(f"   {failed_str}")

    def save_results_to_csv(self, results, filename=None):
        """
        Save results to CSV file with enhanced data

        Args:
            results (dict): Results from screen_multiple_stocks
            filename (str): Custom filename, if None uses timestamp
        """
        if not results['all_results']:
            print("No results to save.")
            return

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'donchian_breakout_screen_{timestamp}.csv'

        df = pd.DataFrame(results['all_results'])

        # Add breakout type column
        df['breakout_type'] = 'None'
        df.loc[df['bullish_breakout'], 'breakout_type'] = 'Bullish'
        df.loc[df['bearish_breakout'], 'breakout_type'] = 'Bearish'
        df.loc[(df['price_position_pct'] > 90) & (df['breakout_type'] == 'None'), 'breakout_type'] = 'Near Bullish'
        df.loc[(df['price_position_pct'] < 10) & (df['breakout_type'] == 'None'), 'breakout_type'] = 'Near Bearish'

        # Sort by breakout type and volume ratio
        df = df.sort_values(['breakout_type', 'volume_ratio'], ascending=[True, False])

        df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to: {filename}")
        print(f"   📊 Total records: {len(df)}")


# Main execution script for production
def run_donchian_screening():
    """
    Main function to run the Donchian screening for production use
    """
    print("🚀 Starting Donchian Breakout Screening...")
    print("=" * 60)

    try:
        # Initialize screener
        screener = EnhancedDonchianBreakoutScreener(period=20)

        # Show available lists
        screener.list_available_stock_lists()

        # Run screening on all indices
        print(f"\n🔍 Running screening on 'all_indices'...")
        results = screener.screen_multiple_stocks(list_name='all_indices')

        # Check if screening was successful
        if 'error' in results:
            print(f"\n❌ Screening failed: {results['error']}")
            return None, None

        # Print results
        screener.print_results(results)

        # Save to CSV for analysis
        screener.save_results_to_csv(results)

        # Export for Next.js
        nextjs_data = screener.export_for_nextjs(results)

        print(f"\n✅ Screening complete! Ready for Next.js integration.")
        return results, nextjs_data

    except FileNotFoundError as e:
        print(f"\n{str(e)}")
        print("\n🔧 To fix this issue:")
        print("   1. Make sure you have run the symbol scraper")
        print("   2. Check that the 'stock_lists' directory exists")
        print("   3. Verify symbol files are present in the directory")
        return None, None

    except ValueError as e:
        print(f"\n{str(e)}")
        print("\n🔧 To fix this issue:")
        print("   1. Run the symbol scraper to generate fresh symbol lists")
        print("   2. Check that symbol files are not empty or corrupted")
        return None, None

    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("\n🔧 Please check:")
        print("   1. Internet connection for downloading stock data")
        print("   2. All required dependencies are installed")
        print("   3. File permissions in the working directory")
        return None, None


# Example usage
if __name__ == "__main__":
    try:
        results, nextjs_data = run_donchian_screening()
        if results is None:
            print("\n❌ Screening could not be completed.")
            exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Screening interrupted by user.")
        exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        exit(1)