import datetime

def get_current_time() -> str:
    """
    Returns the current date and time.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")    


import yfinance as yf

def get_stock_price(symbol: str) -> str:
    """Returns the latest stock price and basic info for the given ticker symbol (e.g. AAPL, MSFT, GOOGL).
    
    Args:
        symbol: The stock ticker symbol (e.g. 'AAPL' for Apple, 'MSFT' for Microsoft).
    
    Returns:
        A string with the current price, currency, company name, and market state.
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.fast_info
        price = info.last_price
        currency = info.currency
        prev_close = info.previous_close
        change = price - prev_close
        change_pct = (change / prev_close) * 100
        return (
            f"Symbol: {symbol.upper()}\n"
            f"Price: {price:.2f} {currency}\n"
            f"Change: {change:+.2f} ({change_pct:+.2f}%)\n"
            f"Previous Close: {prev_close:.2f} {currency}"
        )
    except Exception as e:
        return f"Error fetching stock price for '{symbol}': {e}"

system_instruction = "You are a helpful assistant with access to tools."

tools = [get_current_time, get_stock_price]