## python-portfolio-tracker Functions

# import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


def load_transactions(excel_path, sheet_name=0):
    """
    Loads transaction data from an Excel file, parses dtypes, drops unnecessary columns,
    sorts by Date and Ticker, and sets Date as the index.

    Parameters:
    - excel_path: str, path to the Excel file
    - sheet_name: str or int, sheet name or position (default: 0)

    Returns:
    - transactions: pd.DataFrame, cleaned and sorted transaction data
    """
    
    # Define dtypes
    dtypes = {
        'ISIN': str,
        'Ticker': str,
        'Shares': int,
        'Amount': float,
        'Commissions': float,
        'Taxes': float,
    }

    # Load transactions
    transactions = pd.read_excel(
        excel_path, 
        sheet_name=sheet_name, 
        dtype=dtypes, 
        parse_dates=['Date']
    )

    # Drop unnecessary columns
    transactions = transactions.drop(columns=['Name', 'ISIN'])
    
    # Preceed with minus sign the number of shares and amount of the selling/dividends transactions
    transactions.loc[transactions['Type'] == 'SELL','Shares'] *= -1
    transactions.loc[transactions['Type'] == 'SELL','Amount'] *= -1
    transactions.loc[transactions['Type'] == 'DIVIDEND','Amount'] *= -1

    # Sort by Date and Ticker, set Date as index
    transactions = transactions.sort_values(by=['Date', 'Ticker']).set_index('Date')

    return transactions

  
def portfolio_cash_flows(transactions):
    """
    Calculate the daily cash flows from a transaction DataFrame.

    This function computes the net daily cash flows based on the 'Amount',
    'Commissions', and 'Taxes' columns. It groups transactions by 'Date'
    and sums the relevant columns to determine the total daily cash flow.

    Args:
        transactions (pd.DataFrame): A DataFrame containing at least the following columns:
            - 'Date' (datetime-like index or column)
            - 'Amount' (float): The transaction amount.
            - 'Commissions' (float): The commission fees associated with the transaction.
            - 'Taxes' (float): The taxes associated with the transaction.

    Returns:
        pd.Series: A time series (indexed by 'Date') representing the daily total cash flows.

    Notes:
        - If 'Date' is not a column but an index, it will be reset.
        - The resulting Series will have the name 'Cash_Flows'.
    """
    # Ensure 'Date' is a column (reset index only if needed)
    if transactions.index.name == 'Date':
        transactions = transactions.reset_index()

    # Group by 'Date' and sum the specified columns
    grouped = transactions.groupby('Date')[['Amount', 'Commissions', 'Taxes']].sum()

    # Calculate daily cash flow
    cash_flows = grouped.sum(axis=1)

    # Name the resulting Series
    cash_flows.name = 'Cash_Flows'

    return cash_flows


def portfolio_pl(transactions: pd.DataFrame, market_suffix:str = '.MI', price_ref: str = 'close') -> pd.DataFrame:
    """
    Calculate portfolio-level Profit & Loss (P/L) per Ticker.

    Parameters:
    -----------
    transactions : pd.DataFrame
        DataFrame of portfolio transactions with at least the following columns:
        ['Ticker', 'Type', 'Amount', 'Shares', 'Commissions', 'Taxes']
        
    market_suffix : str
        Suffix to append to tickers for market identification: '.MI' (default) for Borsa Italiana
        
    price_ref : str
        Method to calculate price: 'close' (default) or 'intraday_range' (mean of high and low)

    Returns:
    --------
    pd.DataFrame
        Portfolio P/L per ticker with last available price date as index.
    """
    # Step 0: Get all tickers
    all_tickers = transactions['Ticker'].unique()

    # 1. Total Subscriptions (only 'BUY')
    subs = transactions[transactions['Type'] == 'BUY'].groupby('Ticker')['Amount'].sum()
    subs.name = 'Subscriptions'

    # 2. Total Redemptions ('SELL' as positive inflows)
    redemps = transactions[transactions['Type'] == 'SELL'].groupby('Ticker')['Amount'].sum().abs()
    redemps.name = 'Redemptions'

    # 3. Total Dividends
    div = transactions[transactions['Type'] == 'Dividend'].groupby('Ticker')['Amount'].sum().abs()
    div.name = 'Dividends'

    # 4. Total Costs (Commissions + Taxes)
    costs = transactions.groupby('Ticker')[['Commissions', 'Taxes']].sum().sum(axis=1)
    costs.name = 'Costs'

    # 5. Current shares held
    shares = transactions.groupby('Ticker')['Shares'].sum()
    shares.name = 'Current_Shares'

    # 6. Prices and last date
    tickers_full = [t + market_suffix for t in all_tickers]
    price_data = yf.download(tickers_full, period='1d', group_by='ticker', progress=False, auto_adjust=True)

    prices = {}
    dates = {}
    for t in all_tickers:
        label = t + market_suffix
        if price_ref == 'close':
            val = price_data[label]['Close'].iloc[0]
        elif price_ref == 'intraday_range':
            val = price_data[label][['High', 'Low']].mean(axis=1).iloc[0]
        else:
            raise ValueError("price_ref must be 'close' or 'intraday_range'")
        prices[t] = val
        dates[t] = price_data.index[0].strftime('%Y-%m-%d')

    prices = pd.Series(prices, name='Last_Price')
    close_dates = pd.Series(dates, name='Last_Date')

    # 7. Combine everything and compute P/L
    pf_pl = pd.concat([shares, subs, redemps, div, costs, prices, close_dates], axis=1).fillna(0)

    pf_pl['Market_Value'] = pf_pl['Current_Shares'] * pf_pl['Last_Price']
    pf_pl['Total Return (%)'] = (
        (pf_pl['Market_Value'] + pf_pl['Redemptions'] + pf_pl['Dividends'])/
        (pf_pl['Subscriptions'] + pf_pl['Costs']) - 1) * 100

    return pf_pl


def portfolio_summary(transactions: pd.DataFrame, market_suffix: str = '.MI',
                      price_ref: str = 'close') -> pd.Series:
    """
    Generate a summary of the overall portfolio from the transactions DataFrame.

    Parameters:
    -----------
    transactions : pd.DataFrame
        DataFrame of portfolio transactions with at least the following columns:
        ['Ticker', 'Type', 'Amount', 'Shares', 'Commissions', 'Taxes']
    
    market_suffix : str
        Suffix to append to tickers for market identification: '.MI' (default) for Borsa Italiana
        
    price_ref : str
        Method to calculate price: 'close' (default) or 'intraday_range' (mean of high and low)
        
    Returns:
    --------
    pd.Series
        A Series summarizing Subscriptions, Redemptions, Dividends, Costs, and Current Value.
    """
    pf_pl = portfolio_pl(transactions, market_suffix = market_suffix, price_ref=price_ref)
    
    summary = pf_pl[['Subscriptions', 'Redemptions', 'Dividends', 'Costs', 'Market_Value']].sum()
    summary['Profit (Loss)'] = summary.Market_Value + summary.Redemptions + summary.Dividends\
        - summary.Costs - summary.Subscriptions
    summary['Total Return (%)'] = (summary['Profit (Loss)'] / (summary.Subscriptions + summary.Costs)) * 100
    
    return summary


def compare_asset_allocation(current_allocation: pd.Series, strategy_percentages: list, 
                             strategy_labels: list, title_date: str = ''):
    """
    Compare current and strategic asset allocation using pie charts.

    Parameters:
    -----------
    current_allocation : pd.Series
        Series with tickers as index and current value as values.
        
    strategy_percentages : list
        List of planned allocation percentages in the same order as tickers.
        
    strategy_labels : list
        List of tickers or asset names matching the order in strategy_percentages.
        
    title_date : str
        Optional date string to annotate current allocation chart.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].pie(current_allocation, labels=current_allocation.index,
              autopct='%1.1f%%', pctdistance=1.25, labeldistance=0.6)
    ax[0].set_title(f'Current Asset Allocation {f"({title_date})" if title_date else ""}')

    ax[1].pie(strategy_percentages, labels=strategy_labels,
              autopct='%1.1f%%', pctdistance=1.25, labeldistance=0.6)
    ax[1].set_title('Strategy (Planned) Asset Allocation')

    plt.tight_layout()
    plt.show()


def calculate_rebalance(current_portfolio_data, strategy_percentages, amount_invest=0):
    """
    Calculate the number of shares to buy or sell for each asset to reach target portfolio weights.

    Parameters:
    -----------
    current_portfolio_data : pd.DataFrame
        A DataFrame containing at least the following columns:
        - 'Current_Shares': current number of shares held
        - 'Market_Value' : current market value of the position
        - 'Last_Price'   : latest price per share
        The DataFrame index should contain asset tickers.

    strategy_percentages : list of float
        Target allocation weights as percentages for each asset, in the same order as current_portfolio_data.

    amount_invest : float, optional (default=0)
        Additional capital to invest during rebalancing.

    Returns:
    --------
    pd.DataFrame
        A DataFrame including:
        - 'Current_Shares'
        - 'Target_Shares'
        - 'Shares_To_Trade': positive means buy, negative means sell

    Also prints:
    ------------
    A list of assets and the number of shares to buy/sell to reach target allocation.
    """
    # Extract and copy relevant columns
    allocation = current_portfolio_data.loc[:, ['Current_Shares', 'Market_Value', 'Last_Price']].copy()

    # Total portfolio value including additional investment
    total_value = allocation['Market_Value'].sum() + amount_invest

    # Calculate target values and shares
    target_weights = np.array(strategy_percentages) / 100
    allocation['Target_Value'] = total_value * target_weights
    allocation['Target_Shares'] = (allocation['Target_Value'] // allocation['Last_Price']).astype(int)
    allocation['Shares_To_Trade'] = allocation['Target_Shares'] - allocation['Current_Shares']

    return allocation[['Current_Shares', 'Target_Shares', 'Shares_To_Trade']]


def get_ticker_timeseries(transactions: pd.DataFrame, ticker: str, market_suffix:str = '.MI') -> pd.DataFrame:
    """
    Returns a time series DataFrame tracking Shares Held, 'Price','Net_Costs','Tot_Cost','Income',
    'Market_Value' and'Revenue' over time for a given Ticker, based on a transactions DataFrame.
    
    Parameters:
        transactions (pd.DataFrame): The full portfolio transactions DataFrame.
        ticker (str): The ticker symbol to track (without suffix like '.MI').
        market_suffix (str) : Suffix to append to ticker for market identification: '.MI' (default) 
        for Borsa Italiana
    
    Returns:
        pd.DataFrame: A time-indexed DataFrame with columns:
                      [Shares','Price','Net_Costs','Tot_Cost','Income','Market_Value','Revenue']
    """
    # 1. Filter Transactions for the Ticker
    mask = (transactions['Ticker'] == ticker)

    # 2. Download Price Series from Yahoo Finance
    prices = yf.download(
        tickers=ticker + market_suffix,
        start=transactions[mask].index.min(),
        end=pd.to_datetime("today").strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False
    ).loc[:, 'Adj Close']

    prices.name = 'Adj Close'

    # 3. Daily Shares Held
    shares = transactions[mask]['Shares'].reindex(prices.index, fill_value=0).cumsum()
    shares.name = 'Shares'

    # 4. Subscriptions (BUY)
    subs = (
        transactions[mask & (transactions['Type'] == 'BUY')]['Amount']
        .reindex(prices.index, fill_value=0)
        .cumsum()
    )
    subs.name = 'Subscriptions'

    # 5. Redemptions (SELL)
    redemps = (
        transactions[mask & (transactions['Type'] == 'SELL')]['Amount']
        .abs().reindex(prices.index, fill_value=0)
        .cumsum()
    )
    redemps.name = 'Redemptions'

    # 6. Dividends
    div = (
        transactions[mask & (transactions['Type'] == 'Dividend')]['Amount']
        .abs().reindex(prices.index, fill_value=0)
        .cumsum()
    )
    div.name = 'Dividends'

    # 7. Costs (Commissions + Taxes)
    costs = (
        transactions[mask][['Commissions', 'Taxes']]
        .sum(axis=1)
        .reindex(prices.index, fill_value=0)
        .cumsum()
    )
    costs.name = 'Costs'

    # Combine all series into one DataFrame
    tick_timeseries = pd.concat([shares, prices, subs, redemps, div, costs], axis=1)
    
    tick_timeseries.rename(columns={ticker + market_suffix:'Price'}, inplace=True)
    
    # Add 'Market_Value' and 'Revenue' columns
    tick_timeseries['Market_Value'] = tick_timeseries.Shares * tick_timeseries.Price
    
    tick_timeseries['Revenue'] = tick_timeseries[['Market_Value','Redemptions','Dividends']].sum(axis=1) -\
        tick_timeseries[['Subscriptions','Costs']].sum(axis=1)
    
    # Add 'Net_Costs', 'Tot_Cost', and 'Income' columns
    tick_timeseries['Net_Costs'] = tick_timeseries[['Subscriptions','Costs']].sum(axis=1) -\
        tick_timeseries[['Redemptions','Dividends']].sum(axis=1)
    
    tick_timeseries['Tot_Costs'] = tick_timeseries[['Subscriptions','Costs']].sum(axis=1)
    
    tick_timeseries['Income'] = tick_timeseries[['Market_Value','Redemptions','Dividends']].sum(axis=1) 
    
    # Drop unecessary columns
    tick_timeseries.drop(labels=['Subscriptions', 'Redemptions', 'Dividends', 'Costs'], axis=1, inplace=True)
    
    # re-order columns
    cols = ['Shares','Price','Net_Costs','Tot_Costs','Income','Market_Value','Revenue']
    tick_timeseries = tick_timeseries[cols]
    
    # Name the DataFrame with the ticker
    tick_timeseries.name = ticker

    return tick_timeseries


def build_portfolio_timeseries(transactions, get_ticker_timeseries):
    """
    Build the portfolio-level time series from individual asset time series.
    
    Parameters:
        transactions (pd.DataFrame): DataFrame containing transaction data with a 'Ticker' column.
        get_ticker_timeseries (function): Function to retrieve time series for a given ticker.
    
    Returns:
        pd.DataFrame: Combined portfolio time series with columns:
                      ['Net_Costs', 'Tot_Costs', 'Income', 'Market_Value', 'Revenue']
    """
    cols = ['Net_Costs', 'Tot_Costs', 'Income', 'Market_Value', 'Revenue']
    tickers = transactions['Ticker'].unique()

    # Step 1: Create a dictionary of time series per ticker
    assets = {}
    for tick in tickers:
        assets[tick] = get_ticker_timeseries(transactions=transactions, ticker=tick)

    # Step 2: Aggregate each metric across all tickers
    pf_list = []
    for col in cols:
        sub_list = [assets[tick][col] for tick in tickers]
        # ffill() fills missing values with day before data; 0s fill days earlier than Ticker 1st BUY
        sub_df = pd.concat(sub_list, join='outer', axis=1).ffill().fillna(0)
        pf_list.append(sub_df.sum(axis=1))

    # Step 3: Assemble the final portfolio DataFrame
    portfolio = pd.concat(pf_list, axis=1)
    portfolio.columns = cols

    return portfolio


# Default plot configuration
PLOT_CONFIG = {
    'figure.figsize': (12, 6),
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'lines.linewidth': 1.5,
    'lines.linestyle': '-',
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'grid.color': 'gray',
    'legend.fontsize': 10,
    'font.family': 'serif',  # Or 'sans-serif', etc.
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'axes.grid': True
}

def apply_plot_config():
    """Applies the default plot configuration to matplotlib."""
    plt.rcParams.update(PLOT_CONFIG)


def plot_timeseries(time_series_data, columns_to_plot=None,
                    title=None, ylabel=None, xlabel=None,
                    legend_loc='best', grid=None, column_styles=None,
                    figsize=None, **kwargs):
    """
    Plots one or more time series with consistent styling and optional
    figure size override.

    Parameters
    ----------
    time_series_data : pd.Series or pd.DataFrame
        The time series data to plot. If a DataFrame, specify `columns_to_plot`.
        The index should be a datetime object.
    columns_to_plot : list of str, optional
        List of column names to plot if `time_series_data` is a DataFrame.
        If None and `time_series_data` is a DataFrame, all columns will be plotted.
    title : str, optional
        The title of the plot.
    ylabel : str, optional
        The label for the y-axis.
    xlabel : str, optional
        The label for the x-axis.
    legend_loc : str, optional
        The location of the legend (e.g., 'best', 'upper left', 'lower right').
        Defaults to 'best'.
    grid : bool, optional
        Whether to display a grid on the plot. Defaults to True (or the value in PLOT_CONFIG if None).
    column_styles : dict, optional
        A dictionary where keys are column names (str) and values are dictionaries
        of keyword arguments to pass to `matplotlib.pyplot.plot()` for that
        specific column (e.g., {'Drawdown': {'color': 'red', 'linestyle': '-'}}).
        Defaults to None (uniform styling via `**kwargs`).
    figsize : tuple, optional
        Width and height of the figure in inches (e.g., (15, 8)).
        If None, the default from PLOT_CONFIG is used. Defaults to None.
    **kwargs :
        Additional keyword arguments passed to `matplotlib.pyplot.plot()`. These
        will be applied to all plotted lines unless overridden by `column_styles`.

    Returns
    -------
    None
        Displays the plot.
    """
    apply_plot_config()  # Apply the default configuration

    fig_size = figsize if figsize is not None else PLOT_CONFIG['figure.figsize']
    plt.figure(figsize=fig_size) # Ensure figure size is from config or customized

    if isinstance(time_series_data, pd.Series):
        plt.plot(time_series_data, label=time_series_data.name, **kwargs)
        plt.legend(loc=legend_loc, fontsize=PLOT_CONFIG['legend.fontsize'])
    elif isinstance(time_series_data, pd.DataFrame):
        if columns_to_plot is None:
            columns_to_plot = time_series_data.columns

        if column_styles is None:
            column_styles = {}

        for column in columns_to_plot:
            style = column_styles.get(column, {})
            all_styles = {**kwargs, **style}
            plt.plot(time_series_data.index, time_series_data[column], label=column, **all_styles)
        plt.legend(loc=legend_loc, fontsize=PLOT_CONFIG['legend.fontsize'])
    else:
        raise TypeError("time_series_data must be a pandas Series or DataFrame.")

    if title:
        plt.title(title, fontsize=PLOT_CONFIG['axes.titlesize'])
    if ylabel:
        plt.ylabel(ylabel, fontsize=PLOT_CONFIG['axes.labelsize'])
    if xlabel:
        plt.xlabel(xlabel, fontsize=PLOT_CONFIG['axes.labelsize'])
    if grid is None:
        plt.grid(PLOT_CONFIG['axes.grid'], linestyle=PLOT_CONFIG['grid.linestyle'],
                 alpha=PLOT_CONFIG['grid.alpha'], color=PLOT_CONFIG['grid.color'])
    elif grid is True:
        plt.grid(True, linestyle=PLOT_CONFIG['grid.linestyle'],
                 alpha=PLOT_CONFIG['grid.alpha'], color=PLOT_CONFIG['grid.color'])
    elif grid is False:
        plt.grid(False)

    plt.tight_layout()
    plt.show()


def drawdown_metrics(asset_values_ts: pd.Series) -> pd.DataFrame:
    """
    Calculates drawdown metrics for a given time series of portfolio or asset values.

    Identifies drawdown periods where the asset_values_ts value is below its cumulative max, 
    and returns the top 10 by maximum depth.

    Parameters:
    - asset_values_ts: pd.Series of daily portfolio or a single asset values (indexed by date)

    Returns:
    - pd.DataFrame with columns:
        - 'drawdown': Maximum drawdown (as negative decimal)
        - 'start': Start date of drawdown
        - 'valley': Date of drawdown minimum
        - 'end': Recovery date (NaT if not recovered)
    """
    # Calculate drawdown as percent decline from peak
    cummax = asset_values_ts.cummax()
    dd = asset_values_ts / cummax - 1
    underwater = dd < 0

    # Identify start and end of drawdown periods
    start_idxs, end_idxs = [], []
    for i in range(1, len(underwater)):
        if underwater.iloc[i] and not underwater.iloc[i - 1]:
            start_idxs.append(i)
        elif not underwater.iloc[i] and underwater.iloc[i - 1]:
            end_idxs.append(i)

    # Handle case where drawdown hasn't recovered yet
    if len(start_idxs) > len(end_idxs):
        end_idxs.append(len(asset_values_ts) - 1)

    # Calculate drawdown stats
    data = []
    for start, end in zip(start_idxs, end_idxs):
        valley_idx = dd.iloc[start:end + 1].idxmin()
        drawdown = dd.loc[valley_idx]
        data.append({
            'drawdown': drawdown,
            'start': asset_values_ts.index[start],
            'valley': valley_idx,
            'end': asset_values_ts.index[end] if not underwater.iloc[end] else pd.NaT
        })

    # Return top 10 deepest drawdowns
    df = pd.DataFrame(data)
    df = df.sort_values('drawdown').head(10).reset_index(drop=True)
    return df


def monthly_profit_loss(portfolio_revenue_ts, plot=True):
    """
    Calculate and optionally visualize monthly profit/loss from a portfolio revenue time series.

    Parameters:
    -----------
    portfolio_revenue_ts : pd.DataFrame
        Portfolio or single Asset DataFrame containing a 'Revenue' column with a datetime index.

    plot : bool, default=True
        If True, display a bar plot of monthly P/L.

    Returns:
    --------
    pd.Series
        Series of monthly profit/loss values.
    """
    revenue = portfolio_revenue_ts['Revenue']

    # Resample to get the revenue at the end of each month
    end_of_month_revenue = revenue.resample('M').last()

    # Get the first month's end revenue
    first_month_performance = end_of_month_revenue.iloc[0]

    # Calculate monthly performance
    monthly_pl = end_of_month_revenue.diff()
    monthly_pl.iloc[0] = first_month_performance  # Set first month manually

    if plot:
        # Color bars by sign
        colors = ['green' if val > 0 else 'red' for val in monthly_pl]

        # Create the plot
        plt.figure(figsize=(10, 6))
        ax = monthly_pl.plot(kind='bar', color=colors, edgecolor='black')

        ax.set_xticks(range(len(monthly_pl)))
        ax.set_xticklabels(
            [x.strftime('%b-%Y') for x in monthly_pl.index],
            rotation=90, ha='center'
        )

        plt.title('Monthly Profit (Loss)')
        plt.xlabel('Month')
        plt.ylabel('Revenue (EUR)')
        plt.grid(which="major", color='k', linestyle='-.', linewidth=0.3)
        plt.tight_layout()
        plt.show()

        print('Total Current Portfolio Revenue:', round(monthly_pl.sum(), 2))
        print('As of latest market data:', revenue.index[-1].strftime('%Y-%m-%d'))

    return monthly_pl


def calculate_fiscal_price(transactions, assets, ticker):
    """
    Calculate and assign the fiscal price for a single asset based on its transaction history.

    The fiscal price is computed by tracking the average cost (including commissions) of bought shares 
    within distinct holding periods (reset when holdings drop to zero).

    Parameters:
        transactions (pd.DataFrame): All asset transactions. Must include columns: 
                                     ['Ticker', 'Type', 'Shares', 'Amount', 'Commissions'].
        assets (dict): Dictionary of asset DataFrames keyed by ticker.
                       Each DataFrame must have a datetime index (daily frequency).
        ticker (str): The asset ticker for which the fiscal price is computed.

    Returns:
        pd.Series: The calculated fiscal price time series aligned to the asset's date index.
    """
    # Filter for ticker
    df_ticker = transactions[transactions['Ticker'] == ticker].copy()
    
    # Shares held over time
    shares_held = df_ticker['Shares'].cumsum()
    shares_held.name = 'shares_held'
    
    # Identify reset points where holdings drop to zero
    reset_points = (shares_held == 0)
    share_groups = reset_points.cumsum()
    share_groups.name = 'share_groups'

    # Cost of BUY transactions grouped cumulatively
    buy_cost_groups = (
        df_ticker[df_ticker['Type'] == 'BUY']
        .groupby(share_groups)[['Amount', 'Commissions']]
        .cumsum()
        .sum(axis=1)
    )

    # Shares bought per group
    shares_bought_groups = (
        df_ticker[df_ticker['Type'] == 'BUY']
        .groupby(share_groups)['Shares']
        .cumsum()
    )

    # Fiscal price per group
    fiscal_price = (buy_cost_groups / shares_bought_groups)
    fiscal_price.name = 'fiscal_price'

    # Combine and conditionally forward-fill fiscal price
    df_fiscal_price = pd.concat([shares_held, fiscal_price], axis=1)

    # Forward fill only where shares are held
    ffill_col = df_fiscal_price['fiscal_price'].where(df_fiscal_price['shares_held'] > 0).ffill()

    # Fill NaNs accordingly
    df_fiscal_price['fiscal_price'] = np.where(
        df_fiscal_price['fiscal_price'].isna(),
        np.where(df_fiscal_price['shares_held'] > 0, ffill_col, 0),
        df_fiscal_price['fiscal_price']
    )

    # Drop helper column
    df_fiscal_price.drop(columns=['shares_held'], inplace=True)

    # Align to asset's daily index
    df_fiscal_price = df_fiscal_price.reindex(assets[ticker].index, method='ffill')

    # Assign to asset DataFrame
    assets[ticker]['Fiscal_Price'] = df_fiscal_price['fiscal_price']

    return assets[ticker]['Fiscal_Price']





