# Notes

## Predict The Stock Market With Machine Learning And Python

Notes from https://www.youtube.com/watch?v=1O_BenficgE

- Many real world considerations to take into account when predicting stock prices

### Downloading the data from Yahoo Finance:

```python
import yfinance as yf
sp500 = yf.download("^GSPC)
sp500=sp500.history(period="max")
```

Result is a Pandas Dataframe:

![Pandas Dataframe](images/pandas-dataframe.png)

Each row a trading day - non-trading days are omitted
The columns are:

- Open: The price of the stock at the beginning of the trading day (it need not be the closing price of the previous trading day)
- High: The highest price of the stock on that trading day
- Low: The lowest price of the stock on that trading day
- Close: The price of the stock at closing time
- Volume: How many stocks were traded
- Dividends: Dividends paid by the company
- Stock Splits: If the company splits its stocks, this is the ratio

The index of the data can be found with:

```python
sp500.index
```

This shows that the data is indexed by date

### Cleaning and visualising the data

The first step is to plot the closing price of the stock:

```python
sp500.plot(y="Close", use_index=True)
```

![Closing price of the stock](images/closing-price.png)

We can now remove the columns we are ignoring for the moment for simplicity as we are looking at the whole index not individual stocks:

```python
del sp500["Dividends"]
del sp500["Stock Splits"]
```

## Defining the target of our ML prediction

We want to predict whether the price will go up or down tomorrow. We want to be able to predict this with a high degree of accuracy so our model needs
to say the market will go up on days where it does go up. We can then buy stocks knowing the market is on the up.

First we create a new column called `Tomorrow` which will be the price for market close tomorrow. We can initialise this with the shift operation
which will take the `Close` of the day before and set it as the `Tomorrow` of the day before:

```python
sp500["Tomorrow"] = sp500["Close"].shift(-1)
```

Next we create a Target which is whether todays close is higher than tomorrows close:

```python
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
```

Our data frame now looks like this:

![Dataframe with target](images/dataframe-with-target.png)

resume on 7m12
