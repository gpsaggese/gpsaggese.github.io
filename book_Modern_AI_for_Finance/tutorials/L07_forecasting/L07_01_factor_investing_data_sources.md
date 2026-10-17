# Data Sources for Computing Risk Premia

- This file lists where to get data to compute risk premia, for the notebook
  `L07_01_factor_investing.ipynb`
  - The notebook itself runs on simulated data, so the true premium is known
  - The sources below let you repeat the same procedures on real data
- Availability, access, and licence terms of each source were not checked when this
  file was written
  - Confirm them before building on a source

# Realized and Expected Premia

- A _realized_ premium is the average of past returns
  - Most sources below give realized premia
- An _expected_ (ex-ante) premium is a forward-looking estimate
  - It comes from models or surveys, see `## Expected (Ex-Ante) Premia`

## Ready-Made Factor Premia

- **Kenneth French Data Library**
  - Factors: market, size (`SMB`), value (`HML`), profitability (`RMW`), investment
    (`CMA`), momentum
  - Sorted portfolios, for the US and several regions
  - Free, load with `pandas_datareader.famafrench`
- **AQR Data Library**
  - Factors: quality (`QMJ`), betting against beta (`BAB`), time-series momentum
  - Value and momentum across asset classes
  - Free spreadsheets
- **Global Factor Data (Jensen, Kelly, Pedersen)**
  - Many characteristics in many countries
  - Factor returns are free, stock-level data needs WRDS
- **Open Source Asset Pricing (Chen-Zimmermann)**
  - A large set of published predictors, with signals and long-short returns
  - Free, check whether the stock-level file includes returns
- **Other factor sets**
  - Hou-Xue-Zhang q-factors
  - Stambaugh-Yuan mispricing factors
  - Pastor-Stambaugh liquidity factor

## Raw Data to Compute Premia Yourself

- **CRSP and Compustat through WRDS**
  - Returns, market cap, and book equity for all stocks
  - The standard input for Fama-MacBeth
  - Needs a licence, and the data cannot be redistributed
- **SEC EDGAR financial statements**
  - Free, via the XBRL company-facts API or the Financial Statement Data Sets
  - Book equity must be built by hand
  - Filing dates must be handled to avoid look-ahead bias
- **Prices**
  - Free but weak: `yfinance` has limited history and survivorship bias
  - Paid and cheaper than WRDS: Sharadar, Tiingo, Polygon, Financial Modeling Prep
- **Risk-free rate**
  - Treasury bill rate from FRED (`DTB3`)
  - Or the `RF` column of the French files
- **Commercial vendors**
  - Bloomberg, FactSet, LSEG, S&P Capital IQ, MSCI Barra

## Expected (Ex-Ante) Premia

- **Equity risk premium estimates**
  - Damodaran implied premium, free and updated regularly
  - Fernandez survey of premium estimates
  - Graham-Harvey CFO survey
- **Predictor data for the equity premium**
  - Robert Shiller data: CAPE and a long return history
  - Welch-Goyal dataset

## Premia in Other Asset Classes

- **Term premium**
  - NY Fed ACM model and Kim-Wright model, both on FRED
- **Credit premium**
  - ICE BofA option-adjusted spreads on FRED
  - Moody's BAA-AAA spread
- **FX and commodities**
  - Fed H.10 exchange rates for carry
  - Commodity futures from a data vendor
- **Volatility premium**
  - CBOE VIX against realized volatility

# Mapping to the Notebook

- **Part 3 (sorted portfolios)**
  - The French library applies directly: sorted portfolios and factor series
- **Part 2 (Fama-MacBeth)**
  - Needs stock-level data
  - Options: CRSP and Compustat, the Open Source Asset Pricing signals, or EDGAR plus
    prices
- **Cells 4.1 and 4.2**
  - The French factor series give a real sample to test for significance

# Cautions for Any Source

- A realized premium over a short sample is a noisy estimate of the expected premium
  - Cell 2.2 shows this on simulated data
- Published factors suffer from publication bias
  - Premia often shrink after publication
- Use only data known at the time (point-in-time data)
  - Otherwise look-ahead bias enters the sample
- Free price sources often drop delisted stocks
  - This biases returns upward