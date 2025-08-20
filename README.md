# Horcrux

A Python package for financial data feature extraction, focusing on OHLCV (Open, High, Low, Close, Volume) data processing.

## Features

- **OHLCV Data Processing**: Load and process OHLCV data from parquet files
- **Base Feature Framework**: Extensible base class for creating custom financial features
- **Efficient Data Loading**: Cached data loading for improved performance

## Installation

### Development Installation

```bash
pip install -e .
```

### From PyPI (when published)

```bash
pip install horcrux
```

## Quick Start

```python
from horcrux import OHLCV

# Create an OHLCV feature extractor
ohlcv_feature = OHLCV()

# Get OHLCV data for specific pairs and time range
data = ohlcv_feature.get_feature(
    start="2023-01-01",
    end="2023-07-15", 
    pairs=["AAVE_BTC", "ADA_BTC", "ETH_BTC"]
)

print(data.head())
```

## Configuration

The package expects a configuration file at `~/.config/horcrux/horcrux_config.toml` with:

```toml
ohlcv_path = "/path/to/your/ohlcv_data.pq"
```

## Requirements

- Python 3.8+
- pandas
- pyarrow
- toml
- pydantic

## License

MIT License 