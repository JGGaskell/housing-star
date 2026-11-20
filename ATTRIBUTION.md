# Data attribution

`housing_star` downloads and bundles public-sector data. Both sources are published under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## HM Land Registry Price Paid Data

Contains HM Land Registry data © Crown copyright and database right 2026. This data is licensed
under the Open Government Licence v3.0. Transactions are fetched on demand from the HM Land Registry
SPARQL endpoint and cached locally; they are **not** redistributed with this package.

## Ordnance Survey Code-Point Open

`src/housing_star/data/_resources/centroids.parquet` contains district and sector centroids
(mean easting/northing of unit postcodes) derived from Code-Point Open. Postcodes without
coordinates (positional quality 90, recorded as easting and northing 0) are left out.

- **Source:** Ordnance Survey Code-Point Open, CSV download for Great Britain, from the
  [OS Data Hub](https://osdatahub.os.uk/downloads/open/CodePointOpen).
- **Release used:** version 2025.4.0, copyright date 20 October 2025 (Royal Mail update
  17 October 2025), 1,745,849 unit postcodes.
- **Licence:** OS OpenData, under the Open Government Licence v3.0.

Contains OS data © Crown copyright and database right 2025.
Contains Royal Mail data © Royal Mail copyright and database right 2025.
Contains National Statistics data © Crown copyright and database right 2025.

### Regenerating the table

The download is a zip of headerless CSV files, one per postcode area, in `Data/CSV/`. Their column
names are in `Doc/Code-Point_Open_Column_Headers.csv`. Stack the three columns the build needs into
one Parquet file, then run the build script:

```python
import polars as pl

(
    pl.scan_csv("codepo_gb/Data/CSV/*.csv", has_header=False)
    .select(
        pl.nth(0).alias("Postcode"),
        pl.nth(2).alias("Eastings"),
        pl.nth(3).alias("Northings"),
    )
    .sink_parquet("codepo.parquet")
)
```

```bash
python scripts/build_centroids.py codepo.parquet
```

With release 2025.4.0 this reproduces the bundled file exactly. A later release adds, retires and
moves postcodes, so it gives slightly different centroids; update the release and the copyright
year above when you regenerate.
