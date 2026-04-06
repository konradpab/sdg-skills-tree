# Data Source Reference

Endpoint-specific details for every dataset in the Minimum Viable Stack (MVS).
Use this when a download fails or an API changes its schema.

---

## 1. UNSD SDG Global Database

**What it provides:** Official UN SDG indicator values at country-year level.

**Endpoint:** `https://unstats.un.org/sdgapi/v1/sdg/Series/Data`

**How the pipeline uses it:**
The pipeline fetches 6 series by code (e.g., `EG_ELC_ACCS` for electricity access).
Each request returns paginated JSON with fields `geoAreaCode`, `geoAreaName`,
`timePeriodStart`, and `value`.

**If the API changes:**
1. Open the Swagger UI: https://unstats.un.org/sdgapi/swagger/
2. Navigate to `GET /v1/sdg/Series/Data`
3. Try a test query with `seriesCode=EG_ELC_ACCS&pageSize=5`
4. Compare the response structure to `src/ingest/real_data.py:ingest_unsd_sdg()`
5. Update the field extraction logic if field names changed

**Series codes used:**

| Our name       | Series code      | SDG indicator |
|----------------|------------------|---------------|
| elec_access    | EG_ELC_ACCS      | 7.1.1         |
| renewables     | EG_FEC_RNEW      | 7.2.1         |
| energy_int     | EG_EGY_PRIM      | 7.3.1         |
| manuf_va       | NV_IND_MANF      | 9.2.1         |
| rd_expend      | GB_XPD_RSDV      | 9.5.1         |
| co2_pc         | EN_ATM_CO2E_PC   | 13.2.2        |

To find updated codes, search https://unstats.un.org/sdgs/dataportal.

---

## 2. Ember Yearly Electricity Data

**What it provides:** Electricity generation by source, demand, emissions, capacity — country-year.

**Endpoint:** Direct CSV download from
`https://ember-energy.org/app/uploads/2024/09/yearly_full_release_long_format.csv`

**Format:** Long-format CSV with columns like `Area`, `Year`, `Variable`, `Unit`, `Value`.

**If the URL returns 404:**
1. Go to https://ember-energy.org/data/yearly-electricity-data/
2. Click the download button
3. Note the new URL, update `config.yaml` → `endpoints.ember.yearly_url`
4. If the CSV format changed (different column names), update `ingest_ember()` accordingly

**Variables extracted:**

| Ember variable name | Our column           |
|---------------------|----------------------|
| Coal (% of generation) | coal_share_pct    |
| Gas (% of generation)  | gas_share_pct     |
| Solar (% of generation)| solar_share_pct   |
| Wind (% of generation) | wind_share_pct    |
| Hydro (% of generation)| hydro_share_pct   |
| Nuclear (% of generation)| nuclear_share_pct |
| Total generation (TWh) | generation_twh     |

---

## 3. PRIMAP-hist Emissions

**What it provides:** Historical greenhouse gas emissions by country, gas, and sector.

**Endpoint:** Zenodo CSV.
Current: `https://zenodo.org/records/10006301/files/Guetschow_et_al_2023-PRIMAP-hist_v2.5.1_final_no_extrap_15-Oct-2023.csv`

**Format:** Wide CSV. Country column is `country (ISO3)`. Year columns are `1850`, `1851`, ..., `2022`.
Entity column distinguishes CO2, CH4, N2O, KYOTOGHG, etc.

**If a new version is released:**
1. Go to https://zenodo.org/search?q=primap-hist
2. Find the latest record
3. Copy the direct CSV download URL
4. Update `config.yaml` → `endpoints.primap.download_url`

**Entities extracted:**

| Entity filter | Our column        |
|---------------|-------------------|
| CO2           | co2_mt            |
| KYOTOGHG      | ghg_mt_co2eq      |

---

## 4. World Bank WGI (Governance)

**What it provides:** Six governance dimensions scored on a standard normal scale (~-2.5 to +2.5).

**Endpoint:** World Bank API v2:
`https://api.worldbank.org/v2/country/all/indicator/{code}?date=1996:2023&format=json&per_page=20000`

**Response:** JSON array `[metadata, [records]]`. Each record has `country.id` (ISO2),
`date` (year), `value` (float or null).

**Indicators:**

| Our name              | WB code  |
|-----------------------|----------|
| gov_effectiveness     | GE.EST   |
| regulatory_quality    | RQ.EST   |
| rule_of_law           | RL.EST   |
| control_corruption    | CC.EST   |
| political_stability   | PV.EST   |
| voice_accountability  | VA.EST   |

**If values are null:** WGI is updated biennially (every other year). Odd years
may have null values for some countries. The normalization layer interpolates
gaps of up to 2 years.

---

## 5. World Bank WDI (Macro)

**What it provides:** GDP, population, urbanization, energy, education, industry shares.

**Endpoint:** Same API as WGI:
`https://api.worldbank.org/v2/country/all/indicator/{code}?date=1996:2023&format=json&per_page=20000`

**Indicators:**

| Our name           | WB code              | Description                  |
|--------------------|----------------------|------------------------------|
| gdp_pc_ppp         | NY.GDP.PCAP.PP.KD    | GDP per capita, PPP (const $)|
| gdp_growth         | NY.GDP.MKTP.KD.ZG    | GDP growth (annual %)        |
| industry_gdp_pct   | NV.IND.TOTL.ZS       | Industry (% of GDP)          |
| energy_use_pc      | EG.USE.PCAP.KG.OE    | Energy use per capita (kg OE)|
| edu_expenditure     | SE.XPD.TOTL.GD.ZS    | Education exp. (% GDP)       |
| population         | SP.POP.TOTL           | Total population              |
| urban_pct          | SP.URB.TOTL.IN.ZS    | Urban population (%)          |

**Rate limits:** The WB API occasionally throttles requests. The pipeline adds a
0.5s delay between indicator requests. If you get 429 errors, increase the
sleep in `real_data.py:ingest_wdi()`.

---

## 6. V-Dem (Democracy)

**What it provides:** Democracy indices based on expert coding.

**Endpoint:** Manual download from https://v-dem.net/data/the-v-dem-dataset/

**Why manual:** V-Dem requires accepting a license agreement via their web form.
There is no programmatic API.

**Steps:**
1. Register at https://v-dem.net/data/the-v-dem-dataset/
2. Download "Country-Year: V-Dem Core" in CSV format
3. Place the CSV in `data_raw/vdem/`
4. The pipeline auto-detects any CSV in that directory

**Variables extracted:**

| V-Dem variable     | Our name                 |
|--------------------|--------------------------|
| v2x_polyarchy      | democracy_index          |
| v2x_libdem         | liberal_democracy        |
| v2x_partipdem      | participatory_democracy  |

**Country key:** V-Dem uses ISO3 in the `country_text_id` column, so no mapping is needed.

---

## 7. SWIID (Inequality)

**What it provides:** Standardized Gini coefficients for disposable and market income.

**Endpoint:** `https://fsolt.org/swiid/swiid9_6.zip`

**Format:** Zip containing `swiid_summary.csv` with columns:
`country`, `year`, `gini_disp`, `gini_disp_se`, `gini_mkt`, `gini_mkt_se`.

**If the version changes:** Check https://fsolt.org/swiid/ for the current version
number and update the URL in `config.yaml`.

---

## 8. Carbon Pricing Dashboard

**What it provides:** Carbon tax and ETS price levels plus coverage percentages.

**Endpoint:** `https://carbonpricingdashboard.worldbank.org/sites/default/files/data-latest.xlsx`

**Format:** Excel file. Column names may vary between releases.
The pipeline searches for columns containing "country"/"jurisdiction",
"year", "price", and "coverage".

**If the Excel format changes:** Download manually and inspect the column names.
Update `ingest_carbon_pricing()` to match the new column names.

---

## 9. UCDP GED (Conflict)

**What it provides:** Georeferenced conflict events with fatality estimates.

**Endpoint:** `https://ucdp.uu.se/downloads/ged/ged241-csv.zip`
(version number changes with each release)

**Format:** Zip containing a large CSV with one row per conflict event.
The pipeline aggregates to country-year: count of events + sum of fatalities.

**If the version changes:**
1. Go to https://ucdp.uu.se/downloads/
2. Find "UCDP Georeferenced Event Dataset (GED)"
3. Copy the new zip URL
4. Update `config.yaml` → `endpoints.ucdp.ged_url`

**Country mapping:** UCDP uses country names (not ISO codes), so the pipeline
runs them through the `name_to_iso3()` mapper. If a new country name variant
appears, add it to `src/ingest/country_codes.py`.

---

## Extension Datasets (not in MVS)

These are configured in `config.yaml` but disabled by default. To enable:
set the toggle to `true` and implement the corresponding `ingest_*()` function.

| Dataset | Access | Notes |
|---------|--------|-------|
| UNESCO UIS | API (free) | https://databrowser.uis.unesco.org/resources |
| WVS | Manual download (license) | https://www.worldvaluessurvey.org/ |
| GDELT | API (free, very large) | https://www.gdeltproject.org/ |
| WID | API / download | https://wid.world/ |
| OEC / Atlas | API | https://oec.world/ |
