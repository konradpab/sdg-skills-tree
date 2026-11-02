"""
Country code mapping utilities.
Maps common country name variants → ISO 3166-1 alpha-3 codes.
Used by every ingestion module to harmonize keys.
"""

import re

# Canonical mapping: variant names (lowercased) → ISO3
# This covers the most common mismatches across UN, World Bank, Ember, V-Dem, etc.
_NAME_TO_ISO3 = {
    # Standard names
    "afghanistan": "AFG", "albania": "ALB", "algeria": "DZA", "angola": "AGO",
    "argentina": "ARG", "armenia": "ARM", "australia": "AUS", "austria": "AUT",
    "azerbaijan": "AZE", "bahrain": "BHR", "bangladesh": "BGD", "belarus": "BLR",
    "belgium": "BEL", "benin": "BEN", "bolivia": "BOL", "bosnia and herzegovina": "BIH",
    "botswana": "BWA", "brazil": "BRA", "brunei": "BRN", "bulgaria": "BGR",
    "burkina faso": "BFA", "burundi": "BDI", "cambodia": "KHM", "cameroon": "CMR",
    "canada": "CAN", "chad": "TCD", "chile": "CHL", "china": "CHN",
    "colombia": "COL", "congo": "COG", "costa rica": "CRI", "croatia": "HRV",
    "cuba": "CUB", "cyprus": "CYP", "czech republic": "CZE", "czechia": "CZE",
    "denmark": "DNK", "dominican republic": "DOM", "ecuador": "ECU", "egypt": "EGY",
    "el salvador": "SLV", "estonia": "EST", "ethiopia": "ETH", "finland": "FIN",
    "france": "FRA", "gabon": "GAB", "georgia": "GEO", "germany": "DEU",
    "ghana": "GHA", "greece": "GRC", "guatemala": "GTM", "guinea": "GIN",
    "haiti": "HTI", "honduras": "HND", "hungary": "HUN", "iceland": "ISL",
    "india": "IND", "indonesia": "IDN", "iran": "IRN", "iraq": "IRQ",
    "ireland": "IRL", "israel": "ISR", "italy": "ITA", "ivory coast": "CIV",
    "jamaica": "JAM", "japan": "JPN", "jordan": "JOR", "kazakhstan": "KAZ",
    "kenya": "KEN", "kuwait": "KWT", "kyrgyzstan": "KGZ", "laos": "LAO",
    "latvia": "LVA", "lebanon": "LBN", "libya": "LBY", "lithuania": "LTU",
    "luxembourg": "LUX", "madagascar": "MDG", "malawi": "MWI", "malaysia": "MYS",
    "mali": "MLI", "malta": "MLT", "mauritania": "MRT", "mauritius": "MUS",
    "mexico": "MEX", "moldova": "MDA", "mongolia": "MNG", "montenegro": "MNE",
    "morocco": "MAR", "mozambique": "MOZ", "myanmar": "MMR", "namibia": "NAM",
    "nepal": "NPL", "netherlands": "NLD", "new zealand": "NZL", "nicaragua": "NIC",
    "niger": "NER", "nigeria": "NGA", "north macedonia": "MKD", "norway": "NOR",
    "oman": "OMN", "pakistan": "PAK", "panama": "PAN", "paraguay": "PRY",
    "peru": "PER", "philippines": "PHL", "poland": "POL", "portugal": "PRT",
    "qatar": "QAT", "romania": "ROU", "russia": "RUS", "rwanda": "RWA",
    "saudi arabia": "SAU", "senegal": "SEN", "serbia": "SRB", "sierra leone": "SLE",
    "singapore": "SGP", "slovakia": "SVK", "slovenia": "SVN", "south africa": "ZAF",
    "south korea": "KOR", "spain": "ESP", "sri lanka": "LKA", "sudan": "SDN",
    "sweden": "SWE", "switzerland": "CHE", "syria": "SYR", "taiwan": "TWN",
    "tajikistan": "TJK", "tanzania": "TZA", "thailand": "THA", "togo": "TGO",
    "trinidad and tobago": "TTO", "tunisia": "TUN", "turkey": "TUR",
    "turkmenistan": "TKM", "uganda": "UGA", "ukraine": "UKR",
    "united arab emirates": "ARE", "united kingdom": "GBR", "united states": "USA",
    "uruguay": "URY", "uzbekistan": "UZB", "venezuela": "VEN", "vietnam": "VNM",
    "yemen": "YEM", "zambia": "ZMB", "zimbabwe": "ZWE",
    # Common variant spellings
    "iran, islamic rep.": "IRN", "iran (islamic republic of)": "IRN",
    "korea, rep.": "KOR", "korea, republic of": "KOR", "republic of korea": "KOR",
    "korea (republic of)": "KOR",
    "russian federation": "RUS", "russia": "RUS",
    "türkiye": "TUR", "turkiye": "TUR",
    "viet nam": "VNM",
    "lao pdr": "LAO", "lao people's democratic republic": "LAO",
    "congo, dem. rep.": "COD", "democratic republic of the congo": "COD",
    "dr congo": "COD", "congo, democratic republic of the": "COD",
    "congo, rep.": "COG", "republic of the congo": "COG",
    "côte d'ivoire": "CIV", "cote d'ivoire": "CIV",
    "united states of america": "USA",
    "united kingdom of great britain and northern ireland": "GBR",
    "bolivia (plurinational state of)": "BOL",
    "venezuela (bolivarian republic of)": "VEN",
    "tanzania, united republic of": "TZA", "united republic of tanzania": "TZA",
    "syrian arab republic": "SYR",
    "north korea": "PRK", "korea, dem. people's rep.": "PRK",
    "taiwan, china": "TWN", "chinese taipei": "TWN",
    "hong kong": "HKG", "hong kong, china": "HKG",
    "the netherlands": "NLD",
    "bosnia & herzegovina": "BIH",
    "myanmar (burma)": "MMR", "burma": "MMR",
    "eswatini": "SWZ", "swaziland": "SWZ",
    "palestine": "PSE", "state of palestine": "PSE",
    "timor-leste": "TLS", "east timor": "TLS",
    "cabo verde": "CPV", "cape verde": "CPV",
}

# ISO2 → ISO3 mapping (World Bank API returns ISO2)
_ISO2_TO_ISO3 = {
    "AF": "AFG", "AL": "ALB", "DZ": "DZA", "AO": "AGO", "AR": "ARG",
    "AM": "ARM", "AU": "AUS", "AT": "AUT", "AZ": "AZE", "BH": "BHR",
    "BD": "BGD", "BY": "BLR", "BE": "BEL", "BJ": "BEN", "BO": "BOL",
    "BA": "BIH", "BW": "BWA", "BR": "BRA", "BN": "BRN", "BG": "BGR",
    "BF": "BFA", "BI": "BDI", "KH": "KHM", "CM": "CMR", "CA": "CAN",
    "TD": "TCD", "CL": "CHL", "CN": "CHN", "CO": "COL", "CG": "COG",
    "CR": "CRI", "HR": "HRV", "CU": "CUB", "CY": "CYP", "CZ": "CZE",
    "DK": "DNK", "DO": "DOM", "EC": "ECU", "EG": "EGY", "SV": "SLV",
    "EE": "EST", "ET": "ETH", "FI": "FIN", "FR": "FRA", "GA": "GAB",
    "GE": "GEO", "DE": "DEU", "GH": "GHA", "GR": "GRC", "GT": "GTM",
    "GN": "GIN", "HT": "HTI", "HN": "HND", "HU": "HUN", "IS": "ISL",
    "IN": "IND", "ID": "IDN", "IR": "IRN", "IQ": "IRQ", "IE": "IRL",
    "IL": "ISR", "IT": "ITA", "JM": "JAM", "JP": "JPN", "JO": "JOR",
    "KZ": "KAZ", "KE": "KEN", "KW": "KWT", "KG": "KGZ", "LA": "LAO",
    "LV": "LVA", "LB": "LBN", "LY": "LBY", "LT": "LTU", "LU": "LUX",
    "MG": "MDG", "MW": "MWI", "MY": "MYS", "ML": "MLI", "MT": "MLT",
    "MR": "MRT", "MU": "MUS", "MX": "MEX", "MD": "MDA", "MN": "MNG",
    "ME": "MNE", "MA": "MAR", "MZ": "MOZ", "MM": "MMR", "NA": "NAM",
    "NP": "NPL", "NL": "NLD", "NZ": "NZL", "NI": "NIC", "NE": "NER",
    "NG": "NGA", "MK": "MKD", "NO": "NOR", "OM": "OMN", "PK": "PAK",
    "PA": "PAN", "PY": "PRY", "PE": "PER", "PH": "PHL", "PL": "POL",
    "PT": "PRT", "QA": "QAT", "RO": "ROU", "RU": "RUS", "RW": "RWA",
    "SA": "SAU", "SN": "SEN", "RS": "SRB", "SL": "SLE", "SG": "SGP",
    "SK": "SVK", "SI": "SVN", "ZA": "ZAF", "KR": "KOR", "ES": "ESP",
    "LK": "LKA", "SD": "SDN", "SE": "SWE", "CH": "CHE", "SY": "SYR",
    "TW": "TWN", "TJ": "TJK", "TZ": "TZA", "TH": "THA", "TG": "TGO",
    "TT": "TTO", "TN": "TUN", "TR": "TUR", "TM": "TKM", "UG": "UGA",
    "UA": "UKR", "AE": "ARE", "GB": "GBR", "US": "USA", "UY": "URY",
    "UZ": "UZB", "VE": "VEN", "VN": "VNM", "YE": "YEM", "ZM": "ZMB",
    "ZW": "ZWE", "CD": "COD", "CI": "CIV", "HK": "HKG", "PS": "PSE",
    "TL": "TLS", "CV": "CPV", "SZ": "SWZ", "SS": "SSD", "KP": "PRK",
}


def name_to_iso3(name: str) -> str | None:
    """Convert a country name (any common variant) to ISO3. Returns None if unknown."""
    if not name or not isinstance(name, str):
        return None
    key = name.strip().lower()
    # Direct lookup
    if key in _NAME_TO_ISO3:
        return _NAME_TO_ISO3[key]
    # Try stripping parenthetical qualifiers
    key_clean = re.sub(r"\s*\(.*?\)\s*", "", key).strip()
    if key_clean in _NAME_TO_ISO3:
        return _NAME_TO_ISO3[key_clean]
    return None


def iso2_to_iso3(code: str) -> str | None:
    """Convert ISO 3166-1 alpha-2 to alpha-3."""
    if not code or not isinstance(code, str):
        return None
    return _ISO2_TO_ISO3.get(code.strip().upper())


def is_valid_iso3(code: str) -> bool:
    """Check if a string is a known ISO3 code."""
    if not code or not isinstance(code, str):
        return False
    return code.strip().upper() in set(_NAME_TO_ISO3.values()) | set(_ISO2_TO_ISO3.values())
