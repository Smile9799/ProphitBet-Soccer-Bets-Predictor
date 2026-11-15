import re
from datetime import date
from typing import Optional

import pandas as pd


def add_years_to_dates(
        date_items: list[str],
        current_date: Optional[date] = None,
) -> list[str]:
    """Add years to fixture labels ordered from newest to oldest."""

    today = current_date or date.today()
    parsed_dates = []
    for date_item in date_items:
        normalized_date = re.sub(
            r'(?<=\d)(?:st|nd|rd|th)\b',
            '',
            str(date_item).strip(),
            flags=re.IGNORECASE,
        )
        parsed_date = pd.to_datetime(normalized_date, errors='coerce')
        parsed_dates.append((normalized_date, parsed_date))

    first_valid_date = next(
        (parsed_date for _, parsed_date in parsed_dates if not pd.isna(parsed_date)),
        None,
    )
    if first_valid_date is None:
        return [normalized_date for normalized_date, _ in parsed_dates]

    first_month_day = (first_valid_date.month, first_valid_date.day)
    current_month_day = (today.month, today.day)
    inferred_year = today.year if first_month_day >= current_month_day else today.year + 1
    previous_date: Optional[date] = None
    dated_labels = []

    for normalized_date, parsed_date in parsed_dates:
        if pd.isna(parsed_date):
            dated_labels.append(normalized_date)
            continue

        item_date = date(inferred_year, parsed_date.month, parsed_date.day)
        while previous_date is not None and item_date > previous_date:
            inferred_year -= 1
            item_date = date(inferred_year, parsed_date.month, parsed_date.day)

        dated_labels.append(f'{normalized_date}, {inferred_year:04d}')
        previous_date = item_date

    return dated_labels
