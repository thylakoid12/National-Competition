"""Read source workbooks once; expose detached historical prefixes."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from common.data_io import parse_time_minutes, load_data as load_q1_source
from q2_config import ATTACHMENTS, DT_HOURS, state_labels

@dataclass(frozen=True)
class CanonicalData:
    dates: tuple
    values: np.ndarray
    price: np.ndarray
    interval_start: tuple
    interval_end: tuple

    def history(self, day):
        out = self.values[:day].copy()
        out.flags.writeable = False
        return out

def load_data():
    q1 = load_q1_source(ATTACHMENTS / '附件1.xlsx')
    expected = pd.date_range('2025-01-01', '2025-12-31')
    targets = []
    for name in ['小区负载', '光伏发电实际功率']:
        raw = pd.read_excel(ATTACHMENTS / '附件2.xlsx', sheet_name=name)
        assert raw.shape == (365, 145), (name, raw.shape)
        if not np.array_equal([parse_time_minutes(v) for v in raw.columns[1:]], np.arange(10, 1441, 10)):
            raise ValueError('Invalid right endpoint labels')
        if not pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 0])).equals(expected):
            raise ValueError('Expected 365 ordered unique 2025 dates')
        power = raw.iloc[:, 1:].to_numpy(dtype=float)
        if not np.isfinite(power).all() or (power < 0).any():
            raise ValueError('Missing, negative or nonfinite source power')
        targets.append(power * DT_HOURS)
    values = np.stack(targets, axis=-1)
    values.flags.writeable = False
    price = q1.price.to_numpy(copy=True)
    assert (price > 0).all()
    price.flags.writeable = False
    assert len(expected[expected >= '2025-02-01']) == 334
    return CanonicalData(tuple(d.date().isoformat() for d in expected), values, price,
                         tuple(state_labels()[:-1]), tuple(state_labels()[1:]))
