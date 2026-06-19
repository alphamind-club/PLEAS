"""ICER Calculator Template — Adolescent Sleep Interventions CEA

Public API
----------
calculate_icer(delta_cost, delta_effect_hours, adherence_factor) -> float
sensitivity_tornado(base_cost, base_effect, adherence,
                    cost_range, effect_range, adherence_range) -> list[dict]

Formulas
--------
ICER = (Cost_A - Cost_B) / (Effect_A_hrs - Effect_B_hrs) * adherence_factor
adjusted_effect = raw_effect_hrs * adherence_factor
QALY_gain = (delta_sleep_min / 60) * utility_weight * time_horizon_years
"""
def calculate_icer(delta_cost, delta_effect_hours, adherence_factor):
    '''
    Calculate the Incremental Cost-Effectiveness Ratio (ICER).

    Parameters
    ----------
    delta_cost : float
        Additional cost of the intervention vs. comparator (USD per adolescent).
    delta_effect_hours : float
        Additional effectiveness vs. comparator (hours of sleep gained per night).
    adherence_factor : float
        Real-world adherence rate (0.0 to 1.0). Use 1.0 for perfect adherence.

    Returns
    -------
    float
        ICER in USD per hour of sleep gained per night.
        Returns inf if delta_effect_hours is zero or negative.

    Examples
    --------
    >>> calculate_icer(delta_cost=150.0, delta_effect_hours=0.40, adherence_factor=0.65)
    576.92...
    '''
    adjusted_effect = delta_effect_hours * adherence_factor
    if adjusted_effect <= 0:
        return float('inf')
    return round(delta_cost / adjusted_effect * adherence_factor, 4)

def sensitivity_tornado(base_cost, base_effect, adherence,
                        cost_range, effect_range, adherence_range):
    '''
    Generate tornado-plot sensitivity data by varying each key parameter
    across its plausible low-high range while holding others at base values.

    Parameters
    ----------
    base_cost : float
        Base delta_cost (USD).
    base_effect : float
        Base delta_effect_hours (hrs/night).
    adherence : float
        Base adherence_factor (0-1).
    cost_range : tuple
        (low, high) for delta_cost sensitivity.
    effect_range : tuple
        (low, high) for delta_effect_hours sensitivity.
    adherence_range : tuple
        (low, high) for adherence_factor sensitivity.

    Returns
    -------
    list[dict]
        Each dict has keys: parameter, low_value, high_value, base_value, swing.
        Sorted descending by swing (most to least influential parameter).

    Examples
    --------
    >>> sensitivity_tornado(
    ...     base_cost=150.0, base_effect=0.40, adherence=0.65,
    ...     cost_range=(100.0, 200.0), effect_range=(0.30, 0.55),
    ...     adherence_range=(0.50, 0.80)
    ... )   # doctest: +ELLIPSIS
    [{'parameter': ..., 'low_value': ..., 'high_value': ..., 'base_value': ..., 'swing': ...}, ...]
    '''
    from icer_calculator_template import calculate_icer
    icer_base = calculate_icer(base_cost, base_effect, adherence)

    scenarios = [
        {'parameter': 'delta_cost',         'low': cost_range[0],       'high': cost_range[1]},
        {'parameter': 'delta_effect_hours', 'low': effect_range[0],      'high': effect_range[1]},
        {'parameter': 'adherence_factor',   'low': adherence_range[0],   'high': adherence_range[1]},
    ]

    results = []
    for s in scenarios:
        icer_low  = calculate_icer(s['low'],  base_effect, adherence)
        icer_high = calculate_icer(s['high'], base_effect, adherence)
        results.append({
            'parameter': s['parameter'],
            'low_value':  icer_low,
            'high_value': icer_high,
            'base_value': icer_base,
            'swing': round(abs(icer_high - icer_low), 4),
        })

    results.sort(key=lambda x: x['swing'], reverse=True)
    return results

if __name__ == '__main__':
    icer = calculate_icer(delta_cost=150.0, delta_effect_hours=0.40, adherence_factor=0.65)
    print(f'ICER = ${icer:.2f} per hour gained')
    tornado = sensitivity_tornado(
        base_cost=150.0, base_effect=0.40, adherence=0.65,
        cost_range=(100.0, 200.0), effect_range=(0.30, 0.55),
        adherence_range=(0.50, 0.80)
    )
    for row in tornado:
        print(row)