Achievement Status Calc = 
VAR _type = RELATED(dim_kpi_master[KPI Type])
VAR _direction = RELATED(dim_kpi_master[Direction])
VAR _value = dim_quality[KPI_Value]
VAR _target = dim_quality[Target]
RETURN
SWITCH(
    TRUE(),

    -- Binary: whole number 1/0, Met = 1
    _type = "Binary",
        SWITCH(
            TRUE(),
            _value = 1, "Achieved",
            _value = 0, "Not Achieved",
            "Invalid Binary Value"
        ),

    -- Count: whole number comparison, Direction-aware, exact match fine (no float issue)
    _type = "Count",
        SWITCH(
            TRUE(),
            _direction = "Higher Better", IF(_value >= _target, "Achieved", "Not Achieved"),
            _direction = "Lower Better", IF(_value <= _target, "Achieved", "Not Achieved"),
            "No Direction Set"
        ),

    -- Percentage: float, use ROUND to avoid floating-point precision issues (e.g. 0.899999999 vs 0.90)
    _type = "Percentage",
        SWITCH(
            TRUE(),
            _direction = "Higher Better", IF(ROUND(_value,4) >= ROUND(_target,4), "Achieved", "Not Achieved"),
            _direction = "Lower Better", IF(ROUND(_value,4) <= ROUND(_target,4), "Achieved", "Not Achieved"),
            "No Direction Set"
        ),

    "Type Not Set"
)
