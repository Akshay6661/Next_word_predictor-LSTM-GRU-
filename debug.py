Achievement Status Calc = 
VAR _kpiid = fact_quality[KPI_ID]
VAR _type = LOOKUPVALUE(dim_kpi_master[KPI Type], dim_kpi_master[KPI_ID], _kpiid)
VAR _direction = LOOKUPVALUE(dim_kpi_master[Direction], dim_kpi_master[KPI_ID], _kpiid)
VAR _value = fact_quality[Value]
VAR _target = LOOKUPVALUE(dim_client_kpi_config[Target], dim_client_kpi_config[KPI_ID], _kpiid)
RETURN
SWITCH(
    TRUE(),
    _type = "Binary",
        SWITCH(TRUE(), _value = 1, "Achieved", _value = 0, "Not Achieved", "Invalid Binary Value"),
    _type = "Count",
        SWITCH(TRUE(),
            _direction = "Higher Better", IF(_value >= _target, "Achieved", "Not Achieved"),
            _direction = "Lower Better", IF(_value <= _target, "Achieved", "Not Achieved"),
            "No Direction Set"),
    _type = "Percentage",
        SWITCH(TRUE(),
            _direction = "Higher Better", IF(ROUND(_value,4) >= ROUND(_target,4), "Achieved", "Not Achieved"),
            _direction = "Lower Better", IF(ROUND(_value,4) <= ROUND(_target,4), "Achieved", "Not Achieved"),
            "No Direction Set"),
    "Type Not Set"
)
