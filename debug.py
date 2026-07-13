Achievement Status Calc = 
VAR _kpiid = fact_quality[KPI_ID]
VAR _clientkey = fact_quality[Client_Key]
VAR _type = fact_quality[KPI Type]
VAR _value = fact_quality[Value]
VAR _direction = LOOKUPVALUE(dim_kpi_master[Direction], dim_kpi_master[KPI_ID], _kpiid)
VAR _target = LOOKUPVALUE(
                dim_client_kpi_config[Target], 
                dim_client_kpi_config[KPI_ID], _kpiid,
                dim_client_kpi_config[Client_Key], _clientkey
              )
VAR _defaultTarget = LOOKUPVALUE(dim_kpi_master[Default Target], dim_kpi_master[KPI_ID], _kpiid)
VAR _effectiveTarget = COALESCE(_target, _defaultTarget)
RETURN
SWITCH(
    TRUE(),
    _type = "Binary",
        SWITCH(TRUE(), _value = 1, "Achieved", _value = 0, "Not Achieved", "Invalid Binary Value"),
    _type = "Count",
        SWITCH(TRUE(),
            _direction = "Higher Better", IF(_value >= _effectiveTarget, "Achieved", "Not Achieved"),
            _direction = "Lower Better", IF(_value <= _effectiveTarget, "Achieved", "Not Achieved"),
            "No Direction Set"),
    _type = "Percentage",
        SWITCH(TRUE(),
            _direction = "Higher Better", IF(ROUND(_value,4) >= ROUND(_effectiveTarget,4), "Achieved", "Not Achieved"),
            _direction = "Lower Better", IF(ROUND(_value,4) <= ROUND(_effectiveTarget,4), "Achieved", "Not Achieved"),
            "No Direction Set"),
    "Type Not Set"
)
