Avg KPI Value % = 
IF(
    SELECTEDVALUE(fact_quality[KPI Type]) = "Percentage",
    AVERAGEX(
        VALUES(fact_quality[Date_1]),
        MAX(fact_quality[Value])
    )
)


Avg KPI Target % = 
IF(
    SELECTEDVALUE(fact_quality[KPI Type]) = "Percentage",
    AVERAGEX(
        VALUES(fact_quality[Date_1]),
        MAX(dim_client_kpi_config[Target])
    )
)
