Avg KPI Value Smart = 
SWITCH(
    SELECTEDVALUE(fact_quality[KPI Type]),
    "Percentage", AVERAGEX(VALUES(fact_quality[Date_1]), MAX(fact_quality[Value])),
    "Binary", AVERAGEX(VALUES(fact_quality[Date_1]), MAX(fact_quality[Value])),
    "Count", SUMX(VALUES(fact_quality[Date_1]), MAX(fact_quality[Value]))
)


Avg KPI Target Smart = 
SWITCH(
    SELECTEDVALUE(fact_quality[KPI Type]),
    "Percentage", AVERAGEX(
                    VALUES(fact_quality[Date_1]), 
                    MAX(dim_client_kpi_config[Target])
                 ),
    "Binary",     AVERAGEX(
                    VALUES(fact_quality[Date_1]), 
                    MAX(dim_client_kpi_config[Target])
                 ),
    "Count",      MAX(dim_client_kpi_config[Target])
)
