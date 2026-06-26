KPI Met Flag = 
VAR ActualVal = MAX(fact_quality[Value])
VAR TargetVal = MAX(dim_client_kpi_config[Target])
VAR Direction = SELECTEDVALUE(dim_kpi_master[Direction])
VAR KPIType = SELECTEDVALUE(fact_quality[KPI Type])

RETURN
SWITCH(
    TRUE(),
    KPIType = "Binary",
        IF(ActualVal >= TargetVal, 1, 0),
    KPIType = "Percentage" && Direction = "Higher Better",
        IF(ActualVal >= TargetVal, 1, 0),
    KPIType = "Percentage" && Direction = "Lower Better",
        IF(ActualVal <= TargetVal, 1, 0),
    KPIType = "Count" && Direction = "Lower Better",
        IF(ActualVal <= TargetVal, 1, 0),
    KPIType = "Count" && Direction = "Higher Better",
        IF(ActualVal >= TargetVal, 1, 0)
)



Met Count = 
CALCULATE(
    COUNTROWS(fact_quality),
    [KPI Met Flag] = 1
)


Not Met Count = 
CALCULATE(
    COUNTROWS(fact_quality),
    [KPI Met Flag] = 0
)


Overall Compliance % = 
DIVIDE(
    [Met Count],
    [Met Count] + [Not Met Count],
    0
)


Client Avg Performance % = 
CALCULATE(
    AVERAGEX(
        VALUES(fact_quality[KPI_ID]),
        AVERAGEX(
            VALUES(fact_quality[Date_1]),
            MAX(fact_quality[Value])
        )
    ),
    fact_quality[KPI Type] = "Percentage"
)
