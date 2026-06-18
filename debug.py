Avg KPI Value Smart = 
VAR KPIType = SELECTEDVALUE(fact_quality[KPI Type])
RETURN
SWITCH(
    KPIType,
    "Percentage", AVERAGEX(
                    VALUES(fact_quality[Date_1]),
                    [KPI Value Trend %]
                  ),
    "Binary",     AVERAGEX(
                    VALUES(fact_quality[Date_1]),
                    [KPI Value Trend Binary]
                  ),
    "Count",      AVERAGEX(
                    VALUES(fact_quality[Date_1]),
                    [KPI Value Trend Count]
                  )
)
