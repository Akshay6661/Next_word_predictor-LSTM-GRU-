Avg KPI Target Smart = 
VAR KPIType = SELECTEDVALUE(fact_quality[KPI Type])
RETURN
SWITCH(
    KPIType,
    "Percentage", AVERAGEX(
                    VALUES(fact_quality[Date_1]),
                    [KPI Target Trend %]
                  ),
    "Binary",     AVERAGEX(
                    VALUES(fact_quality[Date_1]),
                    [KPI Target Trend Binary]
                  ),
    "Count",      MAX([KPI Target Trend Count])
)


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
    "Count",      SUMX(
                    VALUES(fact_quality[Date_1]),
                    [KPI Value Trend Count]
                  )
)
