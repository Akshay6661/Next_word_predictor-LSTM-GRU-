Last Missed Date = 
CALCULATE(
    MAX(fact_quality[Date]),
    fact_quality[Achievement Status Calc] = "Not Achieved"
)


Consecutive Misses = 
VAR _currentDate = MAX(fact_quality[Date])
VAR _clientKPI = MAX(fact_quality[KPI_ID]) & MAX(fact_quality[Client_Key])
RETURN
CALCULATE(
    COUNTROWS(fact_quality),
    fact_quality[Achievement Status Calc] = "Not Achieved",
    fact_quality[Date] <= _currentDate,
    ALLEXCEPT(fact_quality, fact_quality[Client_Key], fact_quality[KPI_ID])
)
