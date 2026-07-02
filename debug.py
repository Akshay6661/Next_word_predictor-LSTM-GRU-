Forecast Smart = 
SUMX(
    VALUES(Dim_SOW[SOW_Key]),
    VAR Frequency = MAX(Dim_SOW[Sow_Frequency])
    VAR YearlyForecast = SUM(Dim_SOW[Volume_Forecast])
    VAR MonthlyForecast = CALCULATE(SUM(Fact_Forecast[Volume_Forecast]))
    RETURN
    IF(
        ISBLANK(YearlyForecast) && ISBLANK(MonthlyForecast),
        BLANK(),
        IF(
            Frequency = "Yearly",
            YearlyForecast,
            MonthlyForecast
        )
    )
)
