Value Aggregated = 
VAR _type = SELECTEDVALUE(fact_quality[KPI Type])
RETURN
SWITCH(
    TRUE(),
    _type = "Count", SUM(fact_quality[Value]),
    _type = "Percentage", AVERAGE(fact_quality[Value]),
    _type = "Binary", AVERAGE(fact_quality[Value]),  -- shows as rate, format as %
    BLANK()
)
Target Aggregated = 
VAR _type = SELECTEDVALUE(fact_quality[KPI Type])
VAR _minTarget = MIN(fact_quality[Target])
VAR _maxTarget = MAX(fact_quality[Target])
RETURN
SWITCH(
    TRUE(),
    _type = "Count", _minTarget,
    _type = "Percentage", _minTarget,
    _type = "Binary", 1,  -- Binary target is always "met" = 1, no need to pull from data
    BLANK()
)
Target Inconsistent Flag = 
VAR _minTarget = MIN(fact_quality[Target])
VAR _maxTarget = MAX(fact_quality[Target])
RETURN
IF(_minTarget <> _maxTarget, "⚠ Target Varies", "OK")


Achievement Gap = 
[Value Aggregated] - [Target Aggregated]
